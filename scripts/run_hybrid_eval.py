"""v22: Hybrid scorer evaluation across SWE-agent, ATBench, and synthetic datasets.

Evaluates hybrid scoring (LSTM NLL + heuristic features) under conformal
prediction, with transfer learning, multi-seed validation, and within-hybrid
ablation (leave-one-feature-out).

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_hybrid_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_hybrid_eval.py --seeds 42 123 456 789 1024
    PYTHONPATH=../../lib:src uv run python scripts/run_hybrid_eval.py --10seed
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from trustlens.hybrid import SUPERVISED_MODES, HybridScorer, build_feature_dict
from trustlens.stats import bootstrap_ci, paired_test

OUTPUT_DIR = Path("output/tables")
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
MIN_TRACE_LEN = 3
MAX_TRACE_LEN = 200


# ---------------------------------------------------------------------------
# Data loading (reused patterns from existing eval scripts)
# ---------------------------------------------------------------------------


def load_swe_agent_data(max_traces: int = 20000) -> list[dict]:
    """Load SWE-agent trajectories from HuggingFace (streaming)."""
    import re

    from datasets import load_dataset

    print(f"Loading up to {max_traces} SWE-agent trajectories (streaming)...")
    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split="train",
        streaming=True,
    )

    traces = []
    skipped = 0
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions = []
        for turn in example["trajectory"]:
            if turn.get("role") == "ai" and turn.get("text"):
                code_blocks = re.findall(
                    r"```\s*\n?(.*?)\n?```", turn["text"], re.DOTALL,
                )
                for block in code_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    first_line = block.split("\n")[0].strip()
                    cmd = first_line.split()[0] if first_line.split() else ""
                    cmd = cmd.lstrip("./!$")
                    if cmd:
                        actions.append(cmd)
        if len(actions) < MIN_TRACE_LEN:
            skipped += 1
            continue
        if len(actions) > MAX_TRACE_LEN:
            actions = actions[:MAX_TRACE_LEN]
        traces.append({
            "instance_id": example["instance_id"],
            "actions": actions,
            "solved": bool(example["target"]),
            "n_actions": len(actions),
        })
        if len(traces) % 2000 == 0:
            solved = sum(1 for t in traces if t["solved"])
            print(f"  Loaded {len(traces)} ({solved} solved)")

    solved = sum(1 for t in traces if t["solved"])
    print(f"Loaded {len(traces)} traces ({solved} solved, {skipped} skipped)")
    return traces


def load_atbench_traces() -> list[dict]:
    """Load ATBench and convert to evaluation-ready dicts."""
    from data.trace_loader import load_atbench

    print("Loading ATBench dataset...")
    traces = load_atbench(max_traces=None)
    result = []
    for t in traces:
        actions = [a.tool for a in t.actions]
        if len(actions) < 2:
            continue
        if len(actions) > 50:
            actions = actions[:50]
        risk_source = ""
        if t.task.startswith("risk="):
            parts = t.task.split(" mode=")
            risk_source = parts[0].replace("risk=", "")
        result.append({
            "trace_id": t.trace_id,
            "actions": actions,
            "safe": not t.has_anomaly,
            "label": 1 if t.has_anomaly else 0,
            "risk_source": risk_source,
            "n_actions": len(actions),
        })
    safe = sum(1 for r in result if r["safe"])
    print(f"Loaded {len(result)} traces ({safe} safe, {len(result) - safe} unsafe)")
    return result


def load_synthetic_traces() -> list[dict]:
    """Load synthetic traces from local JSONL files."""
    import orjson

    result = []
    for split_name, path in [("train", "data/train.jsonl"), ("test", "data/test.jsonl")]:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for line in f:
                obj = orjson.loads(line)
                raw_actions = obj.get("tools", obj.get("actions", []))
                # Normalize: actions may be dicts with "tool" key or plain strings
                actions = [
                    a["tool"] if isinstance(a, dict) else str(a)
                    for a in raw_actions
                ]
                if len(actions) >= MIN_TRACE_LEN:
                    # Trace-level label: from explicit field or derived from per-action labels
                    label = obj.get("label", obj.get("anomaly", None))
                    if label is None:
                        label = int(any(
                            a.get("label", 0) == 1
                            for a in raw_actions
                            if isinstance(a, dict)
                        ))
                    result.append({
                        "trace_id": obj.get("trace_id", f"{split_name}_{len(result)}"),
                        "actions": actions[:MAX_TRACE_LEN],
                        "label": int(label),
                        "n_actions": min(len(actions), MAX_TRACE_LEN),
                        "split": split_name,
                    })
    normal = sum(1 for r in result if r["label"] == 0)
    print(f"Loaded {len(result)} synthetic traces ({normal} normal, "
          f"{len(result) - normal} anomalous)")
    return result


# ---------------------------------------------------------------------------
# Vocabulary, encoding, model, training, scoring
# ---------------------------------------------------------------------------


def build_vocabulary(traces: list[dict], min_freq: int = 2) -> dict[str, int]:
    """Build vocabulary from action sequences."""
    counter: Counter[str] = Counter()
    for t in traces:
        counter.update(t["actions"])
    vocab = dict(SPECIAL_TOKENS)
    for cmd, freq in counter.most_common():
        if freq >= min_freq:
            vocab[cmd] = len(vocab)
    print(f"Vocabulary: {len(vocab)} tokens "
          f"({len(vocab) - len(SPECIAL_TOKENS)} commands + "
          f"{len(SPECIAL_TOKENS)} special)")
    return vocab


def encode_trace(
    actions: list[str], vocab: dict[str, int], max_len: int = MAX_TRACE_LEN,
) -> list[int]:
    unk = vocab["<UNK>"]
    ids = [vocab["<BOS>"]]
    for a in actions[:max_len]:
        ids.append(vocab.get(a, unk))
    ids.append(vocab["<EOS>"])
    return ids


class ActionSeqDataset(Dataset):
    def __init__(self, traces: list[dict], vocab: dict[str, int], labels: list[int]):
        self.encoded = [encode_trace(t["actions"], vocab) for t in traces]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.encoded[idx], dtype=torch.long), self.labels[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    seqs, labels = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded, torch.tensor(labels, dtype=torch.long)


class LSTMScorer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(
            d_model, d_model, n_layers, batch_first=True, dropout=0.1,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.head(out)


def train_scorer(
    train_traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 128,
) -> LSTMScorer:
    """Train LSTM scorer on normal action sequences."""
    model = LSTMScorer(len(vocab), d_model=d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ds = ActionSeqDataset(train_traces, vocab, [0] * len(train_traces))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for seqs, _ in loader:
            seqs = seqs.to(device)
            logits = model(seqs[:, :-1])
            targets = seqs[:, 1:]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch + 1}/{epochs}: loss={avg:.4f}")
    return model


def compute_ncs(
    model: LSTMScorer,
    traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-trace NLL scores."""
    model.eval()
    scores = []
    ds = ActionSeqDataset(traces, vocab, [0] * len(traces))
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    with torch.no_grad():
        for seqs, _ in loader:
            seqs = seqs.to(device)
            logits = model(seqs[:, :-1])
            targets = seqs[:, 1:]
            log_probs = torch.log_softmax(logits, dim=-1)
            mask = targets != 0
            token_nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            token_nll = token_nll * mask.float()
            seq_nll = token_nll.sum(dim=1) / mask.float().sum(dim=1).clamp(min=1)
            scores.extend(seq_nll.cpu().numpy().tolist())
    return np.array(scores)


def compute_simple_baselines(traces: list[dict]) -> dict[str, np.ndarray]:
    """Compute heuristic baseline scores."""
    n = len(traces)
    length = np.zeros(n)
    diversity = np.zeros(n)
    repeat = np.zeros(n)
    for i, t in enumerate(traces):
        actions = t["actions"]
        na = len(actions)
        length[i] = na
        diversity[i] = 1.0 - (len(set(actions)) / max(na, 1))
        if na > 1:
            reps = sum(1 for j in range(1, na) if actions[j] == actions[j - 1])
            repeat[i] = reps / (na - 1)
    return {"trace_length": length, "low_diversity": diversity, "repeat_rate": repeat}


# ---------------------------------------------------------------------------
# CP evaluation
# ---------------------------------------------------------------------------


def split_cp_evaluate(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Run split conformal prediction evaluation."""
    cal_normal = cal_scores[cal_labels == 0]
    n = len(cal_normal)
    if n == 0:
        return {"error": "no normal calibration samples", "n_cal": 0}
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    tau = float(np.quantile(cal_normal, min(q_level, 1.0)))
    test_pred = (test_scores > tau).astype(int)
    normal_mask = test_labels == 0
    coverage = float(1 - test_pred[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0
    auroc = float(roc_auc_score(test_labels, test_scores))
    f1 = float(f1_score(test_labels, test_pred))
    recall = float(test_pred[test_labels == 1].mean()) if (test_labels == 1).sum() > 0 else 0.0
    miscoverage_ratio = float((1 - coverage) / alpha) if alpha > 0 else 0.0
    # Standard error of coverage estimate (binomial proportion)
    n_normal_test = int(normal_mask.sum())
    se = float(np.sqrt(coverage * (1 - coverage) / max(n_normal_test, 1)))
    coverage_within_se = coverage >= (1 - alpha) - 1.96 * se

    return {
        "tau": tau,
        "coverage": coverage,
        "miscoverage_ratio": miscoverage_ratio,
        "coverage_within_se": coverage_within_se,
        "coverage_se": se,
        "auroc": auroc,
        "f1": f1,
        "recall": recall,
        "n_cal": n,
        "n_test": len(test_labels),
    }


# ---------------------------------------------------------------------------
# Hybrid scorer builder (fixes label leakage for supervised modes)
# ---------------------------------------------------------------------------


def evaluate_all_scorers(
    cal_features: dict[str, np.ndarray],
    cal_labels: np.ndarray,
    test_features: dict[str, np.ndarray],
    test_labels: np.ndarray,
    cal_lstm: np.ndarray,
    test_lstm: np.ndarray,
    cal_heur: dict[str, np.ndarray],
    test_heur: dict[str, np.ndarray],
    alpha: float,
    seed: int,
) -> dict:
    """Build all scorers and evaluate with proper cal splitting for supervised modes.

    Supervised scorers (fitted, rf, gbt) split cal into cal_fit + cal_cp:
    - cal_fit: train the classifier
    - cal_cp: compute CP threshold (no label leakage)

    Unsupervised scorers (lstm, hybrid_equal, heuristics) use the full cal set.
    """
    # Normal features for z-score normalizer (full cal normal — no label leakage)
    cal_normal_mask = cal_labels == 0
    cal_normal_features = {k: v[cal_normal_mask] for k, v in cal_features.items()}

    # --- Unsupervised scorers (full cal) ---
    hybrid_eq = HybridScorer(mode="equal")
    hybrid_eq.fit_normalizer(cal_normal_features)
    cal_hybrid_eq = hybrid_eq.score(cal_features)
    test_hybrid_eq = hybrid_eq.score(test_features)

    unsupervised = {
        "lstm": (cal_lstm, cal_labels, test_lstm),
        "hybrid_equal": (cal_hybrid_eq, cal_labels, test_hybrid_eq),
    }
    for hname in cal_heur:
        unsupervised[hname] = (cal_heur[hname], cal_labels, test_heur[hname])

    # --- Supervised scorers: split cal into cal_fit + cal_cp ---
    n_cal = len(cal_labels)
    n_fit = n_cal // 2
    idx = np.arange(n_cal)
    rng_split = np.random.RandomState(seed + 9999)
    rng_split.shuffle(idx)
    fit_idx, cp_idx = idx[:n_fit], idx[n_fit:]

    cal_fit_features = {k: v[fit_idx] for k, v in cal_features.items()}
    cal_fit_labels = cal_labels[fit_idx]
    cal_cp_features = {k: v[cp_idx] for k, v in cal_features.items()}
    cal_cp_labels = cal_labels[cp_idx]

    supervised = {}
    for mode in SUPERVISED_MODES:
        scorer = HybridScorer(mode=mode)
        scorer.fit_normalizer(cal_normal_features)
        scorer.fit_classifier(cal_fit_features, cal_fit_labels)
        cal_cp_scores = scorer.score(cal_cp_features)
        test_scores = scorer.score(test_features)
        supervised[f"hybrid_{mode}"] = (cal_cp_scores, cal_cp_labels, test_scores)

    # --- Evaluate all ---
    results: dict = {}
    all_scorers = {**unsupervised, **supervised}
    for sname, (cal_s, cal_l, test_s) in all_scorers.items():
        results[sname] = split_cp_evaluate(cal_s, cal_l, test_s, test_labels, alpha)

    return results


# ---------------------------------------------------------------------------
# Within-hybrid ablation (leave-one-feature-out)
# ---------------------------------------------------------------------------

ALL_HYBRID_FEATURES = ("lstm_nll", "trace_length", "low_diversity", "repeat_rate")


def run_hybrid_ablation(
    cal_features: dict[str, np.ndarray],
    cal_labels: np.ndarray,
    test_features: dict[str, np.ndarray],
    test_labels: np.ndarray,
    alpha: float,
    seed: int,
) -> dict:
    """Leave-one-feature-out ablation for hybrid scorers.

    For each feature in {lstm_nll, trace_length, low_diversity, repeat_rate},
    evaluate hybrid_equal and hybrid_rf with that feature removed.
    """
    cal_normal_mask = cal_labels == 0
    ablation_results: dict = {}

    for drop_feat in ALL_HYBRID_FEATURES:
        keep = [f for f in ALL_HYBRID_FEATURES if f != drop_feat]
        cal_sub = {k: cal_features[k] for k in keep}
        test_sub = {k: test_features[k] for k in keep}
        cal_normal_sub = {k: v[cal_normal_mask] for k, v in cal_sub.items()}

        key_prefix = f"minus_{drop_feat}"

        # Unsupervised: equal-weight on remaining features
        hybrid_eq = HybridScorer(mode="equal")
        hybrid_eq.fit_normalizer(cal_normal_sub)
        cal_eq_scores = hybrid_eq.score(cal_sub)
        test_eq_scores = hybrid_eq.score(test_sub)
        ablation_results[f"{key_prefix}_equal"] = split_cp_evaluate(
            cal_eq_scores, cal_labels, test_eq_scores, test_labels, alpha,
        )

        # Supervised: random forest on remaining features
        n_cal = len(cal_labels)
        n_fit = n_cal // 2
        idx = np.arange(n_cal)
        rng_split = np.random.RandomState(seed + 7777)
        rng_split.shuffle(idx)
        fit_idx, cp_idx = idx[:n_fit], idx[n_fit:]

        cal_fit_sub = {k: v[fit_idx] for k, v in cal_sub.items()}
        cal_cp_sub = {k: v[cp_idx] for k, v in cal_sub.items()}
        cal_fit_labels = cal_labels[fit_idx]
        cal_cp_labels = cal_labels[cp_idx]

        scorer_rf = HybridScorer(mode="rf")
        scorer_rf.fit_normalizer(cal_normal_sub)
        scorer_rf.fit_classifier(cal_fit_sub, cal_fit_labels)
        cal_cp_scores = scorer_rf.score(cal_cp_sub)
        test_rf_scores = scorer_rf.score(test_sub)
        ablation_results[f"{key_prefix}_rf"] = split_cp_evaluate(
            cal_cp_scores, cal_cp_labels, test_rf_scores, test_labels, alpha,
        )

    return ablation_results


def aggregate_ablation_results(
    all_ablation: list[dict],
) -> dict:
    """Aggregate per-seed ablation results."""
    if not all_ablation:
        return {}
    all_keys = set()
    for ab in all_ablation:
        all_keys.update(ab.keys())

    agg = {}
    for key in sorted(all_keys):
        aurocs = [ab[key]["auroc"] for ab in all_ablation if key in ab]
        f1s = [ab[key]["f1"] for ab in all_ablation if key in ab]
        covs = [ab[key]["coverage"] for ab in all_ablation if key in ab]
        n_seeds = len(aurocs)
        if n_seeds == 0:
            continue
        agg[key] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "f1_mean": float(np.mean(f1s)),
            "coverage_mean": float(np.mean(covs)),
            "n_seeds": n_seeds,
        }
    return agg


# ---------------------------------------------------------------------------
# Single-seed hybrid evaluation
# ---------------------------------------------------------------------------


def run_swe_agent_seed(
    traces: list[dict],
    vocab: dict[str, int],
    seed: int,
    epochs: int,
    device: torch.device,
    alpha: float,
    d_model: int,
    transfer_model: LSTMScorer | None = None,
    transfer_vocab: dict[str, int] | None = None,
    finetune_epochs: int = 10,
    run_ablation: bool = False,
) -> dict:
    """Run one seed on SWE-agent: LSTM, heuristics, hybrid (equal + fitted)."""
    set_seeds(seed)
    rng = np.random.RandomState(seed)

    solved = [t for t in traces if t["solved"]]
    failed = [t for t in traces if not t["solved"]]
    rng.shuffle(solved)
    rng.shuffle(failed)

    ns, nf = len(solved), len(failed)
    nts, ntf = int(0.6 * ns), int(0.6 * nf)
    ncs, ncf = int(0.2 * ns), int(0.2 * nf)

    train_traces = solved[:nts] + failed[:ntf]
    cal_traces = solved[nts:nts + ncs] + failed[ntf:ntf + ncf]
    test_traces = solved[nts + ncs:] + failed[ntf + ncf:]
    rng.shuffle(train_traces)
    rng.shuffle(cal_traces)
    rng.shuffle(test_traces)

    cal_labels = np.array([0 if t["solved"] else 1 for t in cal_traces])
    test_labels = np.array([0 if t["solved"] else 1 for t in test_traces])
    train_normal = [t for t in train_traces if t["solved"]]

    print(f"\n  Seed {seed}: train={len(train_traces)} "
          f"({len(train_normal)} normal), "
          f"cal={len(cal_traces)}, test={len(test_traces)}")

    # --- Train LSTM (from scratch, normal-only) ---
    model = train_scorer(train_normal, vocab, device, epochs=epochs, d_model=d_model)
    cal_lstm = compute_ncs(model, cal_traces, vocab, device)
    test_lstm = compute_ncs(model, test_traces, vocab, device)

    # --- Transfer learning (if pre-trained model provided) ---
    transfer_result = None
    if transfer_model is not None and transfer_vocab is not None:
        transfer_cal = compute_ncs(transfer_model, cal_traces, transfer_vocab, device)
        transfer_test = compute_ncs(transfer_model, test_traces, transfer_vocab, device)
        transfer_result = {
            "zero_shot": split_cp_evaluate(
                transfer_cal, cal_labels, transfer_test, test_labels, alpha,
            ),
        }

    # --- Heuristic baselines ---
    cal_heur = compute_simple_baselines(cal_traces)
    test_heur = compute_simple_baselines(test_traces)

    # --- Build hybrid features ---
    cal_features = build_feature_dict(cal_lstm, cal_heur)
    test_features = build_feature_dict(test_lstm, test_heur)

    # --- Evaluate all scorers (with cal split for supervised modes) ---
    results = {"seed": seed}
    scorer_results = evaluate_all_scorers(
        cal_features, cal_labels, test_features, test_labels,
        cal_lstm, test_lstm, cal_heur, test_heur, alpha, seed,
    )
    results.update(scorer_results)

    if transfer_result:
        results["transfer"] = transfer_result

    if run_ablation:
        results["ablation"] = run_hybrid_ablation(
            cal_features, cal_labels, test_features, test_labels, alpha, seed,
        )

    return results


def run_atbench_seed(
    traces: list[dict],
    vocab: dict[str, int],
    seed: int,
    epochs: int,
    device: torch.device,
    alpha: float,
    d_model: int,
    run_ablation: bool = False,
) -> dict:
    """Run one seed on ATBench."""
    set_seeds(seed)
    rng = np.random.RandomState(seed)

    safe = [t for t in traces if t["safe"]]
    unsafe = [t for t in traces if not t["safe"]]
    rng.shuffle(safe)
    rng.shuffle(unsafe)

    ns, nu = len(safe), len(unsafe)
    nts, ntu = int(0.6 * ns), int(0.6 * nu)
    ncs, ncu = int(0.2 * ns), int(0.2 * nu)

    train_traces = safe[:nts] + unsafe[:ntu]
    cal_traces = safe[nts:nts + ncs] + unsafe[ntu:ntu + ncu]
    test_traces = safe[nts + ncs:] + unsafe[ntu + ncu:]
    rng.shuffle(train_traces)
    rng.shuffle(cal_traces)
    rng.shuffle(test_traces)

    cal_labels = np.array([t["label"] for t in cal_traces])
    test_labels = np.array([t["label"] for t in test_traces])
    train_safe = [t for t in train_traces if t["safe"]]

    print(f"\n  ATBench Seed {seed}: train={len(train_safe)} safe, "
          f"cal={len(cal_traces)}, test={len(test_traces)}")

    model = train_scorer(train_safe, vocab, device, epochs=epochs, d_model=d_model)
    cal_lstm = compute_ncs(model, cal_traces, vocab, device)
    test_lstm = compute_ncs(model, test_traces, vocab, device)

    cal_heur = compute_simple_baselines(cal_traces)
    test_heur = compute_simple_baselines(test_traces)

    cal_features = build_feature_dict(cal_lstm, cal_heur)
    test_features = build_feature_dict(test_lstm, test_heur)

    results = {"seed": seed}
    scorer_results = evaluate_all_scorers(
        cal_features, cal_labels, test_features, test_labels,
        cal_lstm, test_lstm, cal_heur, test_heur, alpha, seed,
    )
    results.update(scorer_results)

    if run_ablation:
        results["ablation"] = run_hybrid_ablation(
            cal_features, cal_labels, test_features, test_labels, alpha, seed,
        )

    return results


def run_synthetic_seed(
    traces: list[dict],
    vocab: dict[str, int],
    seed: int,
    epochs: int,
    device: torch.device,
    alpha: float,
    d_model: int,
    run_ablation: bool = False,
) -> dict:
    """Run one seed on synthetic data."""
    set_seeds(seed)
    rng = np.random.RandomState(seed)

    normal = [t for t in traces if t["label"] == 0]
    anomalous = [t for t in traces if t["label"] == 1]
    rng.shuffle(normal)
    rng.shuffle(anomalous)

    nn_count, na_count = len(normal), len(anomalous)
    ntn, nta = int(0.6 * nn_count), int(0.6 * na_count)
    ncn, nca = int(0.2 * nn_count), int(0.2 * na_count)

    train_traces = normal[:ntn] + anomalous[:nta]
    cal_traces = normal[ntn:ntn + ncn] + anomalous[nta:nta + nca]
    test_traces = normal[ntn + ncn:] + anomalous[nta + nca:]
    rng.shuffle(train_traces)
    rng.shuffle(cal_traces)
    rng.shuffle(test_traces)

    cal_labels = np.array([t["label"] for t in cal_traces])
    test_labels = np.array([t["label"] for t in test_traces])
    train_normal = [t for t in train_traces if t["label"] == 0]

    print(f"\n  Synthetic Seed {seed}: train={len(train_normal)} normal, "
          f"cal={len(cal_traces)}, test={len(test_traces)}")

    model = train_scorer(train_normal, vocab, device, epochs=epochs, d_model=d_model)
    cal_lstm = compute_ncs(model, cal_traces, vocab, device)
    test_lstm = compute_ncs(model, test_traces, vocab, device)

    cal_heur = compute_simple_baselines(cal_traces)
    test_heur = compute_simple_baselines(test_traces)

    cal_features = build_feature_dict(cal_lstm, cal_heur)
    test_features = build_feature_dict(test_lstm, test_heur)

    results = {"seed": seed}
    scorer_results = evaluate_all_scorers(
        cal_features, cal_labels, test_features, test_labels,
        cal_lstm, test_lstm, cal_heur, test_heur, alpha, seed,
    )
    results.update(scorer_results)

    if run_ablation:
        results["ablation"] = run_hybrid_ablation(
            cal_features, cal_labels, test_features, test_labels, alpha, seed,
        )

    return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_scorer_results(
    all_results: list[dict],
    scorer_names: list[str],
) -> dict:
    """Aggregate per-seed scorer results with bootstrap CIs."""
    agg = {}
    for sname in scorer_names:
        aurocs = [r[sname]["auroc"] for r in all_results if sname in r]
        f1s = [r[sname]["f1"] for r in all_results if sname in r]
        covs = [r[sname]["coverage"] for r in all_results if sname in r]
        misc_ratios = [
            r[sname]["miscoverage_ratio"]
            for r in all_results if sname in r
        ]
        cov_within_se = sum(
            1 for r in all_results
            if sname in r and r[sname].get("coverage_within_se", False)
        )
        n_seeds = len(aurocs)
        if n_seeds == 0:
            continue
        agg[sname] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auroc_ci": bootstrap_ci(np.array(aurocs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "coverage_mean": float(np.mean(covs)),
            "coverage_std": float(np.std(covs)),
            "mean_miscoverage_ratio": float(np.mean(misc_ratios)),
            "coverage_within_se_pct": 100 * cov_within_se / n_seeds,
            "n_seeds": n_seeds,
        }
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="v22: Hybrid scorer evaluation")
    parser.add_argument("--max-swe-traces", type=int, default=20000)
    parser.add_argument("--swe-epochs", type=int, default=30)
    parser.add_argument("--swe-d-model", type=int, default=128)
    parser.add_argument("--atbench-epochs", type=int, default=30)
    parser.add_argument("--atbench-d-model", type=int, default=64)
    parser.add_argument("--synthetic-epochs", type=int, default=30)
    parser.add_argument("--synthetic-d-model", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
    )
    parser.add_argument(
        "--10seed", dest="ten_seed", action="store_true",
        help="Run 10 seeds on SWE-agent for Wilcoxon validation",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Skip synthetic dataset evaluation",
    )
    parser.add_argument(
        "--skip-transfer", action="store_true",
        help="Skip transfer learning experiment",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run within-hybrid leave-one-feature-out ablation",
    )
    args = parser.parse_args()

    if args.ten_seed:
        args.seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds} ({len(args.seeds)} seeds)")
    print(f"Alpha: {args.alpha}")

    t0 = time.time()
    scorer_names = [
        "lstm", "hybrid_equal", "hybrid_fitted", "hybrid_rf", "hybrid_gbt",
        "trace_length", "low_diversity", "repeat_rate",
    ]

    output = {
        "config": {
            "seeds": args.seeds,
            "n_seeds": len(args.seeds),
            "alpha": args.alpha,
            "version": "v22",
        },
    }

    # ---------------------------------------------------------------
    # 1. SWE-agent evaluation
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SWE-agent Hybrid Evaluation")
    print(f"{'=' * 60}")

    swe_traces = load_swe_agent_data(args.max_swe_traces)
    swe_vocab = build_vocabulary(swe_traces, min_freq=5)

    # Optional: pre-train on synthetic for transfer learning
    transfer_model = None
    transfer_vocab = None
    if not args.skip_transfer:
        synth_traces = load_synthetic_traces()
        if synth_traces:
            transfer_vocab = build_vocabulary(synth_traces, min_freq=2)
            synth_normal = [t for t in synth_traces if t["label"] == 0]
            print(f"\n  Pre-training transfer model on {len(synth_normal)} "
                  f"synthetic normal traces...")
            set_seeds(42)
            transfer_model = train_scorer(
                synth_normal, transfer_vocab, device,
                epochs=args.synthetic_epochs, d_model=args.synthetic_d_model,
            )

    swe_results = []
    for seed in args.seeds:
        r = run_swe_agent_seed(
            swe_traces, swe_vocab, seed, args.swe_epochs, device, args.alpha,
            args.swe_d_model, transfer_model, transfer_vocab,
            run_ablation=args.ablation,
        )
        swe_results.append(r)
        print(f"  -> lstm={r['lstm']['auroc']:.4f}, "
              f"hybrid_eq={r['hybrid_equal']['auroc']:.4f}, "
              f"hybrid_fit={r['hybrid_fitted']['auroc']:.4f}, "
              f"rf={r['hybrid_rf']['auroc']:.4f}, "
              f"gbt={r['hybrid_gbt']['auroc']:.4f}, "
              f"best_heur="
              f"{max(r[h]['auroc'] for h in ('trace_length', 'low_diversity', 'repeat_rate')):.4f}")

    swe_agg = aggregate_scorer_results(swe_results, scorer_names)

    # Pairwise tests: hybrid_equal vs best individual
    best_heur_name = max(
        ["trace_length", "low_diversity", "repeat_rate"],
        key=lambda h: swe_agg.get(h, {}).get("auroc_mean", 0),
    )
    hybrid_aurocs = np.array(
        [r["hybrid_equal"]["auroc"] for r in swe_results],
    )
    best_heur_aurocs = np.array(
        [r[best_heur_name]["auroc"] for r in swe_results],
    )
    lstm_aurocs = np.array([r["lstm"]["auroc"] for r in swe_results])

    swe_paired = {
        "hybrid_vs_best_heuristic": paired_test(hybrid_aurocs, best_heur_aurocs),
        "hybrid_vs_lstm": paired_test(hybrid_aurocs, lstm_aurocs),
        "best_heuristic": best_heur_name,
    }

    print(f"\nSWE-agent Aggregate ({len(args.seeds)} seeds):")
    for sn in scorer_names:
        if sn in swe_agg:
            a = swe_agg[sn]
            print(f"  {sn}: AUROC={a['auroc_mean']:.4f}+/-{a['auroc_std']:.4f}, "
                  f"Coverage={a['coverage_mean']:.4f}, "
                  f"MiscovR={a['mean_miscoverage_ratio']:.3f}")

    swe_output: dict = {
        "config": {
            "max_traces": args.max_swe_traces,
            "epochs": args.swe_epochs,
            "d_model": args.swe_d_model,
        },
        "data_stats": {
            "n_traces": len(swe_traces),
            "n_solved": sum(1 for t in swe_traces if t["solved"]),
            "vocab_size": len(swe_vocab),
        },
        "aggregate": swe_agg,
        "paired_tests": swe_paired,
        "per_seed": swe_results,
    }
    if args.ablation:
        swe_ablations = [r["ablation"] for r in swe_results if "ablation" in r]
        swe_output["ablation_aggregate"] = aggregate_ablation_results(swe_ablations)
        print("\n  SWE-agent Ablation (leave-one-feature-out):")
        for k, v in swe_output["ablation_aggregate"].items():
            print(f"    {k}: AUROC={v['auroc_mean']:.4f}")
    output["swe_agent"] = swe_output

    # ---------------------------------------------------------------
    # 2. ATBench evaluation
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ATBench Hybrid Evaluation")
    print(f"{'=' * 60}")

    atb_traces = load_atbench_traces()
    atb_vocab = build_vocabulary(atb_traces, min_freq=2)

    atb_results = []
    for seed in args.seeds:
        r = run_atbench_seed(
            atb_traces, atb_vocab, seed, args.atbench_epochs, device,
            args.alpha, args.atbench_d_model,
            run_ablation=args.ablation,
        )
        atb_results.append(r)
        print(f"  -> lstm={r['lstm']['auroc']:.4f}, "
              f"hybrid_eq={r['hybrid_equal']['auroc']:.4f}, "
              f"hybrid_fit={r['hybrid_fitted']['auroc']:.4f}, "
              f"rf={r['hybrid_rf']['auroc']:.4f}, "
              f"gbt={r['hybrid_gbt']['auroc']:.4f}")

    atb_agg = aggregate_scorer_results(atb_results, scorer_names)

    print(f"\nATBench Aggregate ({len(args.seeds)} seeds):")
    for sn in scorer_names:
        if sn in atb_agg:
            a = atb_agg[sn]
            print(f"  {sn}: AUROC={a['auroc_mean']:.4f}+/-{a['auroc_std']:.4f}, "
                  f"Coverage={a['coverage_mean']:.4f}, "
                  f"MiscovR={a['mean_miscoverage_ratio']:.3f}")

    atb_output: dict = {
        "config": {
            "epochs": args.atbench_epochs,
            "d_model": args.atbench_d_model,
        },
        "data_stats": {
            "n_traces": len(atb_traces),
            "n_safe": sum(1 for t in atb_traces if t["safe"]),
            "vocab_size": len(atb_vocab),
        },
        "aggregate": atb_agg,
        "per_seed": atb_results,
    }
    if args.ablation:
        atb_ablations = [r["ablation"] for r in atb_results if "ablation" in r]
        atb_output["ablation_aggregate"] = aggregate_ablation_results(atb_ablations)
        print("\n  ATBench Ablation (leave-one-feature-out):")
        for k, v in atb_output["ablation_aggregate"].items():
            print(f"    {k}: AUROC={v['auroc_mean']:.4f}")
    output["atbench"] = atb_output

    # ---------------------------------------------------------------
    # 3. Synthetic evaluation (optional)
    # ---------------------------------------------------------------
    if not args.skip_synthetic:
        print(f"\n{'=' * 60}")
        print("Synthetic Hybrid Evaluation")
        print(f"{'=' * 60}")

        synth_traces = load_synthetic_traces()
        if synth_traces:
            synth_vocab = build_vocabulary(synth_traces, min_freq=2)

            synth_results = []
            for seed in args.seeds:
                r = run_synthetic_seed(
                    synth_traces, synth_vocab, seed, args.synthetic_epochs,
                    device, args.alpha, args.synthetic_d_model,
                    run_ablation=args.ablation,
                )
                synth_results.append(r)
                print(f"  -> lstm={r['lstm']['auroc']:.4f}, "
                      f"hybrid_eq={r['hybrid_equal']['auroc']:.4f}, "
                      f"hybrid_fit={r['hybrid_fitted']['auroc']:.4f}, "
                      f"rf={r['hybrid_rf']['auroc']:.4f}, "
                      f"gbt={r['hybrid_gbt']['auroc']:.4f}")

            synth_agg = aggregate_scorer_results(synth_results, scorer_names)

            print(f"\nSynthetic Aggregate ({len(args.seeds)} seeds):")
            for sn in scorer_names:
                if sn in synth_agg:
                    a = synth_agg[sn]
                    print(f"  {sn}: AUROC={a['auroc_mean']:.4f}+/-"
                          f"{a['auroc_std']:.4f}")

            synth_output: dict = {
                "config": {
                    "epochs": args.synthetic_epochs,
                    "d_model": args.synthetic_d_model,
                },
                "data_stats": {
                    "n_traces": len(synth_traces),
                    "n_normal": sum(1 for t in synth_traces if t["label"] == 0),
                    "vocab_size": len(synth_vocab),
                },
                "aggregate": synth_agg,
                "per_seed": synth_results,
            }
            if args.ablation:
                synth_ablations = [
                    r["ablation"] for r in synth_results if "ablation" in r
                ]
                synth_output["ablation_aggregate"] = aggregate_ablation_results(
                    synth_ablations,
                )
                print("\n  Synthetic Ablation (leave-one-feature-out):")
                for k, v in synth_output["ablation_aggregate"].items():
                    print(f"    {k}: AUROC={v['auroc_mean']:.4f}")
            output["synthetic"] = synth_output
        else:
            print("  No synthetic data found, skipping")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    output["elapsed_seconds"] = elapsed

    print(f"\n{'=' * 60}")
    print(f"v22 Hybrid Evaluation Summary ({elapsed:.0f}s total)")
    print(f"{'=' * 60}")

    for dataset_key, label in [
        ("swe_agent", "SWE-agent"),
        ("atbench", "ATBench"),
        ("synthetic", "Synthetic"),
    ]:
        ds = output.get(dataset_key)
        if not ds:
            continue
        agg = ds["aggregate"]
        print(f"\n{label}:")
        for sn in ["lstm", "hybrid_equal", "hybrid_fitted", "hybrid_rf",
                    "hybrid_gbt", "trace_length", "low_diversity", "repeat_rate"]:
            if sn in agg:
                a = agg[sn]
                print(f"  {sn:20s}: AUROC={a['auroc_mean']:.4f}+/-"
                      f"{a['auroc_std']:.4f}  Cov={a['coverage_mean']:.4f}  "
                      f"MiscovR={a['mean_miscoverage_ratio']:.3f}")

    # Highlight key result
    swe_hybrid = swe_agg.get("hybrid_equal", {})
    swe_best_h = swe_agg.get(best_heur_name, {})
    swe_lstm = swe_agg.get("lstm", {})
    print("\nKey result (SWE-agent):")
    print(f"  hybrid_equal AUROC:      {swe_hybrid.get('auroc_mean', 0):.4f}")
    print(f"  best heuristic ({best_heur_name}): "
          f"{swe_best_h.get('auroc_mean', 0):.4f}")
    print(f"  lstm AUROC:              {swe_lstm.get('auroc_mean', 0):.4f}")
    delta = swe_hybrid.get("auroc_mean", 0) - swe_best_h.get("auroc_mean", 0)
    print(f"  hybrid - best_heuristic: {delta:+.4f}")
    if "hybrid_vs_best_heuristic" in swe_paired:
        pt = swe_paired["hybrid_vs_best_heuristic"]
        print(f"  Paired test p={pt.get('p_value', 'N/A')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "hybrid_eval_v22.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
