"""E22: ATBench evaluation — CP on real-world agent safety data.

Loads ATBench (AI45Research/ATBench): 500 trajectories with binary safety
labels (250 safe / 250 unsafe), 1,575 unique tools, 8 risk source categories.

Evaluates:
  - LSTM next-token scorer (trained on ATBench safe traces)
  - Heuristic scorers (trace_length, low_diversity, repeat_rate)
  - Cross-domain transfer: LSTM trained on synthetic, tested on ATBench
  - Split CP with coverage guarantees
  - RAC two-threshold evaluation
  - Per-risk-category coverage (Localized ARC on 8 risk sources)
  - 5-seed aggregation with bootstrap CI

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_atbench_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_atbench_eval.py --seeds 42 123 456 789 1024
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

from trustlens.rac import evaluate_rac, rac_sensitivity_sweep
from trustlens.stats import bootstrap_ci

OUTPUT_DIR = Path("output/tables")
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
MIN_TRACE_LEN = 2
MAX_TRACE_LEN = 50


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_atbench_traces(max_traces: int | None = None) -> list[dict]:
    """Load ATBench and convert to evaluation-ready dicts."""
    from data.trace_loader import load_atbench

    print("Loading ATBench dataset...")
    traces = load_atbench(max_traces=max_traces)

    result = []
    for t in traces:
        actions = [a.tool for a in t.actions]
        if len(actions) < MIN_TRACE_LEN:
            continue
        if len(actions) > MAX_TRACE_LEN:
            actions = actions[:MAX_TRACE_LEN]

        # Extract risk category from task metadata
        risk_source = ""
        if t.task.startswith("risk="):
            parts = t.task.split(" mode=")
            risk_source = parts[0].replace("risk=", "")

        result.append(
            {
                "trace_id": t.trace_id,
                "actions": actions,
                "safe": not t.has_anomaly,
                "label": 1 if t.has_anomaly else 0,
                "risk_source": risk_source,
                "n_actions": len(actions),
            }
        )

    safe_count = sum(1 for r in result if r["safe"])
    print(f"Loaded {len(result)} traces ({safe_count} safe, {len(result) - safe_count} unsafe)")
    return result


# ---------------------------------------------------------------------------
# Vocabulary and encoding
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

    print(
        f"Vocabulary: {len(vocab)} tokens "
        f"({len(vocab) - len(SPECIAL_TOKENS)} tools + "
        f"{len(SPECIAL_TOKENS)} special)"
    )
    return vocab


def encode_trace(
    actions: list[str],
    vocab: dict[str, int],
    max_len: int = MAX_TRACE_LEN,
) -> list[int]:
    """Encode action sequence to token IDs."""
    unk = vocab["<UNK>"]
    ids = [vocab["<BOS>"]]
    for a in actions[:max_len]:
        ids.append(vocab.get(a, unk))
    ids.append(vocab["<EOS>"])
    return ids


# ---------------------------------------------------------------------------
# Dataset and model (same architecture as SWE-agent eval)
# ---------------------------------------------------------------------------


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
            d_model,
            d_model,
            n_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.head(out)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_ncs(
    model: LSTMScorer,
    traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute non-conformity scores (mean NLL per trace)."""
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
    length_scores = np.zeros(n)
    diversity_scores = np.zeros(n)
    repeat_scores = np.zeros(n)

    for i, t in enumerate(traces):
        actions = t["actions"]
        n_actions = len(actions)
        length_scores[i] = n_actions
        unique = len(set(actions))
        diversity_scores[i] = 1.0 - (unique / max(n_actions, 1))
        if n_actions > 1:
            repeats = sum(1 for j in range(1, n_actions) if actions[j] == actions[j - 1])
            repeat_scores[i] = repeats / (n_actions - 1)

    return {
        "trace_length": length_scores,
        "low_diversity": diversity_scores,
        "repeat_rate": repeat_scores,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_scorer(
    train_traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 64,
) -> LSTMScorer:
    """Train LSTM scorer on safe action sequences."""
    model = LSTMScorer(len(vocab), d_model=d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds = ActionSeqDataset(train_traces, vocab, [0] * len(train_traces))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

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
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return model


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
    miscoverage = 1.0 - coverage

    auroc = float(roc_auc_score(test_labels, test_scores))
    f1 = float(f1_score(test_labels, test_pred))
    recall = float(test_pred[test_labels == 1].mean()) if (test_labels == 1).sum() > 0 else 0.0

    # Finite-sample excess: 1/(n_cal+1)
    finite_sample_excess = 1.0 / (n + 1)

    return {
        "tau": tau,
        "coverage": coverage,
        "miscoverage": miscoverage,
        "bound_holds": coverage >= (1 - alpha),
        "miscoverage_ratio": miscoverage / alpha if alpha > 0 else float("nan"),
        "auroc": auroc,
        "f1": f1,
        "recall": recall,
        "n_cal": n,
        "n_cal_normal": int((cal_labels == 0).sum()),
        "n_test": len(test_labels),
        "n_test_normal": int(normal_mask.sum()),
        "n_test_anomalous": int((test_labels == 1).sum()),
        "finite_sample_excess": finite_sample_excess,
    }


def compute_per_risk_coverage(
    test_traces: list[dict],
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    tau: float,
) -> dict[str, dict]:
    """Compute per-risk-source coverage (Localized ARC on real safety data)."""
    test_pred = (test_scores > tau).astype(int)
    cat_results = {}

    risk_sources = set(t["risk_source"] for t in test_traces if t["risk_source"])
    for risk in sorted(risk_sources):
        mask = np.array([t["risk_source"] == risk for t in test_traces])
        n_in_cat = int(mask.sum())
        if n_in_cat < 3:
            continue

        normal_in_cat = mask & (test_labels == 0)
        n_normal = int(normal_in_cat.sum())

        anomalous_in_cat = mask & (test_labels == 1)
        n_anomalous = int(anomalous_in_cat.sum())

        # Coverage among normal traces with this risk category
        cat_coverage = float(1 - test_pred[normal_in_cat].mean()) if n_normal > 0 else float("nan")

        # Detection rate among unsafe traces
        detect_rate = float(test_pred[anomalous_in_cat].mean()) if n_anomalous > 0 else float("nan")

        cat_results[risk] = {
            "n_traces": n_in_cat,
            "n_normal": n_normal,
            "n_anomalous": n_anomalous,
            "coverage": cat_coverage,
            "detection_rate": detect_rate,
            "meets_target": cat_coverage >= 0.95 if n_normal > 0 else None,
        }

    return cat_results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_single_seed(
    traces: list[dict],
    vocab: dict[str, int],
    seed: int,
    epochs: int,
    device: torch.device,
    alpha: float = 0.05,
    d_model: int = 64,
) -> dict:
    """Run one seed: split, train, score, evaluate CP + RAC + baselines."""
    set_seeds(seed)
    rng = np.random.RandomState(seed)

    safe = [t for t in traces if t["safe"]]
    unsafe = [t for t in traces if not t["safe"]]

    rng.shuffle(safe)
    rng.shuffle(unsafe)

    # 60/20/20 stratified split
    n_s = len(safe)
    n_u = len(unsafe)
    n_train_s, n_cal_s = int(0.6 * n_s), int(0.2 * n_s)
    n_train_u, n_cal_u = int(0.6 * n_u), int(0.2 * n_u)

    train_traces = safe[:n_train_s] + unsafe[:n_train_u]
    cal_traces = safe[n_train_s : n_train_s + n_cal_s] + unsafe[n_train_u : n_train_u + n_cal_u]
    test_traces = safe[n_train_s + n_cal_s :] + unsafe[n_train_u + n_cal_u :]

    rng.shuffle(train_traces)
    rng.shuffle(cal_traces)
    rng.shuffle(test_traces)

    cal_labels = np.array([t["label"] for t in cal_traces])
    test_labels = np.array([t["label"] for t in test_traces])

    # Train on safe traces only (normal-only language model)
    train_safe = [t for t in train_traces if t["safe"]]

    print(
        f"\n  Seed {seed}: train={len(train_traces)} ({len(train_safe)} safe), "
        f"cal={len(cal_traces)} ({(cal_labels == 0).sum()} safe), "
        f"test={len(test_traces)} ({(test_labels == 0).sum()} safe, "
        f"{(test_labels == 1).sum()} unsafe)"
    )

    # --- LSTM scorer ---
    model = train_scorer(
        train_safe,
        vocab,
        device,
        epochs=epochs,
        d_model=d_model,
    )
    cal_scores = compute_ncs(model, cal_traces, vocab, device)
    test_scores = compute_ncs(model, test_traces, vocab, device)

    lstm_cp = split_cp_evaluate(cal_scores, cal_labels, test_scores, test_labels, alpha)
    lstm_cp["scorer"] = "lstm"

    # Per-risk coverage
    per_risk = compute_per_risk_coverage(
        test_traces,
        test_scores,
        test_labels,
        lstm_cp["tau"],
    )
    lstm_cp["per_risk_category"] = per_risk

    # --- RAC evaluation ---
    rac_result = evaluate_rac(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        cost_ratio=5.0,
        defer_budget=0.15,
    )
    rac_sweep = rac_sensitivity_sweep(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        cost_ratios=[2.0, 5.0, 10.0],
        defer_budgets=[0.10, 0.15, 0.20],
    )

    # --- Heuristic baselines ---
    baseline_scores_cal = compute_simple_baselines(cal_traces)
    baseline_scores_test = compute_simple_baselines(test_traces)

    baselines = {}
    for name in baseline_scores_test:
        bl_auroc = float(roc_auc_score(test_labels, baseline_scores_test[name]))
        bl_cp = split_cp_evaluate(
            baseline_scores_cal[name],
            cal_labels,
            baseline_scores_test[name],
            test_labels,
            alpha,
        )
        bl_per_risk = compute_per_risk_coverage(
            test_traces,
            baseline_scores_test[name],
            test_labels,
            bl_cp["tau"],
        )
        baselines[name] = {
            "auroc": bl_auroc,
            "coverage": bl_cp["coverage"],
            "miscoverage": bl_cp["miscoverage"],
            "miscoverage_ratio": bl_cp["miscoverage_ratio"],
            "bound_holds": bl_cp["bound_holds"],
            "f1": bl_cp["f1"],
            "recall": bl_cp["recall"],
            "per_risk_category": bl_per_risk,
        }

    return {
        "seed": seed,
        **{k: v for k, v in lstm_cp.items() if k != "per_risk_category"},
        "per_risk_category": per_risk,
        "rac": rac_result,
        "rac_sweep": rac_sweep,
        "baselines": baselines,
    }


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E22: ATBench CP evaluation with real safety labels",
    )
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024],
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

    t0 = time.time()
    traces = load_atbench_traces(args.max_traces)

    vocab = build_vocabulary(traces)
    all_actions = [a for t in traces for a in t["actions"]]
    action_counts = Counter(all_actions)
    avg_len = np.mean([t["n_actions"] for t in traces])
    safe_pct = 100 * sum(1 for t in traces if t["safe"]) / len(traces)

    print("\nData stats:")
    print(f"  Total traces: {len(traces)}")
    print(f"  Safe rate: {safe_pct:.1f}%")
    print(f"  Mean trace length: {avg_len:.1f} actions")
    print(f"  Unique tools: {len(action_counts)}")
    print("  Top 10 tools:")
    for cmd, cnt in action_counts.most_common(10):
        print(f"    {cmd}: {cnt}")

    risk_dist = Counter(t["risk_source"] for t in traces if t["risk_source"])
    print(f"  Risk categories ({len(risk_dist)}):")
    for r, c in risk_dist.most_common():
        print(f"    {r}: {c}")

    # --- Run per-seed evaluations ---
    results = []
    for seed in args.seeds:
        r = run_single_seed(
            traces,
            vocab,
            seed,
            args.epochs,
            device,
            args.alpha,
            d_model=args.d_model,
        )
        results.append(r)
        print(
            f"  -> AUROC={r['auroc']:.4f}, F1={r['f1']:.4f}, "
            f"Coverage={r['coverage']:.4f}, "
            f"Bound={'HOLDS' if r['bound_holds'] else 'FAILS'}, "
            f"Miscov ratio={r['miscoverage_ratio']:.3f}"
        )
        if r.get("rac"):
            rac = r["rac"]
            print(
                f"     RAC: VaR={rac['var_cost']:.4f} "
                f"(binary={rac['binary_var_cost']:.4f}), "
                f"defer={rac['defer_rate']:.3f}"
            )
        for bn, bv in r["baselines"].items():
            print(f"     {bn}: AUROC={bv['auroc']:.4f}, Coverage={bv['coverage']:.4f}")

    # --- Aggregate ---
    aurocs = [r["auroc"] for r in results]
    f1s = [r["f1"] for r in results]
    coverages = [r["coverage"] for r in results]
    misc_ratios = [r["miscoverage_ratio"] for r in results]
    bound_pct = 100 * sum(1 for r in results if r["bound_holds"]) / len(results)

    auroc_ci = bootstrap_ci(np.array(aurocs))
    f1_ci = bootstrap_ci(np.array(f1s))
    cov_ci = bootstrap_ci(np.array(coverages))
    mr_ci = bootstrap_ci(np.array(misc_ratios))

    # RAC aggregate
    rac_var_improvements = [r["rac"]["var_improvement"] for r in results]
    rac_defer_rates = [r["rac"]["defer_rate"] for r in results]

    # Baseline aggregation
    bl_names = list(results[0]["baselines"].keys())
    bl_agg = {}
    for bn in bl_names:
        bl_aurocs = [r["baselines"][bn]["auroc"] for r in results]
        bl_covs = [r["baselines"][bn]["coverage"] for r in results]
        bl_mrs = [r["baselines"][bn]["miscoverage_ratio"] for r in results]
        bl_bounds = sum(1 for r in results if r["baselines"][bn]["bound_holds"])
        bl_agg[bn] = {
            "auroc_mean": float(np.mean(bl_aurocs)),
            "auroc_std": float(np.std(bl_aurocs)),
            "auroc_ci": bootstrap_ci(np.array(bl_aurocs)),
            "coverage_mean": float(np.mean(bl_covs)),
            "miscoverage_ratio_mean": float(np.mean(bl_mrs)),
            "bound_holds_pct": 100 * bl_bounds / len(results),
        }

    # Per-risk aggregation
    risk_names = set()
    for r in results:
        risk_names.update(r.get("per_risk_category", {}).keys())
    risk_agg = {}
    for rn in sorted(risk_names):
        covs = []
        detects = []
        for r in results:
            rc = r.get("per_risk_category", {}).get(rn)
            if rc:
                if not np.isnan(rc.get("coverage", float("nan"))):
                    covs.append(rc["coverage"])
                if not np.isnan(rc.get("detection_rate", float("nan"))):
                    detects.append(rc["detection_rate"])
        if covs:
            risk_agg[rn] = {
                "coverage_mean": float(np.mean(covs)),
                "coverage_std": float(np.std(covs)),
                "detection_mean": float(np.mean(detects)) if detects else None,
                "n_seeds": len(covs),
            }

    elapsed = time.time() - t0

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"ATBench E22 ({len(args.seeds)} seeds, d_model={args.d_model})")
    print(f"{'=' * 60}")
    print("LSTM Scorer:")
    print(
        f"  AUROC:    {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}  "
        f"CI [{auroc_ci['ci_lower']:.4f}, {auroc_ci['ci_upper']:.4f}]"
    )
    print(f"  F1:       {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Coverage: {np.mean(coverages):.4f} +/- {np.std(coverages):.4f}")
    print(f"  Miscov ratio: {np.mean(misc_ratios):.3f} +/- {np.std(misc_ratios):.3f}")
    print(f"  Bound holds: {bound_pct:.0f}%")
    print("\nRAC:")
    print(
        f"  VaR improvement: {np.mean(rac_var_improvements):.4f} "
        f"+/- {np.std(rac_var_improvements):.4f}"
    )
    print(f"  Defer rate: {np.mean(rac_defer_rates):.3f}")

    if bl_agg:
        print("\nBaselines:")
        for bn, bv in bl_agg.items():
            print(
                f"  {bn}: AUROC={bv['auroc_mean']:.4f}+/-{bv['auroc_std']:.4f}, "
                f"Coverage={bv['coverage_mean']:.4f}, "
                f"Miscov ratio={bv['miscoverage_ratio_mean']:.3f}, "
                f"Bound={bv['bound_holds_pct']:.0f}%"
            )

    if risk_agg:
        print("\nPer-Risk Coverage (LSTM):")
        for rn, rv in sorted(risk_agg.items(), key=lambda x: x[1]["coverage_mean"]):
            det = f", detect={rv['detection_mean']:.3f}" if rv["detection_mean"] else ""
            print(f"  {rn}: cov={rv['coverage_mean']:.4f}+/-{rv['coverage_std']:.4f}{det}")

    print(f"\nElapsed: {elapsed:.0f}s")

    # --- Save ---
    output = {
        "config": {
            "dataset": "AI45Research/ATBench",
            "max_traces": args.max_traces,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "n_seeds": len(args.seeds),
        },
        "data_stats": {
            "n_traces": len(traces),
            "n_safe": sum(1 for t in traces if t["safe"]),
            "n_unsafe": sum(1 for t in traces if not t["safe"]),
            "safe_rate": safe_pct / 100,
            "mean_trace_length": float(avg_len),
            "vocab_size": len(vocab),
            "unique_tools": len(action_counts),
            "risk_distribution": dict(risk_dist.most_common()),
            "top_tools": {cmd: cnt for cmd, cnt in action_counts.most_common(20)},
        },
        "aggregate": {
            "auroc": {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs)), **auroc_ci},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), **f1_ci},
            "coverage": {
                "mean": float(np.mean(coverages)),
                "std": float(np.std(coverages)),
                **cov_ci,
            },
            "miscoverage_ratio": {
                "mean": float(np.mean(misc_ratios)),
                "std": float(np.std(misc_ratios)),
                **mr_ci,
            },
            "bound_holds_pct": bound_pct,
        },
        "rac_aggregate": {
            "var_improvement_mean": float(np.mean(rac_var_improvements)),
            "var_improvement_std": float(np.std(rac_var_improvements)),
            "defer_rate_mean": float(np.mean(rac_defer_rates)),
        },
        "baselines": bl_agg,
        "per_risk_category": risk_agg,
        "per_seed": [
            {k: v for k, v in r.items() if k not in ("per_risk_category", "rac_sweep")}
            for r in results
        ],
        "per_seed_risk_categories": [r.get("per_risk_category", {}) for r in results],
        "elapsed_seconds": elapsed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "atbench_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
