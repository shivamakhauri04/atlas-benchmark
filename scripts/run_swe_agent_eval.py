"""SWE-agent transfer evaluation: CP on real-world traces with natural labels.

Downloads SWE-agent trajectories from HuggingFace, extracts action sequences,
trains LSTM scorer + simple baselines, and runs split CP evaluation with
per-command-category coverage analysis.

v15: Scaled to 20K traces, d_model=128, simple baselines, per-category coverage.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_swe_agent_eval.py \
        --max-traces 20000 --epochs 30 --d-model 128 --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from torch.utils.data import DataLoader, Dataset

from trustlens.stats import bootstrap_ci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("output/tables")
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
MIN_TRACE_LEN = 3
MAX_TRACE_LEN = 200

# SWE-agent specific commands (high-level actions)
SWE_AGENT_COMMANDS = {
    "find_file", "open_file", "goto", "edit", "create", "submit",
    "scroll_down", "scroll_up", "search_dir", "search_file",
}


# ---------------------------------------------------------------------------
# Data loading and parsing
# ---------------------------------------------------------------------------


def extract_commands_from_turn(text: str) -> list[str]:
    """Extract command types from an AI turn's code blocks."""
    commands = []
    code_blocks = re.findall(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
    for block in code_blocks:
        block = block.strip()
        if not block:
            continue
        first_line = block.split("\n")[0].strip()
        if not first_line:
            continue
        cmd = first_line.split()[0] if first_line.split() else ""
        cmd = cmd.lstrip("./!$")
        if cmd:
            commands.append(cmd)
    return commands


def trajectory_to_action_sequence(trajectory: list[dict]) -> list[str]:
    """Convert a SWE-agent trajectory to a sequence of command types."""
    actions = []
    for turn in trajectory:
        if turn.get("role") == "ai" and turn.get("text"):
            cmds = extract_commands_from_turn(turn["text"])
            actions.extend(cmds)
    return actions


def load_swe_agent_data(max_traces: int = 5000) -> list[dict]:
    """Load SWE-agent trajectories from HuggingFace (streaming)."""
    from datasets import load_dataset

    print(f"Loading up to {max_traces} SWE-agent trajectories (streaming)...")
    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split="train",
        streaming=True,
    )

    traces = []
    skipped = 0
    for i, example in enumerate(ds):
        if len(traces) >= max_traces:
            break

        actions = trajectory_to_action_sequence(example["trajectory"])
        if len(actions) < MIN_TRACE_LEN:
            skipped += 1
            continue
        if len(actions) > MAX_TRACE_LEN:
            actions = actions[:MAX_TRACE_LEN]

        traces.append({
            "instance_id": example["instance_id"],
            "actions": actions,
            "solved": bool(example["target"]),
            "exit_status": example.get("exit_status", ""),
            "n_actions": len(actions),
        })

        if len(traces) % 500 == 0:
            solved = sum(1 for t in traces if t["solved"])
            print(f"  Loaded {len(traces)} traces ({solved} solved, "
                  f"{len(traces) - solved} failed)")

    solved = sum(1 for t in traces if t["solved"])
    print(f"\nLoaded {len(traces)} traces total "
          f"({solved} solved, {len(traces) - solved} failed, "
          f"{skipped} skipped)")
    return traces


def build_vocabulary(traces: list[dict], min_freq: int = 5) -> dict[str, int]:
    """Build vocabulary from action sequences."""
    counter: Counter[str] = Counter()
    for trace in traces:
        counter.update(trace["actions"])

    vocab = dict(SPECIAL_TOKENS)
    for cmd, freq in counter.most_common():
        if freq >= min_freq:
            vocab[cmd] = len(vocab)

    print(f"Vocabulary: {len(vocab)} tokens "
          f"({len(vocab) - len(SPECIAL_TOKENS)} commands + "
          f"{len(SPECIAL_TOKENS)} special)")
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
# Dataset and model
# ---------------------------------------------------------------------------


class ActionSeqDataset(Dataset):
    """PyTorch dataset for action sequences."""

    def __init__(
        self,
        traces: list[dict],
        vocab: dict[str, int],
        labels: list[int],
    ):
        self.encoded = [encode_trace(t["actions"], vocab) for t in traces]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.tensor(self.encoded[idx], dtype=torch.long), self.labels[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences and stack labels."""
    seqs, labels = zip(*batch)
    max_len = max(s.size(0) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.size(0)] = s
    return padded, torch.tensor(labels, dtype=torch.long)


class LSTMScorer(nn.Module):
    """Simple LSTM next-token predictor for non-conformity scoring."""

    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lstm = nn.LSTM(
            d_model, d_model, n_layers,
            batch_first=True, dropout=0.1,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.head(out)


class TransformerScorer(nn.Module):
    """Transformer next-token predictor for non-conformity scoring.

    2-layer Transformer encoder mirrors the LSTMScorer interface but
    validates that CP generalizes across architectures on real data.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = MAX_TRACE_LEN + 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        emb = self.embed(x) + self.pos_embed(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.transformer(emb, mask=mask)
        return self.head(out)


def compute_ncs(
    model: nn.Module,
    traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute non-conformity scores (negative log-likelihood of true next token)."""
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_scorer(
    train_traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    d_model: int = 128,
    model_class: type[nn.Module] | None = None,
) -> nn.Module:
    """Train a next-token scorer on action sequences."""
    cls = model_class or LSTMScorer
    model = cls(len(vocab), d_model=d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ds = ActionSeqDataset(train_traces, vocab, [0] * len(train_traces))
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
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
    from sklearn.metrics import f1_score, roc_auc_score

    cal_normal_scores = cal_scores[cal_labels == 0]
    n = len(cal_normal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    tau = float(np.quantile(cal_normal_scores, min(q_level, 1.0)))

    test_pred = (test_scores > tau).astype(int)
    test_normal_mask = test_labels == 0
    coverage = float(1 - test_pred[test_normal_mask].mean()) if test_normal_mask.sum() > 0 else 0.0
    bound_holds = coverage >= (1 - alpha)

    auroc = float(roc_auc_score(test_labels, test_scores))
    f1 = float(f1_score(test_labels, test_pred))

    recall = float(test_pred[test_labels == 1].mean()) if (test_labels == 1).sum() > 0 else 0.0

    return {
        "tau": tau,
        "coverage": coverage,
        "bound_holds": bound_holds,
        "auroc": auroc,
        "f1": f1,
        "recall": recall,
        "n_cal": int(n),
        "n_cal_normal": int((cal_labels == 0).sum()),
        "n_test": len(test_labels),
        "n_test_normal": int(test_normal_mask.sum()),
        "n_test_anomalous": int((test_labels == 1).sum()),
    }


# ---------------------------------------------------------------------------
# Simple baseline scorers
# ---------------------------------------------------------------------------

SWE_CMD_CATEGORIES = {
    "edit": ["edit", "create", "rm", "mv", "cp", "mkdir", "touch"],
    "search": ["search_dir", "search_file", "grep", "rg", "find", "find_file",
               "find.", "ag"],
    "navigate": ["open", "goto", "scroll_down", "scroll_up", "cd..", "cd", "ls",
                 "pwd", "tree"],
    "execute": ["python", "pytest", "bash", "pip", "npm", "node", "make",
                "cargo", "go"],
    "submit": ["submit"],
    "other": [],
}


def categorize_command(cmd: str) -> str:
    """Map a command to its category."""
    for cat, cmds in SWE_CMD_CATEGORIES.items():
        if cat == "other":
            continue
        if cmd in cmds:
            return cat
    return "other"


def compute_simple_baselines(
    traces: list[dict],
) -> dict[str, np.ndarray]:
    """Compute simple baseline scores for anomaly detection."""
    n = len(traces)
    length_scores = np.zeros(n)
    diversity_scores = np.zeros(n)
    repeat_scores = np.zeros(n)

    for i, t in enumerate(traces):
        actions = t["actions"]
        n_actions = len(actions)
        length_scores[i] = n_actions
        unique_cmds = len(set(actions))
        diversity_scores[i] = 1.0 - (unique_cmds / max(n_actions, 1))
        if n_actions > 1:
            repeats = sum(1 for j in range(1, n_actions) if actions[j] == actions[j - 1])
            repeat_scores[i] = repeats / (n_actions - 1)

    return {
        "trace_length": length_scores,
        "low_diversity": diversity_scores,
        "repeat_rate": repeat_scores,
    }


def compute_per_category_coverage(
    test_traces: list[dict],
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    tau: float,
) -> dict[str, dict]:
    """Compute per-command-category coverage (Localized ARC on real data)."""
    test_pred = (test_scores > tau).astype(int)
    cat_results = {}
    for cat_name in SWE_CMD_CATEGORIES:
        cat_mask = np.zeros(len(test_traces), dtype=bool)
        for i, t in enumerate(test_traces):
            cats = [categorize_command(a) for a in t["actions"]]
            if cat_name in cats:
                cat_mask[i] = True

        n_in_cat = int(cat_mask.sum())
        if n_in_cat < 5:
            continue

        normal_in_cat = cat_mask & (test_labels == 0)
        n_normal = int(normal_in_cat.sum())
        if n_normal < 2:
            continue

        cat_coverage = float(1 - test_pred[normal_in_cat].mean())
        cat_results[cat_name] = {
            "n_traces": n_in_cat,
            "n_normal": n_normal,
            "coverage": cat_coverage,
            "meets_target": cat_coverage >= 0.95,
        }

    return cat_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_single_seed(
    traces: list[dict],
    vocab: dict[str, int],
    seed: int,
    epochs: int,
    device: torch.device,
    alpha: float = 0.05,
    d_model: int = 128,
    include_transformer: bool = False,
) -> dict:
    """Run one seed: split data, train, score, evaluate CP + baselines."""
    from sklearn.metrics import roc_auc_score

    set_seeds(seed)
    rng = np.random.RandomState(seed)

    solved = [t for t in traces if t["solved"]]
    failed = [t for t in traces if not t["solved"]]

    rng.shuffle(solved)
    rng.shuffle(failed)

    n_solved = len(solved)
    n_failed = len(failed)

    n_train_solved = int(0.6 * n_solved)
    n_cal_solved = int(0.2 * n_solved)
    n_train_failed = int(0.6 * n_failed)
    n_cal_failed = int(0.2 * n_failed)

    train_traces = solved[:n_train_solved] + failed[:n_train_failed]
    cal_traces = solved[n_train_solved:n_train_solved + n_cal_solved] + \
        failed[n_train_failed:n_train_failed + n_cal_failed]
    test_traces = solved[n_train_solved + n_cal_solved:] + \
        failed[n_train_failed + n_cal_failed:]

    rng.shuffle(train_traces)
    rng.shuffle(cal_traces)
    rng.shuffle(test_traces)

    cal_labels = np.array([0 if t["solved"] else 1 for t in cal_traces])
    test_labels = np.array([0 if t["solved"] else 1 for t in test_traces])

    train_normal = [t for t in train_traces if t["solved"]]
    print(f"\n  Seed {seed}: train={len(train_traces)} "
          f"({len(train_normal)} normal for LSTM), "
          f"cal={len(cal_traces)} ({(cal_labels == 0).sum()} normal), "
          f"test={len(test_traces)} ({(test_labels == 0).sum()} normal, "
          f"{(test_labels == 1).sum()} anomalous)")

    model = train_scorer(
        train_normal, vocab, device, epochs=epochs, d_model=d_model,
    )

    cal_scores = compute_ncs(model, cal_traces, vocab, device)
    test_scores = compute_ncs(model, test_traces, vocab, device)

    result = split_cp_evaluate(
        cal_scores, cal_labels, test_scores, test_labels, alpha,
    )
    result["seed"] = seed

    per_cat = compute_per_category_coverage(
        test_traces, test_scores, test_labels, result["tau"],
    )
    result["per_category"] = per_cat

    baseline_scores_cal = compute_simple_baselines(cal_traces)
    baseline_scores_test = compute_simple_baselines(test_traces)
    baselines = {}
    for name in baseline_scores_test:
        bl_auroc = float(roc_auc_score(test_labels, baseline_scores_test[name]))
        bl_cp = split_cp_evaluate(
            baseline_scores_cal[name], cal_labels,
            baseline_scores_test[name], test_labels, alpha,
        )
        baselines[name] = {
            "auroc": bl_auroc,
            "coverage": bl_cp["coverage"],
            "bound_holds": bl_cp["bound_holds"],
            "f1": bl_cp["f1"],
        }
    result["baselines"] = baselines

    if include_transformer:
        set_seeds(seed)
        tfm_model = train_scorer(
            train_normal, vocab, device, epochs=epochs, d_model=d_model,
            model_class=TransformerScorer,
        )
        tfm_cal_scores = compute_ncs(tfm_model, cal_traces, vocab, device)
        tfm_test_scores = compute_ncs(tfm_model, test_traces, vocab, device)
        tfm_result = split_cp_evaluate(
            tfm_cal_scores, cal_labels, tfm_test_scores, test_labels, alpha,
        )
        tfm_auroc = float(roc_auc_score(test_labels, tfm_test_scores))
        result["transformer"] = {
            "auroc": tfm_auroc,
            "f1": tfm_result["f1"],
            "coverage": tfm_result["coverage"],
            "bound_holds": tfm_result["bound_holds"],
            "recall": tfm_result["recall"],
            "tau": tfm_result["tau"],
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SWE-agent CP evaluation with natural labels",
    )
    parser.add_argument("--max-traces", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 123, 456, 789, 1024],
    )
    parser.add_argument(
        "--include-transformer", action="store_true",
        help="Also train Transformer scorer alongside LSTM for architecture comparison",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Alpha: {args.alpha}")

    t0 = time.time()
    traces = load_swe_agent_data(args.max_traces)

    vocab = build_vocabulary(traces)
    print("\nAction stats:")
    all_actions = [a for t in traces for a in t["actions"]]
    action_counts = Counter(all_actions)
    print(f"  Total actions: {len(all_actions)}")
    print(f"  Unique commands: {len(action_counts)}")
    print("  Top 15 commands:")
    for cmd, cnt in action_counts.most_common(15):
        print(f"    {cmd}: {cnt}")

    avg_len = np.mean([t["n_actions"] for t in traces])
    solved_pct = 100 * sum(1 for t in traces if t["solved"]) / len(traces)
    print(f"  Mean trace length: {avg_len:.1f} actions")
    print(f"  Solved rate: {solved_pct:.1f}%")

    results = []
    for seed in args.seeds:
        r = run_single_seed(
            traces, vocab, seed, args.epochs, device, args.alpha,
            d_model=args.d_model,
            include_transformer=args.include_transformer,
        )
        results.append(r)
        print(f"  → AUROC={r['auroc']:.4f}, F1={r['f1']:.4f}, "
              f"Coverage={r['coverage']:.4f}, "
              f"Bound={'HOLDS' if r['bound_holds'] else 'FAILS'}")
        if r.get("transformer"):
            tv = r["transformer"]
            print(f"  → Transformer: AUROC={tv['auroc']:.4f}, "
                  f"F1={tv['f1']:.4f}, Coverage={tv['coverage']:.4f}, "
                  f"Bound={'HOLDS' if tv['bound_holds'] else 'FAILS'}")
        if r.get("baselines"):
            for bn, bv in r["baselines"].items():
                print(f"    baseline {bn}: AUROC={bv['auroc']:.4f}, "
                      f"Coverage={bv['coverage']:.4f}")
        if r.get("per_category"):
            cats_meeting = sum(
                1 for c in r["per_category"].values() if c["meets_target"]
            )
            print(f"    Per-category: {cats_meeting}/{len(r['per_category'])} "
                  f"meet 95% target")

    aurocs = [r["auroc"] for r in results]
    f1s = [r["f1"] for r in results]
    coverages = [r["coverage"] for r in results]
    bound_holds_pct = 100 * sum(
        1 for r in results if r["bound_holds"]
    ) / len(results)

    auroc_ci = bootstrap_ci(np.array(aurocs))
    f1_ci = bootstrap_ci(np.array(f1s))
    cov_ci = bootstrap_ci(np.array(coverages))

    bl_names = list(results[0].get("baselines", {}).keys())
    bl_agg = {}
    for bn in bl_names:
        bl_aurocs = [r["baselines"][bn]["auroc"] for r in results]
        bl_covs = [r["baselines"][bn]["coverage"] for r in results]
        bl_bounds = sum(
            1 for r in results if r["baselines"][bn]["bound_holds"]
        )
        bl_agg[bn] = {
            "auroc_mean": float(np.mean(bl_aurocs)),
            "auroc_std": float(np.std(bl_aurocs)),
            "coverage_mean": float(np.mean(bl_covs)),
            "bound_holds_pct": 100 * bl_bounds / len(results),
        }

    cat_names = set()
    for r in results:
        cat_names.update(r.get("per_category", {}).keys())
    cat_agg = {}
    for cn in sorted(cat_names):
        cat_covs = []
        cat_meets = 0
        for r in results:
            pc = r.get("per_category", {}).get(cn)
            if pc:
                cat_covs.append(pc["coverage"])
                if pc["meets_target"]:
                    cat_meets += 1
        if cat_covs:
            cat_agg[cn] = {
                "coverage_mean": float(np.mean(cat_covs)),
                "coverage_std": float(np.std(cat_covs)),
                "meets_target_pct": 100 * cat_meets / len(cat_covs),
                "n_seeds": len(cat_covs),
            }

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"SWE-agent CP Evaluation ({len(args.seeds)} seeds, "
          f"d_model={args.d_model})")
    print(f"{'=' * 60}")
    print("LSTM Scorer:")
    print(f"  AUROC:    {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}  "
          f"CI [{auroc_ci['ci_lower']:.4f}, {auroc_ci['ci_upper']:.4f}]")
    print(f"  F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  "
          f"CI [{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]")
    print(f"  Coverage: {np.mean(coverages):.4f} ± {np.std(coverages):.4f}  "
          f"CI [{cov_ci['ci_lower']:.4f}, {cov_ci['ci_upper']:.4f}]")
    print(f"  Bound holds: {bound_holds_pct:.0f}% "
          f"({sum(1 for r in results if r['bound_holds'])}"
          f"/{len(results)} seeds)")

    tfm_agg = {}
    if any(r.get("transformer") for r in results):
        tfm_aurocs = [r["transformer"]["auroc"] for r in results if r.get("transformer")]
        tfm_f1s = [r["transformer"]["f1"] for r in results if r.get("transformer")]
        tfm_covs = [r["transformer"]["coverage"] for r in results if r.get("transformer")]
        tfm_bounds = sum(
            1 for r in results if r.get("transformer", {}).get("bound_holds")
        )
        tfm_agg = {
            "auroc": {"mean": float(np.mean(tfm_aurocs)), "std": float(np.std(tfm_aurocs)),
                      **bootstrap_ci(np.array(tfm_aurocs))},
            "f1": {"mean": float(np.mean(tfm_f1s)), "std": float(np.std(tfm_f1s)),
                   **bootstrap_ci(np.array(tfm_f1s))},
            "coverage": {"mean": float(np.mean(tfm_covs)), "std": float(np.std(tfm_covs)),
                         **bootstrap_ci(np.array(tfm_covs))},
            "bound_holds_pct": 100 * tfm_bounds / len(results),
        }
        print("\nTransformer Scorer:")
        print(f"  AUROC:    {np.mean(tfm_aurocs):.4f} ± {np.std(tfm_aurocs):.4f}")
        print(f"  F1:       {np.mean(tfm_f1s):.4f} ± {np.std(tfm_f1s):.4f}")
        print(f"  Coverage: {np.mean(tfm_covs):.4f} ± {np.std(tfm_covs):.4f}")
        print(f"  Bound holds: {tfm_agg['bound_holds_pct']:.0f}%")

    if bl_agg:
        print("\nSimple Baselines:")
        for bn, bv in bl_agg.items():
            print(f"  {bn}: AUROC={bv['auroc_mean']:.4f}±"
                  f"{bv['auroc_std']:.4f}, "
                  f"Coverage={bv['coverage_mean']:.4f}, "
                  f"Bound={bv['bound_holds_pct']:.0f}%")

    if cat_agg:
        print("\nPer-Category Coverage:")
        for cn, cv in sorted(cat_agg.items(),
                             key=lambda x: x[1]["coverage_mean"]):
            print(f"  {cn}: {cv['coverage_mean']:.4f}±"
                  f"{cv['coverage_std']:.4f} "
                  f"(meets target: {cv['meets_target_pct']:.0f}%)")

    print(f"\nElapsed: {elapsed:.0f}s")

    output = {
        "config": {
            "dataset": "nebius/SWE-agent-trajectories",
            "max_traces": args.max_traces,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "n_seeds": len(args.seeds),
            "include_transformer": args.include_transformer,
        },
        "data_stats": {
            "n_traces": len(traces),
            "n_solved": sum(1 for t in traces if t["solved"]),
            "n_failed": sum(1 for t in traces if not t["solved"]),
            "solved_rate": solved_pct / 100,
            "mean_trace_length": float(avg_len),
            "vocab_size": len(vocab),
            "unique_commands": len(action_counts),
            "top_commands": {
                cmd: cnt for cmd, cnt in action_counts.most_common(20)
            },
        },
        "aggregate": {
            "auroc": {
                "mean": float(np.mean(aurocs)),
                "std": float(np.std(aurocs)),
                **auroc_ci,
            },
            "f1": {
                "mean": float(np.mean(f1s)),
                "std": float(np.std(f1s)),
                **f1_ci,
            },
            "coverage": {
                "mean": float(np.mean(coverages)),
                "std": float(np.std(coverages)),
                **cov_ci,
            },
            "bound_holds_pct": bound_holds_pct,
        },
        "transformer": tfm_agg,
        "baselines": bl_agg,
        "per_category": cat_agg,
        "per_seed": [{k: v for k, v in r.items()
                      if k != "per_category"} for r in results],
        "per_seed_categories": [r.get("per_category", {}) for r in results],
        "elapsed_seconds": elapsed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "swe_agent_eval_v15.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
