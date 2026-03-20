"""Run all baseline methods and compute evaluation metrics.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --data-dir data --checkpoint checkpoints/latest.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import f1_score, roc_auc_score

from baselines.all_baselines import (
    get_calibration_baselines,
    get_permission_baselines,
    get_scoring_baselines,
)
from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from trustlens.encoder import build_encoder


def compute_scores(
    model: torch.nn.Module,
    dataset: TraceDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-trace anomaly scores and aggregate labels."""
    from torch.utils.data import DataLoader

    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]

            scores = model.trace_score(tool_ids, args_hashes, mask)
            all_scores.append(scores.cpu().numpy())

            trace_labels = (labels.sum(dim=1) > 0).numpy().astype(np.int32)
            all_labels.append(trace_labels)

    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate_baseline(
    baseline_name: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Compute standard metrics for a baseline."""
    n = len(labels)
    n_pos = labels.sum()
    n_neg = n - n_pos

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = f1_score(labels, predictions, zero_division=0.0)

    try:
        auroc = roc_auc_score(labels, scores) if n_pos > 0 and n_neg > 0 else float("nan")
    except ValueError:
        auroc = float("nan")

    coverage = tn / max(n_neg, 1)
    unsafe_rate = fn / max(n_pos, 1)
    deferral_rate = predictions.mean()

    return {
        "baseline": baseline_name,
        "auroc": float(auroc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "coverage": float(coverage),
        "unsafe_rate": float(unsafe_rate),
        "deferral_rate": float(deferral_rate),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_total": n,
        "n_anomalous": int(n_pos),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run TrustLens baselines")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = get_device(args.device)
    print(f"Running baselines on {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")

    model = build_encoder(vocab_size=vocab.vocab_size)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded model from {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint at {args.checkpoint}, using random weights")
    model = model.to(device)

    print("Computing scores on train/cal/test sets...")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    train_ds = TraceDataset(train_traces, vocab)
    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    train_scores, train_labels = compute_scores(model, train_ds, device, args.batch_size)
    cal_scores, cal_labels = compute_scores(model, cal_ds, device, args.batch_size)
    test_scores, test_labels = compute_scores(model, test_ds, device, args.batch_size)

    print(f"  Train: {len(train_scores)} traces ({train_labels.sum()} anomalous)")
    print(f"  Cal:   {len(cal_scores)} traces ({cal_labels.sum()} anomalous)")
    print(f"  Test:  {len(test_scores)} traces ({test_labels.sum()} anomalous)")

    def _print_table(header: str, results_list: list[dict]) -> None:
        print(f"\n{'=' * 70}")
        print(f"  {header}")
        print(f"{'=' * 70}")
        print(f"{'Baseline':<30} {'AUROC':>8} {'F1':>8} {'Coverage':>10} {'Unsafe%':>10}")
        print("-" * 70)
        for m in results_list:
            print(
                f"{m['baseline']:<30} "
                f"{m['auroc']:>8.4f} "
                f"{m['f1']:>8.4f} "
                f"{m['coverage']:>10.4f} "
                f"{m['unsafe_rate']:>10.4f}"
            )

    # --- Table 1: Independent Anomaly Scorers ---
    scoring_results = []
    for bl in get_scoring_baselines():
        bl.fit_traces(train_traces, cal_traces, vocab)
        bl_scores = bl.score_traces(test_traces, vocab)
        preds = bl.predict(bl_scores)
        scoring_results.append(
            evaluate_baseline(bl.name(), preds, test_labels, bl_scores)
        )
    _print_table("Table 1: Anomaly Scoring Methods (independent scorers)", scoring_results)

    # --- Table 2: Calibration/Thresholding Variants ---
    calibration_results = []
    for bl in get_calibration_baselines(alpha=args.alpha):
        bl.fit(train_scores, train_labels, cal_scores, cal_labels)
        preds = bl.predict(test_scores)
        calibration_results.append(
            evaluate_baseline(bl.name(), preds, test_labels, test_scores)
        )
    _print_table("Table 2: Calibration/Thresholding (given encoder scores)", calibration_results)

    # --- Table 3: Permission-Based Methods ---
    permission_results = []
    for bl in get_permission_baselines():
        bl.fit(train_scores, train_labels, cal_scores, cal_labels)
        preds = bl.predict(test_scores)
        permission_results.append(
            evaluate_baseline(bl.name(), preds, test_labels, test_scores)
        )
    _print_table("Table 3: Permission-Based Methods", permission_results)

    # Save structured output
    structured = {
        "table1_scoring": scoring_results,
        "table2_calibration": calibration_results,
        "table3_permission": permission_results,
    }
    out_path = tables_dir / "baselines.json"
    out_path.write_text(json.dumps(structured, indent=2))
    print(f"\nResults saved to {out_path}")

    # Also save legacy flat format for backwards compat
    all_results = scoring_results + calibration_results + permission_results
    legacy_path = tables_dir / "baselines_flat.json"
    legacy_path.write_text(json.dumps(all_results, indent=2))
    print(f"Legacy flat results saved to {legacy_path}")


if __name__ == "__main__":
    main()
