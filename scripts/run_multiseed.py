"""Run multi-seed experiments for statistical significance.

Trains the full model with 5 different seeds, evaluates each,
and computes aggregate statistics with bootstrap CIs.

Usage:
    python scripts/run_multiseed.py
    python scripts/run_multiseed.py --seeds 42 123 456 789 1024
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    generate_synthetic_traces,
    split_traces,
    save_traces,
)
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.encoder import build_encoder


def train_and_evaluate(seed, vocab, device, epochs=30, batch_size=64, alpha=0.05):
    """Train model from scratch with given seed and evaluate."""
    set_seeds(seed)

    # Generate fresh data with this seed
    traces = generate_synthetic_traces(n_traces=5000, anomaly_rate=0.05, seed=seed)
    train, cal, test = split_traces(traces, seed=seed)

    train_ds = TraceDataset(train, vocab)
    cal_ds = TraceDataset(cal, vocab)
    test_ds = TraceDataset(test, vocab)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )
    cal_dl = DataLoader(
        cal_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )

    model = build_encoder(vocab_size=vocab.vocab_size, n_layers=3)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            loss = model.training_step(tool_ids, args_hashes, labels, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in cal_dl:
                tool_ids = batch["tool_ids"].to(device)
                args_hashes = batch["args_hashes"].to(device)
                mask = batch["mask"].to(device)

                scores = model.trace_score(tool_ids, args_hashes, mask)
                val_loss += scores.mean().item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Compute scores
    def _get_scores(dl):
        all_s, all_l = [], []
        model.eval()
        with torch.no_grad():
            for batch in dl:
                tool_ids = batch["tool_ids"].to(device)
                args_hashes = batch["args_hashes"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"]
                scores = model.trace_score(tool_ids, args_hashes, mask)
                all_s.append(scores.cpu().numpy())
                all_l.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))
        return np.concatenate(all_s), np.concatenate(all_l)

    cal_scores, cal_labels = _get_scores(cal_dl)
    test_scores, test_labels = _get_scores(test_dl)

    # Split CP
    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n_neg = (test_labels == 0).sum()
    n_pos = test_labels.sum()
    coverage = float((preds[test_labels == 0] == 0).sum() / max(n_neg, 1))
    recall = float((preds[test_labels == 1] == 1).sum() / max(n_pos, 1))
    f1 = float(f1_score(test_labels, preds, zero_division=0.0))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = float("nan")

    miscoverage = float(preds[test_labels == 0].mean())
    bound = alpha + 1.0 / (len(cal_scores) + 1)
    bound_holds = bool(miscoverage <= bound + 1e-9)

    return {
        "seed": seed,
        "n_train": len(train),
        "n_cal": len(cal),
        "n_test": len(test),
        "n_anomalous_test": int(n_pos),
        "auroc": auroc,
        "f1": f1,
        "coverage": coverage,
        "recall": recall,
        "miscoverage": miscoverage,
        "bound": float(bound),
        "bound_holds": bound_holds,
    }


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)
    rng = np.random.RandomState(42)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lower), float(upper)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed experiments")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Running multi-seed experiments on {device}")
    print(f"Seeds: {args.seeds}")

    vocab = TraceVocab.load(args.data_dir / "vocab.json")

    all_results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{len(args.seeds)})")
        print(f"{'='*60}")

        t0 = time.time()
        result = train_and_evaluate(
            seed, vocab, device, epochs=args.epochs, alpha=args.alpha,
        )
        dur = time.time() - t0
        result["duration_s"] = dur

        print(f"  AUROC={result['auroc']:.4f}  F1={result['f1']:.4f}  "
              f"Cov={result['coverage']:.4f}  Recall={result['recall']:.4f}  "
              f"Bound holds={result['bound_holds']}  ({dur:.0f}s)")
        all_results.append(result)

    # Aggregate
    metrics = ["auroc", "f1", "coverage", "recall", "miscoverage"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in all_results]
        lo, hi = bootstrap_ci(vals)
        agg[m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci_lower": lo,
            "ci_upper": hi,
            "values": vals,
        }

    bound_holds_all = all(r["bound_holds"] for r in all_results)
    agg["bound_holds_all_seeds"] = bound_holds_all

    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Mean':>8} {'Std':>8} {'95% CI':>20}")
    print("-" * 55)
    for m in metrics:
        a = agg[m]
        print(f"{m:<15} {a['mean']:>8.4f} {a['std']:>8.4f} "
              f"[{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]")
    print(f"\nBound holds (all seeds): {bound_holds_all}")

    output = {
        "seeds": args.seeds,
        "per_seed": all_results,
        "aggregate": agg,
    }
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "multiseed.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
