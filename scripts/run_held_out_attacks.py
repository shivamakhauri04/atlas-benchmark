"""Held-out attack generalization experiment.

Tests whether the encoder can detect novel attack patterns it was NOT
trained on — critical for establishing generalization beyond the
15 hand-crafted patterns in the synthetic dataset.

Design:
  - Train on patterns A01-A10 (indices 0-9).
  - Test on ALL 15 patterns, report per-pattern AUROC.
  - Compare seen (A01-A10) vs unseen (A11-A15) detection rates.

Usage:
    python scripts/run_held_out_attacks.py
    python scripts/run_held_out_attacks.py --no-args
    python scripts/run_held_out_attacks.py --consequence --outcome-weight 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    generate_synthetic_traces,
    split_traces,
)
from data.tool_registry import ANOMALY_PATTERNS
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from trustlens.encoder import build_consequence_encoder, build_encoder


N_PATTERNS = len(ANOMALY_PATTERNS)
SEEN_INDICES = list(range(10))       # A01-A10
UNSEEN_INDICES = list(range(10, N_PATTERNS))  # A11-A15


def train_encoder(train_traces, vocab, device, epochs=30, use_args=True, d_model=128, seed=42):
    """Train encoder from scratch on the given traces."""
    set_seeds(seed)
    normal_train = [t for t in train_traces if not t.has_anomaly]
    train_ds = TraceDataset(normal_train, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )

    model = build_encoder(
        vocab_size=vocab.vocab_size, d_model=d_model, use_args=use_args,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _epoch in range(epochs):
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            loss = model.training_step(tool_ids, args_hashes, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def score_traces(model, traces, vocab, device):
    """Compute per-trace anomaly scores."""
    ds = TraceDataset(traces, vocab, max_len=200)
    dl = DataLoader(
        ds, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]
            scores = model.trace_score(tool_ids, args_hashes, mask)
            all_scores.append(scores.cpu().numpy())
            all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))
    return np.concatenate(all_scores), np.concatenate(all_labels)


def train_consequence_encoder(
    train_traces, vocab, device, epochs=30, d_model=128, outcome_weight=0.5, seed=42,
):
    """Train a consequence-aware encoder on the given traces."""
    set_seeds(seed)
    normal_train = [t for t in train_traces if not t.has_anomaly]
    train_ds = TraceDataset(normal_train, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )

    model = build_consequence_encoder(
        vocab_size=vocab.vocab_size,
        d_model=d_model,
        use_args=False,
        outcome_weight=outcome_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _epoch in range(epochs):
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            loss = model.training_step(
                tool_ids, args_hashes, labels, mask,
                outcome_hashes=outcome_hashes,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def score_traces_consequence(model, traces, vocab, device):
    """Compute per-trace anomaly scores using consequence-aware encoder."""
    ds = TraceDataset(traces, vocab, max_len=200)
    dl = DataLoader(
        ds, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]
            scores = model.trace_score(
                tool_ids, args_hashes, mask,
                outcome_hashes=outcome_hashes,
            )
            all_scores.append(scores.cpu().numpy())
            all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))
    return np.concatenate(all_scores), np.concatenate(all_labels)


def generate_per_pattern_test(n_per_pattern=200, seed=42):
    """Generate test traces with exactly one known pattern per trace.

    Returns a list of (pattern_index, traces) tuples.
    """
    per_pattern = []
    for pidx in range(N_PATTERNS):
        traces = generate_synthetic_traces(
            n_per_pattern,
            anomaly_rate=0.50,
            seed=seed + pidx * 1000,
            allowed_pattern_indices=[pidx],
        )
        per_pattern.append((pidx, traces))
    return per_pattern


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out attack generalization")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-args", action="store_true")
    p.add_argument("--consequence", action="store_true",
                   help="Use consequence-aware encoder (joint tool+outcome prediction)")
    p.add_argument("--outcome-weight", type=float, default=0.5,
                   help="Weight for outcome NLL in consequence-aware scorer")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--encoder-epochs", type=int, default=30)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-test-per-pattern", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = get_device(args.device)
    use_args = not args.no_args
    use_consequence = args.consequence

    print(f"Held-out attack experiment on {device}")
    if use_consequence:
        print(f"Mode: CONSEQUENCE-AWARE (outcome_weight={args.outcome_weight})")
    else:
        print(f"Mode: sequence-only (use_args={use_args})")
    print(f"Seen patterns: A01-A10 (indices {SEEN_INDICES})")
    print(f"Unseen patterns: A11-A{N_PATTERNS} (indices {UNSEEN_INDICES})")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")

    # Generate training data with ONLY seen patterns
    print("\nGenerating training data (seen patterns only)...")
    train_traces = generate_synthetic_traces(
        args.n_train,
        anomaly_rate=0.05,
        seed=args.seed,
        allowed_pattern_indices=SEEN_INDICES,
    )
    n_anom = sum(1 for t in train_traces if t.has_anomaly)
    print(f"  Train: {len(train_traces)} ({n_anom} anomalous, seen patterns only)")

    # Train encoder
    if use_consequence:
        print(f"Training consequence-aware encoder (outcome_weight={args.outcome_weight})...")
        model = train_consequence_encoder(
            train_traces, vocab, device,
            epochs=args.encoder_epochs, d_model=args.d_model,
            outcome_weight=args.outcome_weight, seed=args.seed,
        )
    else:
        print("Training sequence-only encoder...")
        model = train_encoder(
            train_traces, vocab, device,
            epochs=args.encoder_epochs, use_args=use_args,
            d_model=args.d_model, seed=args.seed,
        )

    # Generate per-pattern test data
    print("Generating per-pattern test data...")
    per_pattern = generate_per_pattern_test(
        n_per_pattern=args.n_test_per_pattern,
        seed=args.seed + 5000,
    )

    # Score each pattern
    score_fn = score_traces_consequence if use_consequence else score_traces
    results = []
    for pidx, test_traces in per_pattern:
        scores, labels = score_fn(model, test_traces, vocab, device)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos

        try:
            auroc = float(roc_auc_score(labels, scores))
        except ValueError:
            auroc = float("nan")

        pattern_type = "seen" if pidx in SEEN_INDICES else "unseen"
        pattern_name = f"A{pidx + 1:02d}"
        results.append({
            "pattern_index": pidx,
            "pattern_name": pattern_name,
            "type": pattern_type,
            "auroc": auroc,
            "n_anomalous": n_pos,
            "n_normal": n_neg,
            "mean_anom_score": float(scores[labels == 1].mean()) if n_pos > 0 else 0,
            "mean_normal_score": float(scores[labels == 0].mean()) if n_neg > 0 else 0,
        })

    # Print summary
    print("\n" + "=" * 60)
    print("HELD-OUT ATTACK GENERALIZATION RESULTS")
    print("=" * 60)
    print(f"{'Pattern':<10} {'Type':<8} {'AUROC':>8}")
    print("-" * 30)
    for r in results:
        print(f"{r['pattern_name']:<10} {r['type']:<8} {r['auroc']:>8.4f}")

    seen_aurocs = [r["auroc"] for r in results if r["type"] == "seen" and not np.isnan(r["auroc"])]
    unseen_aurocs = [r["auroc"] for r in results
                     if r["type"] == "unseen" and not np.isnan(r["auroc"])]

    print("-" * 30)
    if seen_aurocs:
        print(f"{'Seen avg':<10} {'':8} {np.mean(seen_aurocs):>8.4f} "
              f"(std={np.std(seen_aurocs):.4f})")
    if unseen_aurocs:
        print(f"{'Unseen avg':<10} {'':8} {np.mean(unseen_aurocs):>8.4f} "
              f"(std={np.std(unseen_aurocs):.4f})")

    # Save
    output = {
        "config": {
            "seen_patterns": SEEN_INDICES,
            "unseen_patterns": UNSEEN_INDICES,
            "n_train": args.n_train,
            "n_test_per_pattern": args.n_test_per_pattern,
            "encoder_epochs": args.encoder_epochs,
            "use_args": use_args,
            "consequence_aware": use_consequence,
            "outcome_weight": args.outcome_weight if use_consequence else None,
            "d_model": args.d_model,
            "seed": args.seed,
        },
        "per_pattern": results,
        "summary": {
            "seen_auroc_mean": float(np.mean(seen_aurocs)) if seen_aurocs else None,
            "seen_auroc_std": float(np.std(seen_aurocs)) if seen_aurocs else None,
            "unseen_auroc_mean": float(np.mean(unseen_aurocs)) if unseen_aurocs else None,
            "unseen_auroc_std": float(np.std(unseen_aurocs)) if unseen_aurocs else None,
        },
    }
    suffix = "_consequence" if use_consequence else ""
    out_path = tables_dir / f"held_out_attacks{suffix}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
