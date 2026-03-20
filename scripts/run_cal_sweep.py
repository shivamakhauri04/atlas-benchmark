"""Cal set size sweep: test how conformal prediction coverage varies with cal set size.

Demonstrates that the coverage guarantee holds at all sizes and that the
bound tightens (becomes more useful) with more calibration data.

Supports both standard encoder and consequence-aware encoder, with
optional normal-only calibration filtering.

Usage:
    python scripts/run_cal_sweep.py
    python scripts/run_cal_sweep.py --consequence --normal-only --outcome-weight 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.encoder import build_consequence_encoder, build_encoder


def compute_scores(model, dataset, device, batch_size=64):
    """Compute per-trace anomaly scores (standard encoder)."""
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
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


def compute_consequence_scores(model, dataset, device, batch_size=64):
    """Compute per-trace anomaly scores (consequence-aware encoder)."""
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
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


def train_consequence_encoder(
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 30,
    d_model: int = 128,
    outcome_weight: float = 0.5,
    seed: int = 42,
) -> torch.nn.Module:
    """Train a consequence-aware encoder for cal sweep."""
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


def parse_args():
    p = argparse.ArgumentParser(description="Cal set size sweep")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-trials", type=int, default=20, help="Trials per cal size")
    p.add_argument("--consequence", action="store_true",
                   help="Use consequence-aware encoder instead of standard")
    p.add_argument("--normal-only", action="store_true",
                   help="Filter cal scores to normal traces only")
    p.add_argument("--outcome-weight", type=float, default=0.5,
                   help="Outcome weight for consequence encoder")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    use_consequence = args.consequence
    normal_only = args.normal_only

    scorer_desc = "consequence-aware" if use_consequence else "standard"
    cal_desc = "normal-only" if normal_only else "all"
    print(f"Running cal sweep on {device} ({scorer_desc} encoder, {cal_desc} cal)")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")

    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    if use_consequence:
        model = train_consequence_encoder(
            train_traces, vocab, device,
            outcome_weight=args.outcome_weight,
        )
        cal_scores, cal_labels = compute_consequence_scores(
            model, cal_ds, device,
        )
        test_scores, test_labels = compute_consequence_scores(
            model, test_ds, device,
        )
    else:
        model = build_encoder(vocab_size=vocab.vocab_size)
        if args.checkpoint.exists():
            state = torch.load(
                args.checkpoint, map_location=device, weights_only=False,
            )
            model.load_state_dict(state["model_state_dict"])
            print(f"Loaded model from {args.checkpoint}")
        model = model.to(device)
        cal_scores, cal_labels = compute_scores(model, cal_ds, device)
        test_scores, test_labels = compute_scores(model, test_ds, device)

    # If normal-only, restrict to normal cal traces for the sweep pool
    if normal_only:
        normal_mask = cal_labels == 0
        sweep_scores = cal_scores[normal_mask]
        print(f"Normal-only cal: {normal_mask.sum()} / {len(cal_labels)} traces")
    else:
        sweep_scores = cal_scores

    n_sweep = len(sweep_scores)
    cal_sizes = [50, 100, 200, 500, 1000, n_sweep]
    results = []

    print(f"\nAlpha={args.alpha}, Sweep pool={n_sweep}, "
          f"n_trials={args.n_trials}")
    print(f"{'Cal Size':>10} {'Coverage':>12} {'F1':>12} "
          f"{'Tau':>12} {'Bound Slack':>14}")
    print("-" * 62)

    for cal_n in cal_sizes:
        if cal_n > n_sweep:
            continue

        coverages, f1s, taus, bound_holds_list = [], [], [], []

        for trial in range(args.n_trials):
            rng = np.random.RandomState(trial)
            idx = rng.choice(n_sweep, size=cal_n, replace=False)
            sub_scores = sweep_scores[idx]

            tau = split_conformal_calibrate(sub_scores, alpha=args.alpha)
            preds = split_conformal_predict(test_scores, tau)

            n_neg = (test_labels == 0).sum()
            coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
            miscoverage = float(preds[test_labels == 0].mean())
            f1 = f1_score(test_labels, preds, zero_division=0.0)
            bound = args.alpha + 1.0 / (cal_n + 1)

            coverages.append(float(coverage))
            f1s.append(float(f1))
            taus.append(float(tau))
            bound_holds_list.append(bool(miscoverage <= bound + 1e-9))

        cov_mean = np.mean(coverages)
        cov_std = np.std(coverages)
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s)
        tau_mean = np.mean(taus)
        bound_slack = cov_mean - (1 - args.alpha)
        bound_holds_frac = float(np.mean(bound_holds_list))

        results.append({
            "cal_size": cal_n,
            "coverage_mean": float(cov_mean),
            "coverage_std": float(cov_std),
            "f1_mean": float(f1_mean),
            "f1_std": float(f1_std),
            "tau_mean": float(tau_mean),
            "bound_slack": float(bound_slack),
            "bound_holds_frac": bound_holds_frac,
            "n_trials": args.n_trials,
            "bound_holds": bool(cov_mean >= (1 - args.alpha)),
        })

        print(
            f"{cal_n:>10} "
            f"{cov_mean:>8.4f}±{cov_std:.3f} "
            f"{f1_mean:>8.4f}±{f1_std:.3f} "
            f"{tau_mean:>12.4f} "
            f"{bound_slack:>14.4f}"
        )

    suffix = "_consequence" if use_consequence else ""
    out_name = f"cal_sweep{suffix}.json"
    out_path = tables_dir / out_name
    output = {
        "config": {
            "scorer": scorer_desc,
            "calibration": cal_desc,
            "alpha": args.alpha,
            "n_trials": args.n_trials,
            "outcome_weight": args.outcome_weight if use_consequence else None,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
