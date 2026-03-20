"""TRAIL transfer evaluation: real agent traces.

Tests whether TrustLens trained on synthetic data transfers to real
agent execution traces from TRAIL (PatronusAI). Three evaluation modes:

  1. Zero-shot: Train on synthetic, test on TRAIL (no fine-tuning).
  2. Fine-tuned: Train on synthetic, fine-tune on TRAIL normals, test on TRAIL.
  3. Native: Train exclusively on TRAIL normals, test on TRAIL.

For each mode: consequence-aware, base transformer, LSTM baseline.
All encoder training uses unified 80/20 val + early stopping protocol.

Usage:
    python scripts/run_trail_transfer.py
    python scripts/run_trail_transfer.py --seeds 42,123,456 --skip-download
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from run_cp_comparison import (
    _compute_val_loss,
    _split_train_val,
    compute_consequence_scores,
    compute_encoder_scores,
    evaluate_cp,
    train_consequence_encoder_for_seed,
    train_encoder_for_seed,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from baselines.all_baselines import LSTMNextTokenBaseline
from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    download_trail,
    load_traces,
    load_trail,
    split_traces,
)
from trustlens.stats import bootstrap_ci


def _build_trail_vocab(
    trail_traces: list,
    synthetic_vocab: TraceVocab | None = None,
) -> TraceVocab:
    """Build vocabulary covering both TRAIL and (optionally) synthetic tools."""
    vocab = TraceVocab()
    if synthetic_vocab is not None:
        vocab.tool2id = dict(synthetic_vocab.tool2id)
        vocab._next_id = max(vocab.tool2id.values()) + 1 if vocab.tool2id else 4
    vocab.fit(trail_traces)
    return vocab


def _finetune_encoder(
    model: torch.nn.Module,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 20,
    patience: int = 3,
    lr: float = 5e-5,
    seed: int = 42,
    use_outcome: bool = False,
) -> torch.nn.Module:
    """Fine-tune a pre-trained encoder on new traces with early stopping."""
    set_seeds(seed)
    normal = [t for t in train_traces if not t.has_anomaly]
    if len(normal) < 4:
        return model

    tr, val = _split_train_val(normal, val_frac=0.2, seed=seed)
    train_ds = TraceDataset(tr, vocab, max_len=200)
    val_ds = TraceDataset(val, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_traces,
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_traces,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            if use_outcome:
                outcome_hashes = batch["outcome_hashes"].to(device)
                loss = model.training_step(
                    tool_ids,
                    args_hashes,
                    labels,
                    mask,
                    outcome_hashes=outcome_hashes,
                )
            else:
                loss = model.training_step(tool_ids, args_hashes, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_loss = _compute_val_loss(model, val_dl, device, use_outcome=use_outcome)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def run_trail_evaluation(
    mode: str,
    seed: int,
    trail_train: list,
    trail_cal: list,
    trail_test: list,
    vocab: TraceVocab,
    device: torch.device,
    alpha: float = 0.05,
    synthetic_train: list | None = None,
) -> list[dict]:
    """Run one seed of TRAIL evaluation for a given mode.

    Args:
        mode: "zero_shot", "fine_tuned", or "native".
        synthetic_train: Required for zero_shot and fine_tuned modes.
    """
    results = []
    cal_ds = TraceDataset(trail_cal, vocab)
    test_ds = TraceDataset(trail_test, vocab)

    cal_labels = np.array(
        [1 if t.has_anomaly else 0 for t in trail_cal],
        dtype=np.int32,
    )
    test_labels = np.array(
        [1 if t.has_anomaly else 0 for t in trail_test],
        dtype=np.int32,
    )

    n_anom_test = int(test_labels.sum())
    if n_anom_test == 0 or n_anom_test == len(test_labels):
        print(
            f"    WARNING: degenerate test set ({n_anom_test} anomalous "
            f"of {len(test_labels)}), AUROC undefined"
        )

    scorers = [
        ("Transformer", False),
        ("Consequence-Aware", True),
    ]

    for scorer_name, use_consequence in scorers:
        set_seeds(seed)

        if mode == "zero_shot":
            if use_consequence:
                model = train_consequence_encoder_for_seed(
                    seed,
                    synthetic_train,
                    vocab,
                    device,
                )
            else:
                model = train_encoder_for_seed(
                    seed,
                    synthetic_train,
                    vocab,
                    device,
                    use_args=False,
                )

        elif mode == "fine_tuned":
            if use_consequence:
                model = train_consequence_encoder_for_seed(
                    seed,
                    synthetic_train,
                    vocab,
                    device,
                )
                model = _finetune_encoder(
                    model,
                    trail_train,
                    vocab,
                    device,
                    seed=seed,
                    use_outcome=True,
                )
            else:
                model = train_encoder_for_seed(
                    seed,
                    synthetic_train,
                    vocab,
                    device,
                    use_args=False,
                )
                model = _finetune_encoder(
                    model,
                    trail_train,
                    vocab,
                    device,
                    seed=seed,
                )

        elif mode == "native":
            if use_consequence:
                model = train_consequence_encoder_for_seed(
                    seed,
                    trail_train,
                    vocab,
                    device,
                    epochs=30,
                )
            else:
                model = train_encoder_for_seed(
                    seed,
                    trail_train,
                    vocab,
                    device,
                    use_args=False,
                    epochs=30,
                )

        if use_consequence:
            cal_scores, _ = compute_consequence_scores(model, cal_ds, device)
            test_scores, _ = compute_consequence_scores(model, test_ds, device)
        else:
            cal_scores, _ = compute_encoder_scores(model, cal_ds, device)
            test_scores, _ = compute_encoder_scores(model, test_ds, device)

        r = evaluate_cp(
            cal_scores,
            cal_labels,
            test_scores,
            test_labels,
            alpha=alpha,
            normal_only_cal=True,
        )
        r["scorer"] = scorer_name
        r["mode"] = mode
        r["seed"] = seed
        results.append(r)

    # LSTM baseline (no fine-tuning concept — just fit on available train data)
    set_seeds(seed)
    lstm_bl = LSTMNextTokenBaseline(epochs=20)
    if mode in ("zero_shot", "fine_tuned"):
        train_data = synthetic_train if mode == "zero_shot" else trail_train
    else:
        train_data = trail_train
    lstm_bl.fit_traces(train_data, trail_cal, vocab)
    lstm_cal_scores = lstm_bl.score_traces(trail_cal, vocab)
    lstm_test_scores = lstm_bl.score_traces(trail_test, vocab)

    r = evaluate_cp(
        lstm_cal_scores,
        cal_labels,
        lstm_test_scores,
        test_labels,
        alpha=alpha,
        normal_only_cal=True,
    )
    r["scorer"] = "LSTM"
    r["mode"] = mode
    r["seed"] = seed
    results.append(r)

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRAIL transfer evaluation")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument(
        "--trail-dir",
        type=Path,
        default=None,
        help="Path to TRAIL-Benchmark repo (auto-downloads if absent)",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Don't auto-download TRAIL; fail if not present",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="zero_shot,fine_tuned,native",
        help="Comma-separated evaluation modes",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    device = get_device(args.device)

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Load TRAIL ---
    trail_dir = args.trail_dir
    if trail_dir is None:
        trail_dir = args.data_dir / "TRAIL-Benchmark"

    if not (trail_dir / "benchmarking").exists():
        if args.skip_download:
            print(f"TRAIL not found at {trail_dir} and --skip-download set. Exiting.")
            return
        print("Downloading TRAIL-Benchmark...")
        trail_dir = download_trail(args.data_dir)

    print(f"Loading TRAIL from {trail_dir}...")
    trail_traces = load_trail(trail_dir)
    print(f"Total TRAIL traces: {len(trail_traces)}")

    n_anomalous = sum(1 for t in trail_traces if t.has_anomaly)
    n_normal = len(trail_traces) - n_anomalous
    print(f"  Normal: {n_normal}, Anomalous: {n_anomalous}")

    if len(trail_traces) < 10:
        print("Too few TRAIL traces for meaningful evaluation. Exiting.")
        return

    # Split TRAIL: 50% train, 25% cal, 25% test (small dataset)
    trail_train, trail_cal, trail_test = split_traces(
        trail_traces,
        train_frac=0.50,
        cal_frac=0.25,
        seed=42,
    )
    print(f"  TRAIL split: train={len(trail_train)}, cal={len(trail_cal)}, test={len(trail_test)}")

    # --- Load synthetic data ---
    synthetic_train = None
    if "zero_shot" in modes or "fine_tuned" in modes:
        train_path = args.data_dir / "train.jsonl"
        if train_path.exists():
            synthetic_train = load_traces(train_path)
            print(f"Loaded {len(synthetic_train)} synthetic training traces")
        else:
            print(
                f"WARNING: {train_path} not found. "
                "Zero-shot/fine-tuned modes require synthetic data."
            )
            modes = [m for m in modes if m == "native"]
            if not modes:
                return

    # --- Build combined vocabulary ---
    all_traces_for_vocab = list(trail_traces)
    if synthetic_train:
        all_traces_for_vocab.extend(synthetic_train)
    vocab = TraceVocab().fit(all_traces_for_vocab)
    print(f"Combined vocab: {vocab.vocab_size} tools")

    # --- Run evaluation ---
    all_results = []
    for mode in modes:
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode}")
        print(f"{'=' * 60}")

        for i, seed in enumerate(seeds):
            print(f"\n  --- Seed {seed} ({i + 1}/{len(seeds)}) ---")
            seed_results = run_trail_evaluation(
                mode=mode,
                seed=seed,
                trail_train=trail_train,
                trail_cal=trail_cal,
                trail_test=trail_test,
                vocab=vocab,
                device=device,
                alpha=args.alpha,
                synthetic_train=synthetic_train,
            )
            for r in seed_results:
                auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r["auroc"]) else "N/A"
                print(f"    {r['scorer']:<25s} AUROC={auroc_str}  F1={r['f1']:.4f}")
            all_results.extend(seed_results)

    # --- Aggregate ---
    print(f"\n{'=' * 80}")
    print("TRAIL TRANSFER RESULTS (AGGREGATE)")
    print(f"{'=' * 80}")

    agg = {}
    for mode in modes:
        mode_results = [r for r in all_results if r["mode"] == mode]
        scorer_names = sorted(set(r["scorer"] for r in mode_results))
        for scorer in scorer_names:
            runs = [r for r in mode_results if r["scorer"] == scorer]
            auroc_vals = np.array([r["auroc"] for r in runs])
            valid = ~np.isnan(auroc_vals)
            if valid.sum() == 0:
                continue

            auroc_valid = auroc_vals[valid]
            auroc_ci = bootstrap_ci(auroc_valid)

            key = f"{mode}/{scorer}"
            agg[key] = {
                "mode": mode,
                "scorer": scorer,
                "auroc_mean": float(auroc_valid.mean()),
                "auroc_std": float(auroc_valid.std()),
                "auroc_ci_lower": auroc_ci["ci_lower"],
                "auroc_ci_upper": auroc_ci["ci_upper"],
                "n_seeds": len(runs),
                "n_valid": int(valid.sum()),
            }

            print(
                f"  {mode:<15s} {scorer:<25s} "
                f"AUROC={auroc_valid.mean():.4f}+/-{auroc_valid.std():.3f} "
                f"[{auroc_ci['ci_lower']:.3f}, {auroc_ci['ci_upper']:.3f}]"
            )

    # --- Save ---
    output = {
        "config": {
            "seeds": seeds,
            "modes": modes,
            "alpha": args.alpha,
            "n_trail_traces": len(trail_traces),
            "n_trail_anomalous": n_anomalous,
            "trail_split": {
                "train": len(trail_train),
                "cal": len(trail_cal),
                "test": len(trail_test),
            },
        },
        "aggregate": agg,
        "per_seed": all_results,
    }
    out_path = tables_dir / "trail_transfer.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
