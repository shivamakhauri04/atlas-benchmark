"""Distribution shift experiment: validates ACI and FA-CP under non-stationarity.

Design:
  1. Calibrate on 2000 normal-only traces (stationary, 5% anomaly rate).
  2. Test phase 1 (traces 0-999): 5% anomaly rate (stationary).
  3. Test phase 2 (traces 1000-1999): 20% anomaly rate (shifted).
  4. Measure windowed coverage in 100-trace windows.

Expected results:
  - Split CP: coverage holds globally (exchangeability per-trace), but
    no adaptation to the shift.
  - ACI (Theorem 2): adapts tau to maintain long-run coverage under shift.
  - FA-CP (Theorem 3): detects the shift via delta estimation, adjusts
    effective threshold.

Usage:
    python scripts/run_shift_experiment.py
    python scripts/run_shift_experiment.py --checkpoint checkpoints/no_args/no_args_best_epoch_*.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    generate_shifted_traces,
    generate_synthetic_traces,
)
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from trustlens.conformal import (
    ACIState,
    FeedbackAwareCPState,
    aci_step,
    feedback_aware_cp_update,
    split_conformal_calibrate,
    split_conformal_predict,
)
from trustlens.encoder import build_encoder


def compute_scores(model, traces, vocab, device, batch_size=64):
    """Compute per-trace anomaly scores."""
    from torch.utils.data import DataLoader

    ds = TraceDataset(traces, vocab)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
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


def windowed_coverage(preds, labels, window_size=100):
    """Compute coverage in sliding windows."""
    n = len(preds)
    windows = []
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        w_preds = preds[start:end]
        w_labels = labels[start:end]
        normal_mask = w_labels == 0
        if normal_mask.sum() == 0:
            coverage = 1.0
        else:
            coverage = float((w_preds[normal_mask] == 0).sum() / normal_mask.sum())
        anomaly_rate = float(w_labels.mean())
        windows.append({
            "start": start,
            "end": end,
            "coverage": coverage,
            "anomaly_rate": anomaly_rate,
            "n_normal": int(normal_mask.sum()),
            "n_anomalous": int(w_labels.sum()),
        })
    return windows


def run_split_cp_on_shifted(cal_scores, test_scores, test_labels, alpha=0.05):
    """Run Split CP (fixed threshold) on shifted test data."""
    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)
    windows = windowed_coverage(preds, test_labels)
    return {
        "method": "Split CP",
        "tau": float(tau),
        "windows": windows,
        "global_coverage": float(
            (preds[test_labels == 0] == 0).sum() / max((test_labels == 0).sum(), 1)
        ),
    }


def run_aci_on_shifted(cal_scores, test_scores, test_labels, alpha=0.05, gamma=0.01):
    """Run ACI (adaptive threshold) on shifted test data."""
    tau_init = split_conformal_calibrate(cal_scores, alpha=alpha)
    state = ACIState(tau=tau_init, alpha=alpha, gamma=gamma)

    preds = []
    tau_history = [tau_init]
    for score in test_scores:
        is_anom, state = aci_step(state, score)
        preds.append(int(is_anom))
        tau_history.append(state.tau)

    preds = np.array(preds, dtype=np.int32)
    windows = windowed_coverage(preds, test_labels)
    return {
        "method": "ACI",
        "tau_init": float(tau_init),
        "tau_final": float(state.tau),
        "windows": windows,
        "tau_history": [float(t) for t in tau_history],
        "global_coverage": float(
            (preds[test_labels == 0] == 0).sum() / max((test_labels == 0).sum(), 1)
        ),
        "empirical_coverage": float(state.empirical_coverage),
    }


def run_facp_on_shifted(
    cal_scores, test_scores, test_labels,
    alpha=0.05, gamma=0.01, lipschitz_L=1.0,
):
    """Run FA-CP (shift-aware threshold) on shifted test data."""
    tau_init = split_conformal_calibrate(cal_scores, alpha=alpha)
    tau_min = float(np.percentile(cal_scores, 5))
    tau_max = float(np.percentile(cal_scores, 99.5))

    state = FeedbackAwareCPState(
        tau=tau_init, alpha=alpha, gamma=gamma, lipschitz_L=lipschitz_L,
        tau_min=tau_min, tau_max=tau_max,
    )

    preds = []
    tau_history = [tau_init]
    shift_history = [0.0]
    for score in test_scores:
        is_anom, state = feedback_aware_cp_update(state, score)
        preds.append(int(is_anom))
        tau_history.append(state.tau)
        shift_history.append(state.delta_shift)

    preds = np.array(preds, dtype=np.int32)
    windows = windowed_coverage(preds, test_labels)
    return {
        "method": "Feedback-Aware CP",
        "tau_init": float(tau_init),
        "tau_final": float(state.tau),
        "delta_shift_final": float(state.delta_shift),
        "windows": windows,
        "tau_history": [float(t) for t in tau_history],
        "shift_history": [float(s) for s in shift_history],
        "global_coverage": float(
            (preds[test_labels == 0] == 0).sum() / max((test_labels == 0).sum(), 1)
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distribution shift experiment")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-args", action="store_true")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-cal", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--shift-point", type=int, default=1000)
    p.add_argument("--base-rate", type=float, default=0.05)
    p.add_argument("--shifted-rate", type=float, default=0.20)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = get_device(args.device)
    use_args = not args.no_args

    print(f"Distribution shift experiment on {device}")
    print(f"Shift: {args.base_rate:.0%} -> {args.shifted_rate:.0%} "
          f"at trace {args.shift_point}/{args.n_test}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")

    # Load or build model
    model = build_encoder(
        vocab_size=vocab.vocab_size,
        d_model=args.d_model,
        use_args=use_args,
    )
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = Path("checkpoints/no_args/no_args_best_latest.pt") if not use_args \
            else Path("checkpoints/latest.pt")
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"WARNING: No checkpoint at {ckpt}, using random init")
    model = model.to(device)

    # Generate calibration data (stationary, normal-only for calibration)
    print("\nGenerating calibration traces (stationary)...")
    cal_traces = generate_synthetic_traces(
        args.n_cal, anomaly_rate=args.base_rate, seed=args.seed + 1000,
    )
    cal_scores, cal_labels = compute_scores(model, cal_traces, vocab, device)
    cal_normal_scores = cal_scores[cal_labels == 0]
    print(f"  Cal: {len(cal_scores)} total, {len(cal_normal_scores)} normal for calibration")

    # Generate shifted test data
    print("Generating shifted test traces...")
    test_traces = generate_shifted_traces(
        args.n_test,
        shift_point=args.shift_point,
        base_rate=args.base_rate,
        shifted_rate=args.shifted_rate,
        seed=args.seed + 2000,
    )
    test_scores, test_labels = compute_scores(model, test_traces, vocab, device)

    pre_shift = test_labels[:args.shift_point]
    post_shift = test_labels[args.shift_point:]
    print(f"  Pre-shift:  {pre_shift.sum()}/{len(pre_shift)} anomalous "
          f"({pre_shift.mean():.1%})")
    print(f"  Post-shift: {post_shift.sum()}/{len(post_shift)} anomalous "
          f"({post_shift.mean():.1%})")

    # Run all three methods
    print("\nRunning Split CP...")
    split_results = run_split_cp_on_shifted(
        cal_normal_scores, test_scores, test_labels, alpha=args.alpha,
    )

    print("Running ACI...")
    aci_results = run_aci_on_shifted(
        cal_normal_scores, test_scores, test_labels, alpha=args.alpha,
    )

    print("Running FA-CP...")
    facp_results = run_facp_on_shifted(
        cal_normal_scores, test_scores, test_labels,
        alpha=args.alpha, lipschitz_L=1.0,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("DISTRIBUTION SHIFT EXPERIMENT RESULTS")
    print("=" * 70)
    for r in [split_results, aci_results, facp_results]:
        windows = r["windows"]
        pre_windows = [w for w in windows if w["start"] < args.shift_point]
        post_windows = [w for w in windows if w["start"] >= args.shift_point]
        pre_cov = np.mean([w["coverage"] for w in pre_windows]) if pre_windows else 0
        post_cov = np.mean([w["coverage"] for w in post_windows]) if post_windows else 0
        print(f"\n{r['method']}:")
        print(f"  Global coverage: {r['global_coverage']:.4f}")
        print(f"  Pre-shift coverage (mean window): {pre_cov:.4f}")
        print(f"  Post-shift coverage (mean window): {post_cov:.4f}")

    # Save results
    output = {
        "config": {
            "n_cal": args.n_cal,
            "n_test": args.n_test,
            "shift_point": args.shift_point,
            "base_rate": args.base_rate,
            "shifted_rate": args.shifted_rate,
            "alpha": args.alpha,
            "seed": args.seed,
            "use_args": use_args,
        },
        "split_cp": split_results,
        "aci": aci_results,
        "facp": facp_results,
    }
    out_path = tables_dir / "shift_experiment.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    # Save tau histories for plotting
    np.savez(
        tables_dir / "shift_tau_histories.npz",
        aci_tau=np.array(aci_results["tau_history"]),
        facp_tau=np.array(facp_results["tau_history"]),
        facp_shift=np.array(facp_results["shift_history"]),
    )
    print("Tau histories saved for plotting.")


if __name__ == "__main__":
    main()
