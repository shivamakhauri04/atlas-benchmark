"""Run conformal calibration experiments (E2, E3, E4).

E2: Split CP calibration — Theorem 1 coverage guarantee
E3: ACI calibration — Theorem 2 long-run coverage
E4: Feedback-Aware CP — Theorem 3 with shift correction

Usage:
    python scripts/run_conformal.py
    python scripts/run_conformal.py --checkpoint checkpoints/latest.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from trustlens.conformal import (
    ACIState,
    FeedbackAwareCPState,
    aci_step,
    estimate_lipschitz,
    feedback_aware_cp_update,
    split_conformal_calibrate,
    split_conformal_predict,
)
from trustlens.encoder import build_encoder


def compute_scores(model, dataset, device, batch_size=64):
    """Compute per-trace anomaly scores and labels."""
    from torch.utils.data import DataLoader

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


def run_split_cp(cal_scores, cal_labels, test_scores, test_labels, alpha=0.05):
    """E2: Split CP calibration (Theorem 1).

    Uses normal-only calibration (consistent with run_cp_comparison.py).
    """
    print("\n=== E2: Split CP (Theorem 1) ===")

    cal_normal = cal_scores[cal_labels == 0]
    print(f"  Calibrating on {len(cal_normal)}/{len(cal_scores)} normal cal scores")
    tau = split_conformal_calibrate(cal_normal, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n = len(test_labels)
    n_pos = test_labels.sum()
    n_neg = n - n_pos

    coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
    recall = (preds[test_labels == 1] == 1).sum() / max(n_pos, 1)
    f1 = f1_score(test_labels, preds, zero_division=0.0)
    miscoverage = preds[test_labels == 0].mean()

    try:
        auroc = roc_auc_score(test_labels, test_scores)
    except ValueError:
        auroc = float("nan")

    n_cal = len(cal_normal)
    bound = alpha + 1.0 / (n_cal + 1)

    results = {
        "method": "Split CP",
        "theorem": 1,
        "alpha": alpha,
        "tau": float(tau),
        "coverage": float(coverage),
        "miscoverage": float(miscoverage),
        "theoretical_bound": float(bound),
        "bound_holds": bool(miscoverage <= bound + 1e-9),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "n_cal": n_cal,
        "n_test": n,
    }

    print(f"  Threshold tau: {tau:.4f}")
    print(f"  Coverage: {coverage:.4f} (target >= {1-alpha:.2f})")
    print(f"  Miscoverage: {miscoverage:.4f} (bound: {bound:.4f})")
    print(f"  Bound holds: {results['bound_holds']}")
    print(f"  AUROC: {auroc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
    return results


def run_aci(cal_scores, cal_labels, test_scores, test_labels, alpha=0.05, gamma=0.01):
    """E3: ACI calibration (Theorem 2)."""
    print("\n=== E3: ACI (Theorem 2) ===")

    cal_normal = cal_scores[cal_labels == 0]
    tau_init = split_conformal_calibrate(cal_normal, alpha=alpha)
    state = ACIState(tau=tau_init, alpha=alpha, gamma=gamma)

    preds = []
    tau_history = [tau_init]

    for score in test_scores:
        is_anom, state = aci_step(state, score)
        preds.append(int(is_anom))
        tau_history.append(state.tau)

    preds = np.array(preds, dtype=np.int32)
    tau_history = np.array(tau_history)

    n_neg = (test_labels == 0).sum()
    coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
    empirical_coverage = state.empirical_coverage
    f1 = f1_score(test_labels, preds, zero_division=0.0)

    results = {
        "method": "ACI",
        "theorem": 2,
        "alpha": alpha,
        "gamma": gamma,
        "tau_init": float(tau_init),
        "tau_final": float(state.tau),
        "tau_mean": float(tau_history.mean()),
        "tau_std": float(tau_history.std()),
        "coverage": float(coverage),
        "empirical_coverage": float(empirical_coverage),
        "f1": float(f1),
        "n_test": len(test_scores),
    }

    print(f"  Tau: {tau_init:.4f} -> {state.tau:.4f} (mean={tau_history.mean():.4f})")
    print(f"  Coverage: {coverage:.4f}")
    print(f"  Empirical coverage (long-run): {empirical_coverage:.4f}")
    print(f"  F1: {f1:.4f}")
    return results, tau_history


def run_feedback_aware_cp(
    cal_scores, cal_labels, test_scores, test_labels,
    alpha=0.05, gamma=0.01, lipschitz_L=1.0,
):
    """E4: Feedback-Aware CP (Theorem 3)."""
    print("\n=== E4: Feedback-Aware CP (Theorem 3) ===")

    cal_normal = cal_scores[cal_labels == 0]
    tau_init = split_conformal_calibrate(cal_normal, alpha=alpha)
    tau_min = float(np.percentile(cal_normal, 5))
    tau_max = float(np.percentile(cal_normal, 99.5))
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
    tau_history = np.array(tau_history)
    shift_history = np.array(shift_history)

    n_neg = (test_labels == 0).sum()
    coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
    f1 = f1_score(test_labels, preds, zero_division=0.0)

    n = len(test_scores)
    delta = 0.05
    bound_term = alpha + lipschitz_L * shift_history[-1] / n
    bound_term += np.sqrt(np.log(1.0 / delta) / (2 * n))

    results = {
        "method": "Feedback-Aware CP",
        "theorem": 3,
        "alpha": alpha,
        "gamma": gamma,
        "lipschitz_L": lipschitz_L,
        "tau_init": float(tau_init),
        "tau_final": float(state.tau),
        "delta_shift_final": float(state.delta_shift),
        "coverage": float(coverage),
        "coverage_bound": float(state.coverage_bound),
        "miscoverage_bound": float(bound_term),
        "f1": float(f1),
        "n_test": n,
    }

    print(f"  Tau: {tau_init:.4f} -> {state.tau:.4f}")
    print(f"  Delta shift: {state.delta_shift:.4f}")
    print(f"  Coverage: {coverage:.4f}")
    print(f"  Miscoverage bound: {bound_term:.4f}")
    print(f"  F1: {f1:.4f}")
    return results, tau_history, shift_history


def run_alpha_sweep(cal_scores, cal_labels, test_scores, test_labels):
    """Sweep alpha from 0.01 to 0.20 for coverage vs safety tradeoff."""
    print("\n=== Alpha Sweep ===")
    alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    sweep_results = []
    cal_normal = cal_scores[cal_labels == 0]

    for a in alphas:
        tau = split_conformal_calibrate(cal_normal, alpha=a)
        preds = split_conformal_predict(test_scores, tau)
        n_neg = (test_labels == 0).sum()
        coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
        n_pos = test_labels.sum()
        recall = (preds[test_labels == 1] == 1).sum() / max(n_pos, 1)
        f1 = f1_score(test_labels, preds, zero_division=0.0)
        sweep_results.append({
            "alpha": a,
            "tau": float(tau),
            "coverage": float(coverage),
            "recall": float(recall),
            "f1": float(f1),
        })
        print(f"  alpha={a:.2f}: tau={tau:.4f}, coverage={coverage:.4f}, recall={recall:.4f}, f1={f1:.4f}")

    return sweep_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run conformal calibration experiments")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-args", action="store_true", help="Use no-args encoder")
    p.add_argument("--d-model", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = get_device(args.device)
    print(f"Running conformal calibration on {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    use_args = not args.no_args
    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    model = build_encoder(vocab_size=vocab.vocab_size, d_model=args.d_model, use_args=use_args)
    if args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded model from {args.checkpoint}")
    else:
        print(f"WARNING: No checkpoint at {args.checkpoint}")
    model = model.to(device)

    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")
    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    print("Computing scores...")
    cal_scores, cal_labels = compute_scores(model, cal_ds, device, args.batch_size)
    test_scores, test_labels = compute_scores(model, test_ds, device, args.batch_size)
    print(f"  Cal: {len(cal_scores)} ({cal_labels.sum()} anomalous)")
    print(f"  Test: {len(test_scores)} ({test_labels.sum()} anomalous)")

    all_results = {}

    # E2: Split CP
    all_results["split_cp"] = run_split_cp(
        cal_scores, cal_labels, test_scores, test_labels, alpha=args.alpha
    )

    # E3: ACI
    aci_results, aci_tau = run_aci(
        cal_scores, cal_labels, test_scores, test_labels, alpha=args.alpha
    )
    all_results["aci"] = aci_results

    # Estimate Lipschitz constant
    print("\n=== Lipschitz Estimation ===")
    L = estimate_lipschitz(model, cal_ds, n_samples=200, device=device)
    print(f"  Estimated L: {L:.4f}")
    all_results["lipschitz_L"] = float(L)

    # E4: Feedback-Aware CP
    facp_results, facp_tau, facp_shift = run_feedback_aware_cp(
        cal_scores, cal_labels, test_scores, test_labels,
        alpha=args.alpha, lipschitz_L=max(L, 0.1),
    )
    all_results["feedback_aware_cp"] = facp_results

    # Alpha sweep
    all_results["alpha_sweep"] = run_alpha_sweep(
        cal_scores, cal_labels, test_scores, test_labels
    )

    # Save results
    out_path = tables_dir / "conformal.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")

    # Save tau histories for plotting
    np.savez(
        args.output_dir / "tables" / "tau_histories.npz",
        aci_tau=aci_tau,
        facp_tau=facp_tau,
        facp_shift=facp_shift,
    )
    print("Tau histories saved for plotting.")


if __name__ == "__main__":
    main()
