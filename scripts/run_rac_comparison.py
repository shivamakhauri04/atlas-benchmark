"""E13: RAC two-threshold vs standard calibration comparison.

Compares the Risk-Averse Calibration (RAC) two-threshold decision
(approve/defer/reject) against standard binary Split CP (approve/reject).

Key metrics:
  - VaR cost: C_FN * FNR + FPR (lower is better)
  - Coverage: P(approve | normal) (should be >= 1-alpha)
  - Defer rate: fraction of actions sent to human review
  - FNR/FPR reduction from the defer zone

Also runs E16: Three-way decision sensitivity analysis over
cost_ratio and defer_budget parameters.

Usage:
    python scripts/run_rac_comparison.py
    python scripts/run_rac_comparison.py --seeds 42,123,456,789,1024
"""

import argparse
import json
from pathlib import Path

import numpy as np
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from torch.utils.data import DataLoader

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from trustlens.rac import evaluate_rac, rac_sensitivity_sweep
from trustlens.stats import bootstrap_ci

# Reuse training utilities from run_cp_comparison
from run_cp_comparison import (
    compute_consequence_scores,
    evaluate_cp,
    train_consequence_encoder_for_seed,
)


def run_rac_for_seed(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    vocab: TraceVocab,
    device,
    alpha: float,
    encoder_epochs: int,
    outcome_weight: float,
    cost_ratios: list[float],
    defer_budgets: list[float],
) -> dict:
    """Run RAC comparison for one seed."""
    set_seeds(seed)

    # Train consequence-aware encoder
    model = train_consequence_encoder_for_seed(
        seed,
        train_traces,
        vocab,
        device,
        epochs=encoder_epochs,
        outcome_weight=outcome_weight,
    )

    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    cal_scores, _ = compute_consequence_scores(model, cal_ds, device)
    test_scores, _ = compute_consequence_scores(model, test_ds, device)

    # Standard binary CP (for comparison baseline)
    binary_result = evaluate_cp(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        normal_only_cal=True,
    )

    # RAC at default cost_ratio=5.0, defer_budget=0.15
    rac_default = evaluate_rac(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        cost_ratio=5.0,
        defer_budget=0.15,
        normal_only_cal=True,
    )

    # Sensitivity sweep (E16)
    sweep_results = rac_sensitivity_sweep(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        cost_ratios=cost_ratios,
        defer_budgets=defer_budgets,
        normal_only_cal=True,
    )

    return {
        "seed": seed,
        "binary_cp": binary_result,
        "rac_default": rac_default,
        "sensitivity_sweep": sweep_results,
    }


def aggregate_rac_results(all_results: list[dict]) -> dict:
    """Aggregate per-seed RAC results into mean +/- std with bootstrap CIs."""
    # Binary CP metrics
    binary_aurocs = np.array([r["binary_cp"]["auroc"] for r in all_results])
    binary_f1s = np.array([r["binary_cp"]["f1"] for r in all_results])
    binary_covs = np.array([r["binary_cp"]["coverage"] for r in all_results])

    # RAC default metrics
    rac_vars = np.array([r["rac_default"]["var_cost"] for r in all_results])
    binary_vars = np.array([r["rac_default"]["binary_var_cost"] for r in all_results])
    var_improvements = np.array([r["rac_default"]["var_improvement"] for r in all_results])
    rac_fnrs = np.array([r["rac_default"]["fnr"] for r in all_results])
    rac_fprs = np.array([r["rac_default"]["fpr"] for r in all_results])
    rac_defers = np.array([r["rac_default"]["defer_rate"] for r in all_results])
    rac_covs = np.array([r["rac_default"]["coverage_approve_normal"] for r in all_results])

    return {
        "binary_cp": {
            "auroc": {"mean": float(binary_aurocs.mean()), "std": float(binary_aurocs.std()),
                      **bootstrap_ci(binary_aurocs)},
            "f1": {"mean": float(binary_f1s.mean()), "std": float(binary_f1s.std()),
                   **bootstrap_ci(binary_f1s)},
            "coverage": {"mean": float(binary_covs.mean()), "std": float(binary_covs.std()),
                         **bootstrap_ci(binary_covs)},
        },
        "rac_default": {
            "var_cost": {"mean": float(rac_vars.mean()), "std": float(rac_vars.std()),
                         **bootstrap_ci(rac_vars)},
            "binary_var_cost": {"mean": float(binary_vars.mean()), "std": float(binary_vars.std()),
                                **bootstrap_ci(binary_vars)},
            "var_improvement": {"mean": float(var_improvements.mean()),
                                "std": float(var_improvements.std()),
                                **bootstrap_ci(var_improvements)},
            "fnr": {"mean": float(rac_fnrs.mean()), "std": float(rac_fnrs.std()),
                    **bootstrap_ci(rac_fnrs)},
            "fpr": {"mean": float(rac_fprs.mean()), "std": float(rac_fprs.std()),
                    **bootstrap_ci(rac_fprs)},
            "defer_rate": {"mean": float(rac_defers.mean()), "std": float(rac_defers.std()),
                           **bootstrap_ci(rac_defers)},
            "coverage": {"mean": float(rac_covs.mean()), "std": float(rac_covs.std()),
                         **bootstrap_ci(rac_covs)},
        },
        "n_seeds": len(all_results),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E13: RAC vs standard CP comparison")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--encoder-epochs", type=int, default=30)
    p.add_argument("--outcome-weight", type=float, default=0.5)
    p.add_argument(
        "--cost-ratios",
        type=str,
        default="1,2,5,10,20",
        help="Comma-separated cost ratios for sensitivity sweep",
    )
    p.add_argument(
        "--defer-budgets",
        type=str,
        default="0.05,0.10,0.15,0.20,0.30",
        help="Comma-separated defer budgets for sensitivity sweep",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)
    cost_ratios = [float(c) for c in args.cost_ratios.split(",")]
    defer_budgets = [float(d) for d in args.defer_budgets.split(",")]

    print(f"E13: RAC vs Standard CP Comparison on {device}")
    print(f"Seeds: {seeds}")
    print(f"Cost ratios: {cost_ratios}")
    print(f"Defer budgets: {defer_budgets}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    cal_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_traces],
        dtype=np.int32,
    )
    test_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in test_traces],
        dtype=np.int32,
    )

    print(f"Train: {len(train_traces)}, Cal: {len(cal_traces)}, Test: {len(test_traces)}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        result = run_rac_for_seed(
            seed,
            train_traces,
            cal_traces,
            test_traces,
            cal_labels,
            test_labels,
            vocab,
            device,
            args.alpha,
            args.encoder_epochs,
            args.outcome_weight,
            cost_ratios,
            defer_budgets,
        )
        rac_d = result["rac_default"]
        print(
            f"  Binary VaR={result['binary_cp']['auroc']:.4f} AUROC, "
            f"RAC VaR={rac_d['var_cost']:.4f}, "
            f"Improvement={rac_d['var_improvement']:.4f}"
        )
        print(
            f"  FNR={rac_d['fnr']:.4f}, FPR={rac_d['fpr']:.4f}, "
            f"Defer={rac_d['defer_rate']:.4f}, "
            f"Coverage={rac_d['coverage_approve_normal']:.4f}"
        )
        all_results.append(result)

    # Aggregate
    agg = aggregate_rac_results(all_results)

    print("\n" + "=" * 80)
    print("E13: RAC vs STANDARD CP (AGGREGATE)")
    print("=" * 80)

    print(f"\nBinary CP:")
    bp = agg["binary_cp"]
    print(f"  AUROC:    {bp['auroc']['mean']:.4f} +/- {bp['auroc']['std']:.3f}")
    print(f"  F1:       {bp['f1']['mean']:.4f} +/- {bp['f1']['std']:.3f}")
    print(f"  Coverage: {bp['coverage']['mean']:.4f} +/- {bp['coverage']['std']:.4f}")

    print(f"\nRAC (cost_ratio=5, defer_budget=15%):")
    rp = agg["rac_default"]
    print(f"  VaR Cost:    {rp['var_cost']['mean']:.4f} +/- {rp['var_cost']['std']:.3f}")
    print(f"  Binary VaR:  {rp['binary_var_cost']['mean']:.4f} +/- {rp['binary_var_cost']['std']:.3f}")
    print(f"  Improvement: {rp['var_improvement']['mean']:.4f} +/- {rp['var_improvement']['std']:.3f}")
    print(f"  FNR:         {rp['fnr']['mean']:.4f} +/- {rp['fnr']['std']:.3f}")
    print(f"  FPR:         {rp['fpr']['mean']:.4f} +/- {rp['fpr']['std']:.3f}")
    print(f"  Defer Rate:  {rp['defer_rate']['mean']:.4f} +/- {rp['defer_rate']['std']:.3f}")
    print(f"  Coverage:    {rp['coverage']['mean']:.4f} +/- {rp['coverage']['std']:.4f}")

    # Save results
    output = {
        "config": {
            "seeds": seeds,
            "alpha": args.alpha,
            "encoder_epochs": args.encoder_epochs,
            "outcome_weight": args.outcome_weight,
            "cost_ratios": cost_ratios,
            "defer_budgets": defer_budgets,
        },
        "aggregate": agg,
        "per_seed": all_results,
    }
    out_path = tables_dir / "rac_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
