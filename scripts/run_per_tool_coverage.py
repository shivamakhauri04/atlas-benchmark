"""E14: Per-tool conditional coverage evaluation.

Evaluates Localized ARC (Theorem 5) — per-tool-category coverage
guarantees vs standard marginal coverage.

Key metrics:
  - Per-category coverage: P(approve | normal, category=t) for each t
  - Worst-case category coverage (the critical Theorem 5 metric)
  - Number of categories meeting the 1-alpha target independently
  - Comparison with global (marginal) CP coverage

Usage:
    python scripts/run_per_tool_coverage.py
    python scripts/run_per_tool_coverage.py --seeds 42,123,456,789,1024
"""

import argparse
import json
from pathlib import Path

import numpy as np
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from trustlens.localized_arc import (
    adaptive_alpha_allocation,
    build_category_map,
    evaluate_localized_arc,
    extract_trace_categories,
    localized_arc_calibrate,
    localized_arc_predict,
)
from trustlens.stats import bootstrap_ci

from run_cp_comparison import (
    compute_consequence_scores,
    train_consequence_encoder_for_seed,
)


def run_per_tool_for_seed(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    cal_categories: np.ndarray,
    test_categories: np.ndarray,
    vocab: TraceVocab,
    device,
    alpha: float,
    encoder_epochs: int,
    outcome_weight: float,
    min_samples: int,
) -> dict:
    """Run per-tool coverage evaluation for one seed."""
    set_seeds(seed)

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

    # Primary: mixed-cal (v12 — formal guarantee holds)
    result = evaluate_localized_arc(
        cal_scores,
        cal_labels,
        cal_categories,
        test_scores,
        test_labels,
        test_categories,
        alpha=alpha,
        min_samples=min_samples,
        normal_only_cal=False,
    )
    result["seed"] = seed

    # Adaptive alpha pass: allocate per-category alpha budgets
    for method in ("bonferroni", "overlap_weighted"):
        per_cat_alpha = adaptive_alpha_allocation(
            cal_scores, cal_labels, cal_categories,
            base_alpha=alpha, method=method,
        )
        adaptive_result = evaluate_localized_arc(
            cal_scores,
            cal_labels,
            cal_categories,
            test_scores,
            test_labels,
            test_categories,
            alpha=alpha,
            min_samples=min_samples,
            normal_only_cal=False,
            per_category_alpha=per_cat_alpha,
        )
        adaptive_result["alpha_allocation"] = {
            k: float(v) for k, v in per_cat_alpha.items()
        }
        adaptive_result["allocation_method"] = method
        result[f"adaptive_{method}"] = adaptive_result

    return result


def aggregate_per_tool_results(all_results: list[dict]) -> dict:
    """Aggregate per-seed per-tool results into mean +/- std."""
    # Overall metrics
    loc_covs = np.array([r["localized_overall_coverage"] for r in all_results])
    glob_covs = np.array([r["global_overall_coverage"] for r in all_results])
    loc_f1s = np.array([r["localized_f1"] for r in all_results])
    glob_f1s = np.array([r["global_f1"] for r in all_results])
    worst_covs = np.array([r["worst_case_coverage"] for r in all_results])
    cats_meeting = np.array([r["categories_meeting_target"] for r in all_results])

    agg = {
        "overall": {
            "localized_coverage": {
                "mean": float(loc_covs.mean()),
                "std": float(loc_covs.std()),
                **bootstrap_ci(loc_covs),
            },
            "global_coverage": {
                "mean": float(glob_covs.mean()),
                "std": float(glob_covs.std()),
                **bootstrap_ci(glob_covs),
            },
            "localized_f1": {
                "mean": float(loc_f1s.mean()),
                "std": float(loc_f1s.std()),
                **bootstrap_ci(loc_f1s),
            },
            "global_f1": {
                "mean": float(glob_f1s.mean()),
                "std": float(glob_f1s.std()),
                **bootstrap_ci(glob_f1s),
            },
            "worst_case_coverage": {
                "mean": float(worst_covs.mean()),
                "std": float(worst_covs.std()),
                **bootstrap_ci(worst_covs),
            },
            "categories_meeting_target": {
                "mean": float(cats_meeting.mean()),
                "std": float(cats_meeting.std()),
            },
        },
        "n_seeds": len(all_results),
    }

    # Per-category aggregation
    all_cats = set()
    for r in all_results:
        all_cats.update(r["per_category"].keys())

    per_cat_agg = {}
    for cat in sorted(all_cats):
        cat_loc_covs = []
        cat_glob_covs = []
        cat_loc_recalls = []
        cat_ns = []
        for r in all_results:
            if cat in r["per_category"]:
                pc = r["per_category"][cat]
                cat_loc_covs.append(pc["localized_coverage"])
                cat_glob_covs.append(pc["global_coverage"])
                cat_loc_recalls.append(pc["localized_recall"])
                cat_ns.append(pc["n_total"])

        if cat_loc_covs:
            loc_arr = np.array(cat_loc_covs)
            glob_arr = np.array(cat_glob_covs)
            recall_arr = np.array(cat_loc_recalls)

            per_cat_agg[cat] = {
                "localized_coverage_mean": float(loc_arr.mean()),
                "localized_coverage_std": float(loc_arr.std()),
                "global_coverage_mean": float(glob_arr.mean()),
                "global_coverage_std": float(glob_arr.std()),
                "coverage_diff_mean": float((loc_arr - glob_arr).mean()),
                "localized_recall_mean": float(recall_arr.mean()),
                "localized_recall_std": float(recall_arr.std()),
                "mean_n_traces": float(np.mean(cat_ns)),
                "meets_target_fraction": float(
                    np.mean(loc_arr >= (1 - all_results[0]["alpha"] - 0.01))
                ),
            }

    agg["per_category"] = per_cat_agg
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E14: Per-tool conditional coverage")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--encoder-epochs", type=int, default=30)
    p.add_argument("--outcome-weight", type=float, default=0.5)
    p.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Min calibration samples per category for independent threshold",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)

    print(f"E14: Per-Tool Conditional Coverage on {device}")
    print(f"Seeds: {seeds}, alpha={args.alpha}, min_samples={args.min_samples}")

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

    # Extract tool categories for each trace
    category_map = build_category_map()
    cal_categories = extract_trace_categories(cal_traces, category_map)
    test_categories = extract_trace_categories(test_traces, category_map)

    print(f"Train: {len(train_traces)}, Cal: {len(cal_traces)}, Test: {len(test_traces)}")
    unique_cats = sorted(set(test_categories))
    print(f"Tool categories found: {len(unique_cats)}: {unique_cats}")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        result = run_per_tool_for_seed(
            seed,
            train_traces,
            cal_traces,
            test_traces,
            cal_labels,
            test_labels,
            cal_categories,
            test_categories,
            vocab,
            device,
            args.alpha,
            args.encoder_epochs,
            args.outcome_weight,
            args.min_samples,
        )

        print(
            f"  Overall: Localized cov={result['localized_overall_coverage']:.4f}, "
            f"Global cov={result['global_overall_coverage']:.4f}, "
            f"Worst-case={result['worst_case_coverage']:.4f}"
        )
        print(
            f"  Categories meeting target: "
            f"{result['categories_meeting_target']}/{result['n_categories']}"
        )
        all_results.append(result)

    # Aggregate
    agg = aggregate_per_tool_results(all_results)

    print("\n" + "=" * 80)
    print("E14: PER-TOOL CONDITIONAL COVERAGE (AGGREGATE)")
    print("=" * 80)

    ov = agg["overall"]
    print(f"\nOverall Coverage:")
    print(
        f"  Localized: {ov['localized_coverage']['mean']:.4f} "
        f"+/- {ov['localized_coverage']['std']:.4f}"
    )
    print(
        f"  Global:    {ov['global_coverage']['mean']:.4f} "
        f"+/- {ov['global_coverage']['std']:.4f}"
    )
    print(
        f"  Worst-case: {ov['worst_case_coverage']['mean']:.4f} "
        f"+/- {ov['worst_case_coverage']['std']:.4f}"
    )
    print(
        f"  Categories meeting target: "
        f"{ov['categories_meeting_target']['mean']:.1f} "
        f"+/- {ov['categories_meeting_target']['std']:.1f}"
    )

    print(f"\nPer-Category Breakdown:")
    print(
        f"  {'Category':<15} {'Loc Cov':>10} {'Glob Cov':>10} "
        f"{'Diff':>8} {'Recall':>10} {'N':>6} {'Meets?':>8}"
    )
    print("  " + "-" * 70)
    for cat, cv in sorted(agg["per_category"].items()):
        meets = f"{cv['meets_target_fraction'] * 100:.0f}%"
        print(
            f"  {cat:<15} "
            f"{cv['localized_coverage_mean']:.4f}     "
            f"{cv['global_coverage_mean']:.4f}     "
            f"{cv['coverage_diff_mean']:>+.4f} "
            f"{cv['localized_recall_mean']:.4f}     "
            f"{cv['mean_n_traces']:>5.0f} "
            f"{meets:>8}"
        )

    # Save results
    output = {
        "config": {
            "seeds": seeds,
            "alpha": args.alpha,
            "encoder_epochs": args.encoder_epochs,
            "outcome_weight": args.outcome_weight,
            "min_samples": args.min_samples,
        },
        "aggregate": agg,
        "per_seed": all_results,
    }
    out_path = tables_dir / "per_tool_coverage.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
