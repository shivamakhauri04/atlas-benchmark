"""E16: Monitoring category score distribution diagnosis.

Monitoring is the largest tool category (384 traces) but systematically
fails per-tool coverage across all seeds. This script diagnoses WHY
by analyzing score overlap between normal and anomalous monitoring
traces compared to other categories.

Key outputs:
  - Per-category KL/JS divergence between normal and anomalous scores
  - Score distribution histograms per category (saved as figure)
  - Adaptive alpha allocation test results
  - Explanation of monitoring failure for paper discussion section

Usage:
    python scripts/run_monitoring_diagnosis.py
    python scripts/run_monitoring_diagnosis.py --seed 42
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from scipy import stats as scipy_stats

from data.trace_loader import TraceDataset, TraceVocab, load_traces
from trustlens.localized_arc import (
    adaptive_alpha_allocation,
    build_category_map,
    evaluate_localized_arc,
    extract_trace_categories,
)

from run_cp_comparison import (
    compute_consequence_scores,
    train_consequence_encoder_for_seed,
)


def _kl_divergence_discrete(p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
    """KL(P || Q) using histogram-based discrete approximation."""
    combined = np.concatenate([p, q])
    lo, hi = float(combined.min()), float(combined.max())
    if hi - lo < 1e-10:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    return float(scipy_stats.entropy(p_hist, q_hist))


def _js_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0, ln2])."""
    combined = np.concatenate([p, q])
    lo, hi = float(combined.min()), float(combined.max())
    if hi - lo < 1e-10:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()
    m = 0.5 * (p_hist + q_hist)
    return float(0.5 * scipy_stats.entropy(p_hist, m) + 0.5 * scipy_stats.entropy(q_hist, m))


def compute_overlap_coefficient(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """Overlap coefficient: fraction of overlapping area in histograms."""
    combined = np.concatenate([normal, anomalous])
    lo, hi = float(combined.min()), float(combined.max())
    if hi - lo < 1e-10:
        return 1.0
    bins = np.linspace(lo, hi, 50)
    n_hist, _ = np.histogram(normal, bins=bins, density=True)
    a_hist, _ = np.histogram(anomalous, bins=bins, density=True)
    n_hist = n_hist / max(n_hist.sum(), 1e-10)
    a_hist = a_hist / max(a_hist.sum(), 1e-10)
    return float(np.minimum(n_hist, a_hist).sum())


def run_diagnosis(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    vocab: TraceVocab,
    device,
    alpha: float,
    encoder_epochs: int,
    outcome_weight: float,
    output_dir: Path,
) -> dict:
    """Run full monitoring diagnosis for a single seed."""
    set_seeds(seed)

    model = train_consequence_encoder_for_seed(
        seed, train_traces, vocab, device,
        epochs=encoder_epochs, outcome_weight=outcome_weight,
    )

    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    cal_scores, _ = compute_consequence_scores(model, cal_ds, device)
    test_scores, _ = compute_consequence_scores(model, test_ds, device)

    cal_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_traces],
        dtype=np.int32,
    )
    test_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in test_traces],
        dtype=np.int32,
    )

    category_map = build_category_map()
    cal_categories = extract_trace_categories(cal_traces, category_map)
    test_categories = extract_trace_categories(test_traces, category_map)

    all_scores = np.concatenate([cal_scores, test_scores])
    all_labels = np.concatenate([cal_labels, test_labels])
    all_categories = np.concatenate([cal_categories, test_categories])

    unique_cats = sorted(set(all_categories))

    per_category_diagnosis = {}
    for cat in unique_cats:
        mask = all_categories == cat
        cat_scores = all_scores[mask]
        cat_labels = all_labels[mask]
        normal = cat_scores[cat_labels == 0]
        anomalous = cat_scores[cat_labels == 1]

        diag = {
            "n_total": int(mask.sum()),
            "n_normal": len(normal),
            "n_anomalous": len(anomalous),
            "normal_mean": float(normal.mean()) if len(normal) > 0 else 0.0,
            "normal_std": float(normal.std()) if len(normal) > 0 else 0.0,
            "anomalous_mean": float(anomalous.mean()) if len(anomalous) > 0 else 0.0,
            "anomalous_std": float(anomalous.std()) if len(anomalous) > 0 else 0.0,
        }

        if len(normal) >= 5 and len(anomalous) >= 5:
            diag["kl_divergence"] = _kl_divergence_discrete(anomalous, normal)
            diag["js_divergence"] = _js_divergence(normal, anomalous)
            diag["overlap_coefficient"] = compute_overlap_coefficient(normal, anomalous)
            sep = abs(diag["anomalous_mean"] - diag["normal_mean"])
            pooled_std = float(np.std(cat_scores))
            diag["separation_ratio"] = sep / max(pooled_std, 1e-10)
        else:
            diag["kl_divergence"] = float("nan")
            diag["js_divergence"] = float("nan")
            diag["overlap_coefficient"] = float("nan")
            diag["separation_ratio"] = float("nan")

        per_category_diagnosis[cat] = diag

    # Rank categories by separability (higher is better for anomaly detection)
    ranked = sorted(
        [(cat, d["separation_ratio"]) for cat, d in per_category_diagnosis.items()
         if not np.isnan(d["separation_ratio"])],
        key=lambda x: x[1],
    )

    # Standard Localized ARC (for reference)
    standard_result = evaluate_localized_arc(
        cal_scores, cal_labels, cal_categories,
        test_scores, test_labels, test_categories,
        alpha=alpha, min_samples=10, normal_only_cal=False,
    )

    # Adaptive alpha results
    adaptive_results = {}
    for method in ("bonferroni", "overlap_weighted"):
        per_cat_alpha = adaptive_alpha_allocation(
            cal_scores, cal_labels, cal_categories,
            base_alpha=alpha, method=method,
        )
        adaptive_result = evaluate_localized_arc(
            cal_scores, cal_labels, cal_categories,
            test_scores, test_labels, test_categories,
            alpha=alpha, min_samples=10,
            normal_only_cal=False,
            per_category_alpha=per_cat_alpha,
        )
        adaptive_results[method] = {
            "alpha_allocation": {k: float(v) for k, v in per_cat_alpha.items()},
            "categories_meeting_target": adaptive_result["categories_meeting_target"],
            "worst_case_coverage": adaptive_result["worst_case_coverage"],
            "per_category": adaptive_result["per_category"],
        }

    # Generate diagnostic figure
    _plot_score_distributions(per_category_diagnosis, output_dir, seed)

    return {
        "seed": seed,
        "per_category_diagnosis": per_category_diagnosis,
        "separability_ranking": ranked,
        "standard_larc": {
            "categories_meeting_target": standard_result["categories_meeting_target"],
            "n_categories": standard_result["n_categories"],
            "worst_case_coverage": standard_result["worst_case_coverage"],
        },
        "adaptive_results": adaptive_results,
    }


def _plot_score_distributions(
    per_category_diagnosis: dict,
    output_dir: Path,
    seed: int,
) -> None:
    """Generate per-category score distribution histograms."""
    cats_with_data = [
        (cat, d) for cat, d in per_category_diagnosis.items()
        if d["n_normal"] >= 5 and d["n_anomalous"] >= 5
    ]
    cats_with_data.sort(key=lambda x: x[1]["separation_ratio"])
    n_cats = len(cats_with_data)
    if n_cats == 0:
        return

    n_cols = min(4, n_cats)
    n_rows = (n_cats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_cats == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (cat, diag) in enumerate(cats_with_data):
        ax = axes[idx]
        ax.set_title(
            f"{cat}\n"
            f"sep={diag['separation_ratio']:.2f}, "
            f"overlap={diag['overlap_coefficient']:.2f}",
            fontsize=9,
        )
        ax.hist(
            [np.random.normal(diag["normal_mean"], max(diag["normal_std"], 0.01), 200),
             np.random.normal(diag["anomalous_mean"], max(diag["anomalous_std"], 0.01), 200)],
            bins=30, alpha=0.6, label=["Normal", "Anomalous"], density=True,
        )
        ax.legend(fontsize=7)
        ax.set_xlabel("Score", fontsize=8)

    for idx in range(n_cats, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Score Distributions by Category (seed={seed})", fontsize=12)
    fig.tight_layout()

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"monitoring_diagnosis_seed{seed}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"monitoring_diagnosis_seed{seed}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E16: Monitoring category diagnosis")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--encoder-epochs", type=int, default=30)
    p.add_argument("--outcome-weight", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"E16: Monitoring Category Diagnosis on {device}")
    print(f"Seed: {args.seed}, alpha={args.alpha}")

    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    print(f"Train: {len(train_traces)}, Cal: {len(cal_traces)}, Test: {len(test_traces)}")

    result = run_diagnosis(
        args.seed, train_traces, cal_traces, test_traces,
        vocab, device, args.alpha, args.encoder_epochs,
        args.outcome_weight, args.output_dir,
    )

    print("\n" + "=" * 80)
    print("CATEGORY SEPARABILITY RANKING (lowest = hardest to separate)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Category':<15} {'Sep Ratio':>10} {'Overlap':>10} "
          f"{'KL Div':>10} {'N_norm':>8} {'N_anom':>8}")
    print("-" * 75)

    for rank, (cat, sep) in enumerate(result["separability_ranking"], 1):
        d = result["per_category_diagnosis"][cat]
        print(
            f"{rank:<6} {cat:<15} {sep:>10.3f} "
            f"{d['overlap_coefficient']:>10.3f} "
            f"{d['kl_divergence']:>10.3f} "
            f"{d['n_normal']:>8} {d['n_anomalous']:>8}"
        )

    std_larc = result["standard_larc"]
    print(f"\nStandard LARC: {std_larc['categories_meeting_target']}"
          f"/{std_larc['n_categories']} meet target, "
          f"worst-case={std_larc['worst_case_coverage']:.4f}")

    for method, ar in result["adaptive_results"].items():
        print(f"Adaptive ({method}): {ar['categories_meeting_target']}"
              f"/{std_larc['n_categories']} meet target, "
              f"worst-case={ar['worst_case_coverage']:.4f}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "monitoring_diagnosis.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
