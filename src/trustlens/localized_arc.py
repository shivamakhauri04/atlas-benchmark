"""Localized Adaptive Risk Control — per-tool conditional coverage guarantees.

Implements Theorem 5: For each tool category t,
    P(Y in C_t(X)) >= 1 - alpha_t

where C_t is the conformal prediction set calibrated on category-specific
scores and alpha_t is the per-category miscoverage target.

Based on:
    Zecchin & Simeone — Localized Adaptive Risk Control (arXiv May 2024)

The key insight: marginal coverage (95% overall) can hide severe
undercoverage on high-risk tools. A system that achieves 99% coverage
on `file_read` but only 70% on `database_drop` is dangerous. Localized
ARC provides independent guarantees per tool category.

The RKHS threshold functions are simplified here to per-category quantile
thresholds — the kernel-based formulation in Zecchin & Simeone reduces to
independent split CP when the feature space is discrete (tool categories).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PerToolThreshold:
    """Calibrated threshold for a single tool category."""

    category: str
    tau: float
    n_cal: int
    alpha: float


@dataclass(frozen=True)
class LocalizedARCResult:
    """Collection of per-tool thresholds from Localized ARC calibration."""

    thresholds: dict[str, PerToolThreshold]
    global_tau: float
    alpha: float
    min_samples_per_category: int


def get_tool_category(tool_name: str) -> str:
    """Look up the category for a tool name from the registry.

    Falls back to "unknown" for tools not in the registry.
    """
    from data.tool_registry import TOOL_CATEGORIES

    for category, tools in TOOL_CATEGORIES.items():
        for name, _risky in tools:
            if name == tool_name:
                return category
    return "unknown"


def build_category_map() -> dict[str, str]:
    """Build a tool_name -> category mapping from the registry."""
    from data.tool_registry import TOOL_CATEGORIES

    mapping = {}
    for category, tools in TOOL_CATEGORIES.items():
        for name, _risky in tools:
            mapping[name] = category
    return mapping


def extract_trace_categories(
    traces: list,
    category_map: dict[str, str] | None = None,
) -> np.ndarray:
    """Extract the dominant tool category for each trace.

    The dominant category is the most frequent tool category in the trace.
    Returns an array of category strings, one per trace.
    """
    if category_map is None:
        category_map = build_category_map()

    categories = []
    for trace in traces:
        cat_counts: dict[str, int] = {}
        for action in trace.actions:
            cat = category_map.get(action.tool, "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if cat_counts:
            dominant = max(cat_counts, key=cat_counts.get)
        else:
            dominant = "unknown"
        categories.append(dominant)
    return np.array(categories)


def extract_action_categories(
    traces: list,
    category_map: dict[str, str] | None = None,
) -> list[list[str]]:
    """Extract per-action tool categories for each trace.

    Returns a list of lists: categories[i][j] = category of trace i, action j.
    """
    if category_map is None:
        category_map = build_category_map()

    all_cats = []
    for trace in traces:
        trace_cats = [
            category_map.get(action.tool, "unknown") for action in trace.actions
        ]
        all_cats.append(trace_cats)
    return all_cats


def adaptive_alpha_allocation(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    cal_categories: np.ndarray,
    base_alpha: float = 0.05,
    method: str = "bonferroni",
) -> dict[str, float]:
    """Allocate per-category alpha budgets to improve worst-case coverage.

    The issue: uniform alpha across categories fails when score distributions
    vary. Monitoring traces have high overlap between normal/anomalous scores,
    so they need a tighter (lower) alpha to maintain coverage.

    Args:
        cal_scores: Calibration scores. Shape (n,).
        cal_labels: Binary labels. Shape (n,).
        cal_categories: Category for each sample. Shape (n,).
        base_alpha: Overall miscoverage budget.
        method: Allocation strategy:
            "bonferroni" — equal split: alpha_t = base_alpha / n_categories
            "overlap_weighted" — harder categories (more overlap) get tighter alpha

    Returns:
        Dict mapping category -> allocated alpha.
    """
    unique_cats = sorted(set(cal_categories))
    n_cats = len(unique_cats)

    if method == "bonferroni":
        return {cat: base_alpha / n_cats for cat in unique_cats}

    if method == "overlap_weighted":
        overlaps = {}
        for cat in unique_cats:
            mask = cal_categories == cat
            cat_scores = cal_scores[mask]
            cat_labels = cal_labels[mask]
            normal = cat_scores[cat_labels == 0]
            anomalous = cat_scores[cat_labels == 1]
            if len(normal) < 2 or len(anomalous) < 2:
                overlaps[cat] = 0.5
                continue
            n_med = float(np.median(normal))
            a_med = float(np.median(anomalous))
            if abs(a_med - n_med) < 1e-10:
                overlaps[cat] = 1.0
            else:
                combined_std = float(np.std(np.concatenate([normal, anomalous])))
                overlaps[cat] = combined_std / max(abs(a_med - n_med), 1e-10)

        total_overlap = sum(overlaps.values())
        if total_overlap < 1e-10:
            return {cat: base_alpha / n_cats for cat in unique_cats}

        allocated = {}
        for cat in unique_cats:
            weight = overlaps[cat] / total_overlap
            allocated[cat] = base_alpha * weight
        return allocated

    raise ValueError(f"Unknown method: {method}")


def localized_arc_calibrate(
    scores: np.ndarray | torch.Tensor,
    categories: np.ndarray,
    *,
    alpha: float = 0.05,
    min_samples: int = 10,
    per_category_alpha: dict[str, float] | None = None,
) -> LocalizedARCResult:
    """Calibrate per-category conformal thresholds (Theorem 5).

    For each tool category with sufficient calibration samples, computes
    an independent split CP threshold at level alpha. Categories with
    fewer than min_samples fall back to the global threshold.

    This is the discrete-feature specialization of Localized ARC:
    when the conditioning variable is categorical (tool type), the
    RKHS kernel reduces to indicator functions and the optimal threshold
    function is piecewise constant over categories.

    Args:
        scores: Calibration non-conformity scores. Shape (n,).
        categories: Category label for each score. Shape (n,).
        alpha: Target miscoverage rate per category.
        min_samples: Minimum calibration samples per category.
            Categories below this threshold use the global tau.

    Returns:
        LocalizedARCResult with per-category thresholds.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64)

    n = len(scores)
    q_global = np.ceil((1 - alpha) * (n + 1)) / n
    q_global = min(q_global, 1.0)
    global_tau = float(np.quantile(scores, q_global)) if n > 0 else float("inf")

    unique_cats = sorted(set(categories))
    thresholds = {}

    for cat in unique_cats:
        mask = categories == cat
        cat_scores = scores[mask]
        n_cat = len(cat_scores)
        cat_alpha = per_category_alpha.get(cat, alpha) if per_category_alpha else alpha

        if n_cat >= min_samples:
            q_cat = np.ceil((1 - cat_alpha) * (n_cat + 1)) / n_cat
            q_cat = min(q_cat, 1.0)
            tau_cat = float(np.quantile(cat_scores, q_cat))
        else:
            tau_cat = global_tau

        thresholds[cat] = PerToolThreshold(
            category=cat,
            tau=tau_cat,
            n_cal=n_cat,
            alpha=cat_alpha,
        )

    return LocalizedARCResult(
        thresholds=thresholds,
        global_tau=global_tau,
        alpha=alpha,
        min_samples_per_category=min_samples,
    )


def localized_arc_predict(
    scores: np.ndarray | torch.Tensor,
    categories: np.ndarray,
    arc_result: LocalizedARCResult,
) -> np.ndarray:
    """Apply per-category thresholds to predict anomalies.

    Each score is compared against the threshold for its tool category.
    Unknown categories use the global threshold.

    Returns:
        Binary predictions (1 = anomalous, 0 = normal). Shape (n,).
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64)

    preds = np.zeros(len(scores), dtype=np.int32)
    for i, (s, cat) in enumerate(zip(scores, categories)):
        if cat in arc_result.thresholds:
            tau = arc_result.thresholds[cat].tau
        else:
            tau = arc_result.global_tau
        if s > tau:
            preds[i] = 1
    return preds


def evaluate_localized_arc(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    cal_categories: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_categories: np.ndarray,
    *,
    alpha: float = 0.05,
    min_samples: int = 10,
    normal_only_cal: bool = True,
    per_category_alpha: dict[str, float] | None = None,
) -> dict:
    """Evaluate Localized ARC and report per-category coverage.

    Args:
        cal_scores, cal_labels, cal_categories: Calibration data.
        test_scores, test_labels, test_categories: Test data.
        alpha: Target miscoverage rate per category.
        min_samples: Minimum samples per category for independent threshold.
        normal_only_cal: If True, calibrate on normal samples only.

    Returns:
        Dict with per-category coverage, overall coverage, and comparison
        with standard (marginal) CP.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    from trustlens.conformal import split_conformal_calibrate, split_conformal_predict

    # Filter to normal for calibration if requested
    if normal_only_cal:
        normal_mask = cal_labels == 0
        cal_scores_for_cp = cal_scores[normal_mask]
        cal_cats_for_cp = cal_categories[normal_mask]
    else:
        cal_scores_for_cp = cal_scores
        cal_cats_for_cp = cal_categories

    # Localized ARC calibration
    arc_result = localized_arc_calibrate(
        cal_scores_for_cp,
        cal_cats_for_cp,
        alpha=alpha,
        min_samples=min_samples,
        per_category_alpha=per_category_alpha,
    )

    # Localized predictions
    loc_preds = localized_arc_predict(test_scores, test_categories, arc_result)

    # Standard (marginal) CP for comparison
    global_tau = split_conformal_calibrate(cal_scores_for_cp, alpha=alpha)
    global_preds = split_conformal_predict(test_scores, global_tau)

    # Overall metrics
    n_normal = int((test_labels == 0).sum())
    n_anomalous = int((test_labels == 1).sum())

    loc_coverage = float(
        (loc_preds[test_labels == 0] == 0).sum() / max(n_normal, 1)
    )
    global_coverage = float(
        (global_preds[test_labels == 0] == 0).sum() / max(n_normal, 1)
    )

    loc_f1 = float(f1_score(test_labels, loc_preds, zero_division=0.0))
    global_f1 = float(f1_score(test_labels, global_preds, zero_division=0.0))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = float("nan")

    # Per-category breakdown
    per_category = {}
    unique_cats = sorted(set(test_categories))

    for cat in unique_cats:
        cat_mask = test_categories == cat
        cat_labels = test_labels[cat_mask]
        cat_loc_preds = loc_preds[cat_mask]
        cat_global_preds = global_preds[cat_mask]

        n_cat = int(cat_mask.sum())
        n_cat_normal = int((cat_labels == 0).sum())
        n_cat_anom = int((cat_labels == 1).sum())

        cat_loc_cov = float(
            (cat_loc_preds[cat_labels == 0] == 0).sum() / max(n_cat_normal, 1)
        )
        cat_global_cov = float(
            (cat_global_preds[cat_labels == 0] == 0).sum() / max(n_cat_normal, 1)
        )

        cat_loc_recall = float(
            (cat_loc_preds[cat_labels == 1] == 1).sum() / max(n_cat_anom, 1)
        )
        cat_global_recall = float(
            (cat_global_preds[cat_labels == 1] == 1).sum() / max(n_cat_anom, 1)
        )

        threshold_info = arc_result.thresholds.get(cat)
        per_category[cat] = {
            "n_total": n_cat,
            "n_normal": n_cat_normal,
            "n_anomalous": n_cat_anom,
            "localized_coverage": cat_loc_cov,
            "global_coverage": cat_global_cov,
            "coverage_diff": cat_loc_cov - cat_global_cov,
            "localized_recall": cat_loc_recall,
            "global_recall": cat_global_recall,
            "tau": threshold_info.tau if threshold_info else arc_result.global_tau,
            "n_cal": threshold_info.n_cal if threshold_info else 0,
            "uses_global_fallback": (
                threshold_info is None
                or threshold_info.n_cal < min_samples
            ),
            "coverage_meets_target": cat_loc_cov >= (1 - alpha - 0.01),
        }

    # Summary statistics
    cats_meeting_target = sum(
        1 for v in per_category.values() if v["coverage_meets_target"]
    )
    cats_with_own_threshold = sum(
        1 for v in per_category.values() if not v["uses_global_fallback"]
    )

    # Worst-case category coverage (the critical metric for Theorem 5)
    cat_coverages = [
        v["localized_coverage"]
        for v in per_category.values()
        if v["n_normal"] >= 5  # skip tiny categories
    ]
    worst_case_coverage = min(cat_coverages) if cat_coverages else 0.0

    return {
        "alpha": alpha,
        "min_samples": min_samples,
        "calibration": "normal_only" if normal_only_cal else "mixed",
        "auroc": auroc,
        "localized_overall_coverage": loc_coverage,
        "global_overall_coverage": global_coverage,
        "localized_f1": loc_f1,
        "global_f1": global_f1,
        "worst_case_coverage": worst_case_coverage,
        "n_categories": len(unique_cats),
        "categories_meeting_target": cats_meeting_target,
        "categories_with_own_threshold": cats_with_own_threshold,
        "global_tau": global_tau,
        "per_category": per_category,
        "n_test": len(test_labels),
        "n_anomalous": n_anomalous,
    }
