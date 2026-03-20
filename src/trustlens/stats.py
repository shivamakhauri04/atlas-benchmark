"""Statistical testing utilities for multi-seed experiment analysis.

Provides:
- Bootstrap confidence intervals (primary for small N)
- Wilcoxon signed-rank tests (for N >= 6 paired observations)
- Benjamini-Hochberg correction for multiple comparisons
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for a 1-D array of observations.

    Args:
        values: Array of metric values (e.g., AUROC across seeds).
        confidence: Confidence level (default 0.95).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with mean, std, ci_lower, ci_upper, n_obs.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"), "n_obs": 0}

    if n == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "ci_lower": v, "ci_upper": v, "n_obs": 1}

    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "ci_lower": lo,
        "ci_upper": hi,
        "n_obs": n,
    }


def paired_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    method: str = "auto",
) -> dict:
    """Run a paired statistical test between two methods' per-seed results.

    For N >= 6: Wilcoxon signed-rank test (two-sided).
    For N < 6: Bootstrap difference CI (Wilcoxon can't achieve p < 0.05).

    Args:
        values_a: Per-seed metric for method A.
        values_b: Per-seed metric for method B.
        method: "wilcoxon", "bootstrap", or "auto" (default).

    Returns:
        Dict with test_used, statistic, p_value, difference_mean,
        difference_ci_lower, difference_ci_upper.
    """
    n = len(values_a)
    diffs = values_a - values_b

    if method == "auto":
        method = "wilcoxon" if n >= 6 else "bootstrap"

    result = {
        "n_pairs": n,
        "difference_mean": float(diffs.mean()),
        "difference_std": float(diffs.std()),
    }

    if method == "wilcoxon" and n >= 6:
        try:
            stat, p = stats.wilcoxon(values_a, values_b, alternative="two-sided")
            result["test_used"] = "wilcoxon"
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
        except ValueError:
            result["test_used"] = "wilcoxon_failed"
            result["statistic"] = float("nan")
            result["p_value"] = float("nan")
    else:
        result["test_used"] = "bootstrap_ci"
        result["statistic"] = float("nan")
        result["p_value"] = float("nan")

    ci = bootstrap_ci(diffs)
    result["difference_ci_lower"] = ci["ci_lower"]
    result["difference_ci_upper"] = ci["ci_upper"]
    result["significant_by_ci"] = bool(
        ci["ci_lower"] > 0 or ci["ci_upper"] < 0
    )

    return result


def correct_multiple_tests(
    p_values: list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> dict:
    """Apply Benjamini-Hochberg (or other) correction for multiple testing.

    Args:
        p_values: List of raw p-values from pairwise tests.
        method: Correction method (default "fdr_bh" for Benjamini-Hochberg).
        alpha: Family-wise error rate.

    Returns:
        Dict with corrected_p_values, rejected (bool per test),
        method, alpha.
    """
    valid_mask = [not np.isnan(p) for p in p_values]
    valid_ps = [p for p, v in zip(p_values, valid_mask) if v]

    if not valid_ps:
        return {
            "corrected_p_values": [float("nan")] * len(p_values),
            "rejected": [False] * len(p_values),
            "method": method,
            "alpha": alpha,
        }

    rejected_valid, corrected_valid, _, _ = multipletests(
        valid_ps, alpha=alpha, method=method,
    )

    corrected = []
    rejected = []
    valid_idx = 0
    for is_valid in valid_mask:
        if is_valid:
            corrected.append(float(corrected_valid[valid_idx]))
            rejected.append(bool(rejected_valid[valid_idx]))
            valid_idx += 1
        else:
            corrected.append(float("nan"))
            rejected.append(False)

    return {
        "corrected_p_values": corrected,
        "rejected": rejected,
        "method": method,
        "alpha": alpha,
    }


def compute_pairwise_stats(
    aggregate: dict,
    reference_scorer: str,
    metric: str = "auroc",
    per_seed_results: list[dict] | None = None,
) -> list[dict]:
    """Compute pairwise statistical tests against a reference scorer.

    Args:
        aggregate: Dict of {scorer_name: {metric_mean, metric_std, ...}}.
        reference_scorer: Name of the baseline to compare against.
        metric: Which metric to test (e.g., "auroc", "f1").
        per_seed_results: Raw per-seed results list for extracting
            individual values. Each dict has "scorer" and the metric key.

    Returns:
        List of dicts with comparison results (one per non-reference scorer).
    """
    if per_seed_results is None:
        return []

    ref_values = np.array([
        r[metric] for r in per_seed_results if r["scorer"] == reference_scorer
    ])

    comparisons = []
    p_values = []

    for scorer_name in sorted(aggregate.keys()):
        if scorer_name == reference_scorer:
            continue
        if "(mixed cal)" in scorer_name:
            continue

        other_values = np.array([
            r[metric] for r in per_seed_results if r["scorer"] == scorer_name
        ])

        if len(ref_values) == 0 or len(other_values) == 0:
            continue
        if len(ref_values) != len(other_values):
            continue

        test_result = paired_test(other_values, ref_values)
        test_result["scorer"] = scorer_name
        test_result["reference"] = reference_scorer
        test_result["metric"] = metric
        comparisons.append(test_result)
        p_values.append(test_result["p_value"])

    if p_values:
        correction = correct_multiple_tests(p_values)
        for i, comp in enumerate(comparisons):
            comp["p_value_corrected"] = correction["corrected_p_values"][i]
            comp["rejected_bh"] = correction["rejected"][i]

    return comparisons
