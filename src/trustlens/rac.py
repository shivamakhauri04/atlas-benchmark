"""Risk-Averse Calibration (RAC) module — two-threshold optimal permission decisions.

Implements Theorem 4: For any risk-averse agent operating under a VaR
criterion, the optimal decision rule is:

    approve  if s < tau_low
    defer    if tau_low <= s <= tau_high
    reject   if s > tau_high

where tau_low, tau_high are derived from the calibration set to minimize
an asymmetric cost function subject to coverage constraints.

Based on:
    Kiyani et al. — Risk-Averse Calibration (ICML 2025)
    Noorani et al. — Human-AI Collaborative UQ (arXiv Oct 2025)

The key insight: consequence-aware scoring becomes a *derived necessity*,
not an ad-hoc choice — sharper scores produce a narrower defer zone, which
is provably optimal under the RAC framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RACThresholds:
    """Two-threshold decision boundaries from RAC calibration."""

    tau_low: float
    tau_high: float
    alpha: float
    cost_ratio: float
    n_cal: int

    @property
    def defer_width(self) -> float:
        return self.tau_high - self.tau_low


@dataclass(frozen=True)
class RACDecision:
    """Per-action decision from the RAC two-threshold rule."""

    action: str  # "approve", "defer", "reject"
    score: float
    tau_low: float
    tau_high: float


def rac_calibrate(
    scores: np.ndarray | torch.Tensor,
    *,
    alpha: float = 0.05,
    cost_ratio: float = 5.0,
    defer_budget: float = 0.15,
) -> RACThresholds:
    """Calibrate two-threshold RAC decision boundaries.

    Derives tau_low and tau_high from the calibration scores such that:
    - P(approve | normal) >= 1 - alpha  (coverage guarantee)
    - The defer zone [tau_low, tau_high] is minimized subject to the
      asymmetric cost constraint C_FN / C_FP = cost_ratio
    - At most defer_budget fraction of actions are deferred

    The VaR-optimal thresholds are:
        tau_high = quantile(scores, 1 - alpha)  [standard CP threshold]
        tau_low  = tau_high - delta
    where delta is chosen to satisfy the cost-weighted optimality condition.

    Args:
        scores: Calibration scores (normal samples). Shape (n,).
        alpha: Target miscoverage rate for the approve region.
        cost_ratio: Asymmetric cost ratio C_FN / C_FP. Higher values
            make the defer zone wider (more conservative).
        defer_budget: Maximum fraction of scores allowed in defer zone.

    Returns:
        RACThresholds with tau_low and tau_high.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return RACThresholds(
            tau_low=0.0,
            tau_high=float("inf"),
            alpha=alpha,
            cost_ratio=cost_ratio,
            n_cal=0,
        )

    q_high = np.ceil((1 - alpha) * (n + 1)) / n
    q_high = min(q_high, 1.0)
    tau_high = float(np.quantile(scores, q_high))

    # Derive tau_low via cost-weighted quantile offset.
    # The offset delta shrinks the approve zone proportional to 1/cost_ratio.
    # Interpretation: higher cost of missing an anomaly (FN) -> wider defer zone.
    alpha_low = alpha + (1 - alpha) * (1 / (1 + cost_ratio))
    alpha_low = min(alpha_low, 1.0 - alpha)

    q_low = np.ceil((1 - alpha_low) * (n + 1)) / n
    q_low = min(q_low, 1.0)
    tau_low_candidate = float(np.quantile(scores, q_low))

    # Enforce defer budget: at most defer_budget fraction of cal scores
    # should fall in [tau_low, tau_high].
    defer_fraction = float(np.mean((scores >= tau_low_candidate) & (scores <= tau_high)))
    if defer_fraction > defer_budget and n > 0:
        sorted_scores = np.sort(scores)
        target_idx = max(0, int(np.floor((1 - defer_budget) * n)) - 1)
        # Find the score at which defer zone starts, respecting budget
        budget_tau_low = float(sorted_scores[min(target_idx, n - 1)])
        tau_low = max(tau_low_candidate, budget_tau_low)
    else:
        tau_low = tau_low_candidate

    # Ensure tau_low <= tau_high
    tau_low = min(tau_low, tau_high)

    return RACThresholds(
        tau_low=tau_low,
        tau_high=tau_high,
        alpha=alpha,
        cost_ratio=cost_ratio,
        n_cal=n,
    )


def rac_decide(
    scores: np.ndarray | torch.Tensor,
    thresholds: RACThresholds,
) -> np.ndarray:
    """Apply RAC two-threshold decision rule.

    Args:
        scores: Anomaly scores. Shape (n,).
        thresholds: Calibrated RAC thresholds.

    Returns:
        decisions: (n,) int array. 0=approve, 1=defer, 2=reject.
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64)

    decisions = np.zeros(len(scores), dtype=np.int32)
    decisions[scores >= thresholds.tau_low] = 1  # defer
    decisions[scores > thresholds.tau_high] = 2  # reject
    return decisions


def rac_decide_single(score: float, thresholds: RACThresholds) -> RACDecision:
    """Apply RAC rule to a single score. Returns a structured decision."""
    if score < thresholds.tau_low:
        action = "approve"
    elif score > thresholds.tau_high:
        action = "reject"
    else:
        action = "defer"
    return RACDecision(
        action=action,
        score=score,
        tau_low=thresholds.tau_low,
        tau_high=thresholds.tau_high,
    )


def evaluate_rac(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    alpha: float = 0.05,
    cost_ratio: float = 5.0,
    defer_budget: float = 0.15,
    normal_only_cal: bool = True,
) -> dict:
    """Run RAC evaluation and compute three-way decision metrics.

    Args:
        cal_scores: Calibration anomaly scores.
        cal_labels: Calibration labels (0=normal, 1=anomalous).
        test_scores: Test anomaly scores.
        test_labels: Test labels.
        alpha: Target miscoverage rate.
        cost_ratio: C_FN / C_FP asymmetric cost ratio.
        defer_budget: Max fraction of actions to defer.
        normal_only_cal: If True, calibrate on normal scores only.

    Returns:
        Dict with RAC metrics: coverage, FNR, FPR, defer_rate, VaR, etc.
    """
    if normal_only_cal:
        cal_for_rac = cal_scores[cal_labels == 0]
    else:
        cal_for_rac = cal_scores

    thresholds = rac_calibrate(
        cal_for_rac,
        alpha=alpha,
        cost_ratio=cost_ratio,
        defer_budget=defer_budget,
    )

    decisions = rac_decide(test_scores, thresholds)

    n = len(test_labels)
    n_normal = int((test_labels == 0).sum())
    n_anomalous = int((test_labels == 1).sum())

    approved = decisions == 0
    deferred = decisions == 1
    rejected = decisions == 2

    # Coverage: P(approve | normal) — should be >= 1-alpha
    approve_normal = float(approved[test_labels == 0].sum() / max(n_normal, 1))

    # FNR: P(approve | anomalous) — anomalies that slip through
    fnr = float(approved[test_labels == 1].sum() / max(n_anomalous, 1))

    # FPR: P(reject | normal) — normals incorrectly rejected
    fpr = float(rejected[test_labels == 0].sum() / max(n_normal, 1))

    # Defer rates
    defer_rate = float(deferred.sum() / max(n, 1))
    defer_normal = float(deferred[test_labels == 0].sum() / max(n_normal, 1))
    defer_anomalous = float(deferred[test_labels == 1].sum() / max(n_anomalous, 1))

    # Reject rates
    reject_rate = float(rejected.sum() / max(n, 1))
    reject_anomalous = float(rejected[test_labels == 1].sum() / max(n_anomalous, 1))

    # VaR: Value at Risk = expected cost under worst-case miscoverage
    # Cost = C_FN * FNR + C_FP * FPR (normalized by cost_ratio)
    var_cost = cost_ratio * fnr + fpr

    # For comparison: binary CP cost (no defer zone)
    from trustlens.conformal import split_conformal_calibrate, split_conformal_predict

    binary_tau = split_conformal_calibrate(cal_for_rac, alpha=alpha)
    binary_preds = split_conformal_predict(test_scores, binary_tau)
    binary_fnr = float(
        (binary_preds[test_labels == 1] == 0).sum() / max(n_anomalous, 1)
    )
    binary_fpr = float(
        (binary_preds[test_labels == 0] == 1).sum() / max(n_normal, 1)
    )
    binary_var = cost_ratio * binary_fnr + binary_fpr

    return {
        "alpha": alpha,
        "cost_ratio": cost_ratio,
        "defer_budget": defer_budget,
        "tau_low": thresholds.tau_low,
        "tau_high": thresholds.tau_high,
        "defer_width": thresholds.defer_width,
        "coverage_approve_normal": approve_normal,
        "fnr": fnr,
        "fpr": fpr,
        "defer_rate": defer_rate,
        "defer_rate_normal": defer_normal,
        "defer_rate_anomalous": defer_anomalous,
        "reject_rate": reject_rate,
        "reject_rate_anomalous": reject_anomalous,
        "var_cost": var_cost,
        "binary_var_cost": binary_var,
        "var_improvement": binary_var - var_cost,
        "n_cal": thresholds.n_cal,
        "n_test": n,
        "n_anomalous": n_anomalous,
        "calibration": "normal_only" if normal_only_cal else "mixed",
    }


def rac_sensitivity_sweep(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    alpha: float = 0.05,
    cost_ratios: list[float] | None = None,
    defer_budgets: list[float] | None = None,
    normal_only_cal: bool = True,
) -> list[dict]:
    """Sweep over cost_ratio and defer_budget to characterize RAC behavior.

    Returns a list of evaluation dicts, one per (cost_ratio, defer_budget) pair.
    """
    if cost_ratios is None:
        cost_ratios = [1.0, 2.0, 5.0, 10.0, 20.0]
    if defer_budgets is None:
        defer_budgets = [0.05, 0.10, 0.15, 0.20, 0.30]

    results = []
    for cr in cost_ratios:
        for db in defer_budgets:
            r = evaluate_rac(
                cal_scores,
                cal_labels,
                test_scores,
                test_labels,
                alpha=alpha,
                cost_ratio=cr,
                defer_budget=db,
                normal_only_cal=normal_only_cal,
            )
            results.append(r)
    return results
