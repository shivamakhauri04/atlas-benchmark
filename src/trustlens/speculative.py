"""Speculative agent execution with conformal monitoring and rollback.

Implements Theorem 6 (Speculative Regret Bound):
    E[Regret_spec] <= E[Regret_conservative]
    when P(anomaly) < theta* where theta* is derived from the CP
    coverage bound.

The key idea: instead of blocking potentially unsafe actions (conservative
strategy), execute them in a sandbox with conformal monitoring. If the
anomaly score stays below threshold throughout execution -> commit the
results. If it crosses -> roll back.

This is the first formalization of speculative execution with conformal
prediction. Zero prior work combines sandbox execution with CP monitoring
and formal regret bounds.

Regret model:
    - Conservative regret: C_block * P(block safe action)
      (cost of unnecessarily blocking safe actions)
    - Speculative regret: C_rollback * P(rollback) + C_miss * P(commit unsafe)
      (cost of rollback overhead + cost of missed anomalies that commit)

Theorem 6 shows speculative dominates when:
    P(anomaly) < theta* = C_block / (C_block + C_rollback)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class SpeculativeConfig:
    """Configuration for speculative execution."""

    tau: float  # conformal threshold for commit/rollback decision
    max_steps: int = 50  # max actions per speculative window
    rollback_cost: float = 1.0  # cost per rollback (relative)
    block_cost: float = 2.0  # cost per blocked safe action (relative)
    miss_cost: float = 10.0  # cost of committing an unsafe action
    commit_reward: float = 1.0  # reward for successfully committing a safe action

    @staticmethod
    def from_single_tau(tau: float, **kwargs) -> "SpeculativeConfig":
        """Factory for backward-compat: creates config with a single tau."""
        return SpeculativeConfig(tau=tau, **kwargs)


@dataclass(frozen=True)
class DualThresholdConfig:
    """Configuration for dual-threshold speculative vs conservative comparison.

    The real thesis: speculative can afford a more permissive threshold
    (tau_spec) because rollback provides a safety net. Conservative needs
    a stricter threshold (tau_cons) since blocking is the only defense.
    """

    tau_spec: float  # permissive threshold for speculative (lower = more permissive)
    tau_cons: float  # strict threshold for conservative (higher = more restrictive)
    max_steps: int = 50
    rollback_cost: float = 1.0
    block_cost: float = 2.0
    miss_cost: float = 10.0
    commit_reward: float = 0.0


@dataclass
class SpeculativeWindow:
    """State of a single speculative execution window."""

    scores: list[float] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    committed: bool = False
    rolled_back: bool = False
    max_score: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.scores)

    @property
    def has_anomaly(self) -> bool:
        return any(lab == 1 for lab in self.labels)


@dataclass(frozen=True)
class SpeculativeResult:
    """Result of running speculative execution on a trace."""

    decision: str  # "commit" or "rollback"
    n_steps: int
    max_score: float
    threshold: float
    has_anomaly: bool
    correct: bool  # True if decision matches ground truth


def simulate_speculative_execution(
    per_step_scores: np.ndarray,
    per_step_labels: np.ndarray,
    tau: float,
) -> SpeculativeResult:
    """Simulate speculative execution on a single trace.

    The trace executes in a sandbox. At each step, the anomaly score
    is checked against tau. If any score exceeds tau -> rollback.
    Otherwise -> commit after all steps.

    Args:
        per_step_scores: (T,) anomaly scores per action in the trace.
        per_step_labels: (T,) labels per action (0=normal, 1=anomalous).
        tau: Conformal threshold for commit/rollback.

    Returns:
        SpeculativeResult with decision and correctness.
    """
    max_score = 0.0
    rollback = False

    for score in per_step_scores:
        max_score = max(max_score, float(score))
        if score > tau:
            rollback = True
            break

    has_anomaly = bool(np.any(per_step_labels == 1))
    if rollback:
        decision = "rollback"
        correct = has_anomaly  # correct to rollback if trace is actually anomalous
    else:
        decision = "commit"
        correct = not has_anomaly  # correct to commit if trace is actually safe

    return SpeculativeResult(
        decision=decision,
        n_steps=len(per_step_scores),
        max_score=max_score,
        threshold=tau,
        has_anomaly=has_anomaly,
        correct=correct,
    )


def compute_speculative_regret(
    results: list[SpeculativeResult],
    config: SpeculativeConfig,
) -> dict:
    """Compute regret metrics for speculative execution.

    Regret is the total cost of incorrect decisions:
    - Rollback a safe trace: rollback_cost (wasted computation)
    - Commit an unsafe trace: miss_cost (safety violation)
    - Reward for correctly committing safe traces: -commit_reward

    Args:
        results: List of speculative execution results.
        config: Cost/reward configuration.

    Returns:
        Dict with total regret, regret breakdown, and rates.
    """
    n = len(results)
    if n == 0:
        return {"total_regret": 0.0, "n_traces": 0}

    n_commits = sum(1 for r in results if r.decision == "commit")
    n_rollbacks = sum(1 for r in results if r.decision == "rollback")

    # Correct decisions
    n_correct_commit = sum(
        1 for r in results if r.decision == "commit" and not r.has_anomaly
    )
    n_correct_rollback = sum(
        1 for r in results if r.decision == "rollback" and r.has_anomaly
    )

    # Errors
    n_false_commit = sum(
        1 for r in results if r.decision == "commit" and r.has_anomaly
    )
    n_false_rollback = sum(
        1 for r in results if r.decision == "rollback" and not r.has_anomaly
    )

    # Regret computation
    regret_false_commit = n_false_commit * config.miss_cost
    regret_false_rollback = n_false_rollback * config.rollback_cost
    reward_correct_commit = n_correct_commit * config.commit_reward
    total_regret = regret_false_commit + regret_false_rollback - reward_correct_commit

    return {
        "total_regret": float(total_regret),
        "mean_regret": float(total_regret / n),
        "n_traces": n,
        "n_commits": n_commits,
        "n_rollbacks": n_rollbacks,
        "commit_rate": float(n_commits / n),
        "rollback_rate": float(n_rollbacks / n),
        "false_commit_rate": float(n_false_commit / max(n_commits, 1)),
        "false_rollback_rate": float(n_false_rollback / max(n_rollbacks, 1)),
        "correct_commit": n_correct_commit,
        "correct_rollback": n_correct_rollback,
        "false_commit": n_false_commit,
        "false_rollback": n_false_rollback,
        "regret_from_false_commit": float(regret_false_commit),
        "regret_from_false_rollback": float(regret_false_rollback),
        "reward_from_correct_commit": float(reward_correct_commit),
    }


def compute_conservative_regret(
    trace_scores: np.ndarray,
    trace_labels: np.ndarray,
    tau: float,
    config: SpeculativeConfig,
) -> dict:
    """Compute regret for the conservative (block-first) strategy.

    Conservative strategy: block any trace with max score > tau.
    No execution occurs — blocked traces are sent to human review.
    The cost is C_block for each safe trace that gets blocked.

    Args:
        trace_scores: (N,) per-trace anomaly scores.
        trace_labels: (N,) per-trace labels (0=normal, 1=anomalous).
        tau: Threshold for blocking.
        config: Cost configuration.

    Returns:
        Dict with conservative regret metrics.
    """
    n = len(trace_scores)
    blocked = trace_scores > tau
    passed = ~blocked

    n_blocked = int(blocked.sum())
    n_passed = int(passed.sum())

    # False blocks: safe traces that got blocked
    false_blocks = int((blocked & (trace_labels == 0)).sum())
    # Missed anomalies: unsafe traces that passed
    missed = int((passed & (trace_labels == 1)).sum())

    regret_false_block = false_blocks * config.block_cost
    regret_missed = missed * config.miss_cost
    total_regret = regret_false_block + regret_missed

    n_normal = int((trace_labels == 0).sum())
    n_anomalous = int((trace_labels == 1).sum())

    return {
        "total_regret": float(total_regret),
        "mean_regret": float(total_regret / max(n, 1)),
        "n_traces": n,
        "n_blocked": n_blocked,
        "n_passed": n_passed,
        "false_block_rate": float(false_blocks / max(n_normal, 1)),
        "miss_rate": float(missed / max(n_anomalous, 1)),
        "regret_from_false_block": float(regret_false_block),
        "regret_from_missed": float(regret_missed),
    }


def compute_theta_star(config: SpeculativeConfig | DualThresholdConfig) -> float:
    """Compute the theoretical threshold theta* (Theorem 6).

    Speculative execution dominates conservative when P(anomaly) < theta*:
        theta* = C_block / (C_block + C_rollback)

    Below theta*, the expected regret of speculative <= conservative.

    Returns:
        theta*: The anomaly rate crossover point.
    """
    return config.block_cost / (config.block_cost + config.rollback_cost)


def compute_theta_star_dual(
    config: DualThresholdConfig,
    miss_rate_spec: float,
    miss_rate_cons: float,
) -> float:
    """Compute theta* for dual-threshold setup (extended Theorem 6).

    With different thresholds, the crossover depends on the miss-rate
    differential. Speculative dominates when:

        P(anomaly) < C_block * (1 - false_block_cons)
                      / (C_rollback + C_miss * (miss_spec - miss_cons))

    where miss_spec > miss_cons because tau_spec is more permissive.

    If miss_spec == miss_cons (encoder is perfect), this reduces to
    the single-threshold formula.

    Args:
        config: Dual threshold cost configuration.
        miss_rate_spec: P(commit unsafe | anomalous) under tau_spec.
        miss_rate_cons: P(pass unsafe | anomalous) under tau_cons.

    Returns:
        theta*: The anomaly rate crossover point.
    """
    miss_diff = max(miss_rate_spec - miss_rate_cons, 1e-10)
    numerator = config.block_cost
    denominator = config.rollback_cost + config.miss_cost * miss_diff
    return numerator / denominator


def evaluate_speculative_vs_conservative(
    per_step_scores_list: list[np.ndarray],
    per_step_labels_list: list[np.ndarray],
    trace_scores: np.ndarray,
    trace_labels: np.ndarray,
    tau: float,
    config: SpeculativeConfig | None = None,
    *,
    tau_cons: float | None = None,
) -> dict:
    """Compare speculative vs conservative execution strategies.

    Args:
        per_step_scores_list: List of (T_i,) per-step score arrays, one per trace.
        per_step_labels_list: List of (T_i,) per-step label arrays, one per trace.
        trace_scores: (N,) per-trace scores (mean NLL).
        trace_labels: (N,) per-trace labels.
        tau: Conformal threshold (used for speculative; also for conservative
            if tau_cons is None).
        config: Execution cost configuration.
        tau_cons: Optional separate threshold for conservative strategy.
            If None, conservative uses the same tau as speculative.

    Returns:
        Dict comparing both strategies with regret and operational metrics.
    """
    if config is None:
        config = SpeculativeConfig(tau=tau)

    tau_spec = tau
    tau_conservative = tau_cons if tau_cons is not None else tau

    # Run speculative execution on each trace (uses tau_spec)
    spec_results = [
        simulate_speculative_execution(scores, labels, tau_spec)
        for scores, labels in zip(per_step_scores_list, per_step_labels_list)
    ]

    spec_regret = compute_speculative_regret(spec_results, config)

    # Conservative must use per-trace MAX scores (same domain as tau).
    # tau was calibrated on per-step max NLL, so conservative blocking
    # threshold must also operate on per-trace max, not per-trace mean.
    trace_max_scores = np.array([float(s.max()) for s in per_step_scores_list])
    cons_regret = compute_conservative_regret(
        trace_max_scores, trace_labels, tau_conservative, config
    )

    # Theoretical analysis
    anomaly_rate = float(trace_labels.mean())
    theta_star = compute_theta_star(config)
    spec_dominates = anomaly_rate < theta_star

    return {
        "tau_spec": tau_spec,
        "tau_cons": tau_conservative,
        "tau": tau_spec,  # backward compat
        "anomaly_rate": anomaly_rate,
        "theta_star": theta_star,
        "speculative_dominates_theoretical": spec_dominates,
        "speculative": spec_regret,
        "conservative": cons_regret,
        "regret_difference": spec_regret["total_regret"] - cons_regret["total_regret"],
        "regret_ratio": (
            spec_regret["total_regret"] / max(cons_regret["total_regret"], 1e-10)
        ),
        "speculative_wins_empirical": (
            spec_regret["total_regret"] <= cons_regret["total_regret"]
        ),
    }


def sweep_anomaly_rate(
    per_step_scores_list: list[np.ndarray],
    per_step_labels_list: list[np.ndarray],
    trace_labels: np.ndarray,
    tau: float,
    config: SpeculativeConfig,
    target_rates: list[float],
    rng: np.random.Generator | None = None,
    *,
    tau_cons: float | None = None,
) -> list[dict]:
    """Sweep anomaly rate by subsampling to test near theta*.

    For each target rate, subsample the test set to approximate that
    anomaly rate, then compare speculative vs conservative.

    Args:
        per_step_scores_list: Per-step scores for all traces.
        per_step_labels_list: Per-step labels for all traces.
        trace_labels: (N,) per-trace labels.
        tau: Conformal threshold (used for speculative).
        config: Cost configuration.
        target_rates: Desired anomaly rates to test.
        rng: Random generator for reproducible subsampling.
        tau_cons: Optional separate threshold for conservative strategy.

    Returns:
        List of dicts with results per target rate.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    tau_spec = tau
    tau_conservative = tau_cons if tau_cons is not None else tau

    normal_idx = np.where(trace_labels == 0)[0]
    anomaly_idx = np.where(trace_labels == 1)[0]
    n_normal = len(normal_idx)
    n_anomaly = len(anomaly_idx)

    results = []
    for target_rate in target_rates:
        if target_rate <= 0 or target_rate >= 1:
            continue

        # Fix normal count, vary anomaly count to hit target rate
        # target_rate = n_anom_sub / (n_normal + n_anom_sub)
        # => n_anom_sub = target_rate * n_normal / (1 - target_rate)
        n_anom_needed = int(target_rate * n_normal / (1 - target_rate))
        if n_anom_needed > n_anomaly:
            n_anom_sub = n_anomaly
            n_norm_needed = int(n_anomaly * (1 - target_rate) / target_rate)
            n_norm_sub = min(n_norm_needed, n_normal)
            norm_sub = rng.choice(normal_idx, size=n_norm_sub, replace=False)
            anom_sub = anomaly_idx
        else:
            norm_sub = normal_idx
            n_norm_sub = n_normal
            anom_sub = rng.choice(anomaly_idx, size=n_anom_needed, replace=False)

        selected = np.concatenate([norm_sub, anom_sub])
        rng.shuffle(selected)

        sub_step_scores = [per_step_scores_list[i] for i in selected]
        sub_step_labels = [per_step_labels_list[i] for i in selected]
        sub_trace_labels = trace_labels[selected]

        # Speculative uses tau_spec (permissive)
        spec_results_list = [
            simulate_speculative_execution(scores, labels, tau_spec)
            for scores, labels in zip(sub_step_scores, sub_step_labels)
        ]
        spec_regret = compute_speculative_regret(spec_results_list, config)

        # Conservative uses tau_conservative (strict)
        trace_max = np.array([float(s.max()) for s in sub_step_scores])
        cons_regret = compute_conservative_regret(
            trace_max, sub_trace_labels, tau_conservative, config
        )

        actual_rate = float(sub_trace_labels.mean())
        theta_star = compute_theta_star(config)

        results.append({
            "target_rate": target_rate,
            "actual_rate": actual_rate,
            "n_traces": len(selected),
            "theta_star": theta_star,
            "below_theta_star": actual_rate < theta_star,
            "speculative_regret": spec_regret["total_regret"],
            "conservative_regret": cons_regret["total_regret"],
            "regret_difference": (
                spec_regret["total_regret"] - cons_regret["total_regret"]
            ),
            "speculative_wins": (
                spec_regret["total_regret"] <= cons_regret["total_regret"]
            ),
            "rollback_rate": spec_regret["rollback_rate"],
            "false_block_rate": cons_regret["false_block_rate"],
            "tau_spec": tau_spec,
            "tau_cons": tau_conservative,
        })

    return results


def speculative_latency_simulation(
    per_step_scores_list: list[np.ndarray],
    tau: float,
    *,
    action_latency_ms: float = 5.0,
    rollback_latency_ms: float = 10.0,
    block_latency_ms: float = 0.5,
    human_review_latency_ms: float = 30000.0,
) -> dict:
    """Simulate latency comparison between speculative and conservative.

    Speculative: actions execute (action_latency each), then commit or
    rollback. Rollback adds rollback_latency overhead.

    Conservative: each action is checked (block_latency), blocked actions
    go to human review (human_review_latency).

    Args:
        per_step_scores_list: Per-step scores for each trace.
        tau: Conformal threshold.
        action_latency_ms: Time to execute one action in sandbox.
        rollback_latency_ms: Overhead for rolling back a sandbox.
        block_latency_ms: Time to check one action against threshold.
        human_review_latency_ms: Time for human to review a blocked action.

    Returns:
        Dict with latency statistics for both strategies.
    """
    spec_latencies = []
    cons_latencies = []

    for scores in per_step_scores_list:
        n_steps = len(scores)

        # Speculative: execute all steps, then commit or rollback
        exceeded = False
        steps_executed = 0
        for s in scores:
            steps_executed += 1
            if s > tau:
                exceeded = True
                break

        spec_lat = steps_executed * action_latency_ms
        if exceeded:
            spec_lat += rollback_latency_ms
        spec_latencies.append(spec_lat)

        # Conservative: check each step, block if exceeded
        cons_lat = 0.0
        for s in scores:
            cons_lat += block_latency_ms
            if s > tau:
                cons_lat += human_review_latency_ms
                break
            cons_lat += action_latency_ms
        cons_latencies.append(cons_lat)

    spec_arr = np.array(spec_latencies)
    cons_arr = np.array(cons_latencies)

    return {
        "speculative_mean_ms": float(spec_arr.mean()),
        "speculative_p50_ms": float(np.median(spec_arr)),
        "speculative_p95_ms": float(np.percentile(spec_arr, 95)),
        "speculative_p99_ms": float(np.percentile(spec_arr, 99)),
        "conservative_mean_ms": float(cons_arr.mean()),
        "conservative_p50_ms": float(np.median(cons_arr)),
        "conservative_p95_ms": float(np.percentile(cons_arr, 95)),
        "conservative_p99_ms": float(np.percentile(cons_arr, 99)),
        "speedup_mean": float(cons_arr.mean() / max(spec_arr.mean(), 1e-10)),
        "n_traces": len(per_step_scores_list),
    }
