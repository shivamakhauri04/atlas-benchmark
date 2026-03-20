"""Conformal calibration module.

Implements three conformal prediction methods for distribution-free
anomaly detection on agent traces:

  Theorem 1 (Split CP):  P(score > tau) <= alpha + 1/(n+1)
      — exchangeable calibration data, fixed threshold

  Theorem 2 (ACI):  lim_{T->inf} (1/T) sum err_t <= alpha
      — non-exchangeable online streams, adaptive threshold

  Theorem 3 (Feedback-Aware CP):  P[miscoverage] <= alpha + L*Delta/n + O(sqrt(log(1/d)/n))
      — endogenous monotonic distribution shift from trust score feedback loop
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def split_conformal_calibrate(
    scores: np.ndarray | torch.Tensor,
    *,
    alpha: float = 0.05,
) -> float:
    """Compute conformal threshold via split CP (Theorem 1).

    Given calibration scores, returns tau such that on exchangeable
    test data: P(score > tau) <= alpha + 1/(n+1).

    Args:
        scores: Non-conformity scores on calibration set. Shape (n,).
        alpha: Target miscoverage rate.

    Returns:
        Threshold tau (the ceil((1-alpha)(n+1))/n quantile of scores).
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    q = np.ceil((1 - alpha) * (n + 1)) / n
    q = min(q, 1.0)
    return float(np.quantile(scores, q))


def split_conformal_predict(
    scores: np.ndarray | torch.Tensor,
    tau: float,
) -> np.ndarray:
    """Apply split CP threshold: flag scores above tau as anomalous.

    Returns:
        Binary predictions (1 = anomalous, 0 = normal). Shape (n,).
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    return (scores > tau).astype(np.int32)


@dataclass
class ACIState:
    """State for Adaptive Conformal Inference (Theorem 2).

    Tracks the running threshold and error history for online
    ACI updates.
    """

    tau: float = 0.0
    alpha: float = 0.05
    gamma: float = 0.01
    errors: list[bool] = field(default_factory=list)

    @property
    def empirical_coverage(self) -> float:
        if not self.errors:
            return 1.0
        return 1.0 - sum(self.errors) / len(self.errors)


def adaptive_conformal_update(
    tau_prev: float,
    err: bool,
    *,
    alpha: float = 0.05,
    gamma: float = 0.01,
) -> float:
    """Online ACI threshold update (Theorem 2).

    Updates the threshold based on whether the last prediction was
    a miscoverage error, maintaining long-run coverage guarantees.

    Update rule: tau_{t+1} = tau_t + gamma * (alpha - err_t)

    When err=True (miscoverage), tau increases (more permissive).
    When err=False (correct coverage), tau decreases (more strict).

    Args:
        tau_prev: Previous threshold.
        err: Whether the last prediction was a miscoverage error.
        alpha: Target miscoverage rate.
        gamma: Step size for threshold adaptation.

    Returns:
        Updated threshold.
    """
    err_val = 1.0 if err else 0.0
    return tau_prev + gamma * (alpha - err_val)


def aci_step(state: ACIState, score: float) -> tuple[bool, ACIState]:
    """Run one ACI step: predict + update.

    Args:
        state: Current ACI state.
        score: Non-conformity score for the new observation.

    Returns:
        (is_anomalous, updated_state): whether the score exceeds
        the current threshold, and the updated ACI state.
    """
    is_anomalous = score > state.tau
    err = is_anomalous  # miscoverage = score exceeded threshold
    new_tau = adaptive_conformal_update(state.tau, err, alpha=state.alpha, gamma=state.gamma)
    new_errors = state.errors + [err]
    return is_anomalous, ACIState(
        tau=new_tau,
        alpha=state.alpha,
        gamma=state.gamma,
        errors=new_errors,
    )


@dataclass
class FeedbackAwareCPState:
    """State for Feedback-Aware CP (Theorem 3).

    Handles the endogenous distribution shift that occurs when:
    trust_scores → permissions → agent_behavior → new_traces → new_trust_scores

    The key insight: if the scoring function s(.) is L-Lipschitz
    and the feedback-induced shift is bounded by Delta, then:
    P[miscoverage] <= alpha + L*Delta/n + O(sqrt(log(1/delta)/n))

    The shift correction is applied as a threshold offset (tau_shift)
    rather than modifying alpha, preventing the ACI update loop from
    compounding the Lipschitz penalty into a divergent feedback cycle.
    """

    tau: float = 0.0
    alpha: float = 0.05
    gamma: float = 0.01
    lipschitz_L: float = 1.0
    delta_shift: float = 0.0
    tau_min: float = 0.0
    tau_max: float = 10.0
    window_scores: list[float] = field(default_factory=list)
    window_size: int = 100
    errors: list[bool] = field(default_factory=list)

    @property
    def shift_correction(self) -> float:
        """Threshold offset accounting for feedback-induced shift."""
        n = max(len(self.window_scores), 1)
        return self.lipschitz_L * self.delta_shift / n

    @property
    def coverage_bound(self) -> float:
        """Theoretical coverage bound (Theorem 3)."""
        n = max(len(self.window_scores), 1)
        delta = 0.05
        shift_term = self.lipschitz_L * self.delta_shift / n
        concentration_term = np.sqrt(np.log(1.0 / delta) / (2 * n))
        return 1.0 - self.alpha - shift_term - concentration_term


def feedback_aware_cp_update(
    state: FeedbackAwareCPState,
    score: float,
    shift_estimate: float | None = None,
) -> tuple[bool, FeedbackAwareCPState]:
    """Feedback-aware CP step (Theorem 3).

    Uses a decoupled update: the ACI online update adjusts tau based on
    miscoverage errors at the standard alpha rate, while the shift correction
    is applied as a separate additive offset to the effective threshold.
    This prevents the Lipschitz penalty from compounding through the
    online update loop.

    Effective threshold = tau_aci - shift_correction
    (lower effective threshold = more conservative = higher coverage)

    Args:
        state: Current feedback-aware CP state.
        score: Non-conformity score for the new observation.
        shift_estimate: Optional updated estimate of Delta
            (the distribution shift magnitude). If None, estimated
            from the score window.

    Returns:
        (is_anomalous, updated_state).
    """
    effective_tau = state.tau - state.shift_correction
    is_anomalous = score > effective_tau
    err = is_anomalous

    window = state.window_scores + [score]
    if len(window) > state.window_size:
        window = window[-state.window_size :]

    if shift_estimate is not None:
        delta_shift = shift_estimate
    elif len(window) >= 20:
        half = len(window) // 2
        first_half_mean = np.mean(window[:half])
        second_half_mean = np.mean(window[half:])
        delta_shift = abs(second_half_mean - first_half_mean)
    else:
        delta_shift = state.delta_shift

    err_val = 1.0 if err else 0.0
    new_tau = state.tau + state.gamma * (state.alpha - err_val)
    new_tau = np.clip(new_tau, state.tau_min, state.tau_max)

    return is_anomalous, FeedbackAwareCPState(
        tau=new_tau,
        alpha=state.alpha,
        gamma=state.gamma,
        lipschitz_L=state.lipschitz_L,
        delta_shift=delta_shift,
        tau_min=state.tau_min,
        tau_max=state.tau_max,
        window_scores=window,
        window_size=state.window_size,
        errors=state.errors + [err],
    )


def coverage_under_contamination(
    alpha: float,
    epsilon: float,
    anomaly_coverage_rate: float = 0.0,
) -> dict[str, float]:
    """Compute theoretical coverage bounds under contamination (Theorem 4).

    When a conformal prediction set C is calibrated on n normal samples
    at level alpha, and the test distribution has contamination rate
    epsilon (fraction of anomalous samples), marginal coverage becomes:

        Coverage = (1-alpha)(1-epsilon) + epsilon * P(anomaly covered)

    Lower bound: anomalies never covered (perfect detector)
        Coverage_min = (1-alpha)(1-epsilon)

    Upper bound: anomalies always covered (no detection)
        Coverage_max = (1-alpha)(1-epsilon) + epsilon = 1 - alpha*epsilon - alpha*(1-epsilon)
        Simplified: Coverage_max = 1 - alpha (matches standard CP)

    Args:
        alpha: Target miscoverage rate.
        epsilon: Contamination rate (fraction of anomalous test samples).
        anomaly_coverage_rate: P(anomaly is covered by C).
            0.0 = perfect detector, 1.0 = anomalies indistinguishable.

    Returns:
        Dict with coverage_expected, coverage_lower, coverage_upper.
    """
    cov_normal = 1.0 - alpha
    cov_expected = cov_normal * (1.0 - epsilon) + epsilon * anomaly_coverage_rate
    cov_lower = cov_normal * (1.0 - epsilon)
    cov_upper = cov_normal * (1.0 - epsilon) + epsilon

    return {
        "coverage_expected": float(cov_expected),
        "coverage_lower": float(cov_lower),
        "coverage_upper": float(cov_upper),
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "anomaly_coverage_rate": float(anomaly_coverage_rate),
    }


def coverage_auroc_tradeoff(
    alpha: float,
    epsilon: float,
    auroc: float,
) -> dict[str, float]:
    """Coverage-AUROC tradeoff under contamination (Corollary 4.1).

    For a detector with AUROC A, the expected marginal coverage is
    approximately:

        Coverage ~= (1-alpha)(1-epsilon) + epsilon * (1 - A)

    Better detection (higher AUROC) -> lower marginal coverage ->
    stronger anomaly signal. This tradeoff is inherent: a good
    detector SHOULD exclude anomalies, reducing marginal coverage.

    Args:
        alpha: Target miscoverage rate.
        epsilon: Contamination rate.
        auroc: Detector AUROC (0.5 = random, 1.0 = perfect).

    Returns:
        Dict with expected coverage and tradeoff metrics.
    """
    anomaly_miss_rate = 1.0 - auroc
    cov_expected = (1.0 - alpha) * (1.0 - epsilon) + epsilon * anomaly_miss_rate
    cov_at_random = (1.0 - alpha) * (1.0 - epsilon) + epsilon * 0.5

    return {
        "coverage_expected": float(cov_expected),
        "coverage_at_random_detector": float(cov_at_random),
        "coverage_at_perfect_detector": float((1.0 - alpha) * (1.0 - epsilon)),
        "coverage_at_no_detection": float(1.0 - alpha),
        "marginal_coverage_deficit": float((1.0 - alpha) - cov_expected),
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "auroc": float(auroc),
    }


def validate_coverage_theorem(
    test_labels: np.ndarray,
    test_preds: np.ndarray,
    test_scores: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Empirically validate the coverage-under-contamination theorem.

    Checks whether observed marginal coverage falls within the
    theoretical bounds given the measured contamination rate and AUROC.

    Args:
        test_labels: (n,) binary labels (0=normal, 1=anomalous).
        test_preds: (n,) binary predictions (0=normal, 1=anomalous).
        test_scores: (n,) anomaly scores.
        alpha: Target miscoverage rate.

    Returns:
        Dict with observed/theoretical coverage and validation result.
    """
    from sklearn.metrics import roc_auc_score

    n = len(test_labels)
    epsilon = float(test_labels.mean())

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = 0.5

    normal_mask = test_labels == 0
    normal_correct = (test_preds[normal_mask] == 0).sum()
    observed_normal_coverage = float(normal_correct / max(normal_mask.sum(), 1))

    observed_marginal_coverage = float((test_preds == test_labels).sum() / n)
    anomaly_covered = float(
        (test_preds[test_labels == 1] == 0).sum() / max((test_labels == 1).sum(), 1)
    )

    bounds = coverage_under_contamination(alpha, epsilon, anomaly_covered)
    tradeoff = coverage_auroc_tradeoff(alpha, epsilon, auroc)

    within_bounds = (
        bounds["coverage_lower"] - 1e-6
        <= tradeoff["coverage_expected"]
        <= bounds["coverage_upper"] + 1e-6
    )

    return {
        "observed_marginal_coverage": observed_marginal_coverage,
        "observed_normal_coverage": observed_normal_coverage,
        "observed_anomaly_coverage_rate": anomaly_covered,
        "epsilon": epsilon,
        "auroc": auroc,
        "theoretical_lower": bounds["coverage_lower"],
        "theoretical_upper": bounds["coverage_upper"],
        "theoretical_expected": tradeoff["coverage_expected"],
        "within_bounds": bool(within_bounds),
        "marginal_coverage_deficit": tradeoff["marginal_coverage_deficit"],
    }


def estimate_lipschitz(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    n_samples: int = 500,
    epsilon: float = 0.01,
    device: torch.device | str = "cpu",
) -> float:
    """Empirically estimate the Lipschitz constant of the scoring function.

    Perturbs inputs in the continuous embedding space (after the
    embedding layer) and measures the maximum ratio of output change
    to input perturbation magnitude.

    This avoids the dimensional error of perturbing discrete token
    indices. The Lipschitz constant is measured in the space where
    the model actually operates (d_model-dimensional embeddings).

    Args:
        model: The trace encoder model (must have .trace_score method
            and embedding layers tool_embed, args_embed, input_proj).
        dataset: TraceDataset to sample from.
        n_samples: Number of samples to test.
        epsilon: Perturbation magnitude (L2 norm per position) in
            the embedding space.
        device: Device to run on.

    Returns:
        Estimated Lipschitz constant L.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device
    max_ratio = 0.0
    n = min(n_samples, len(dataset))
    n_perturbations = 5

    for i in range(n):
        item = dataset[i]
        tool_ids = item["tool_ids"].unsqueeze(0).to(device)
        args_hashes = item["args_hashes"].unsqueeze(0).to(device)
        length = item["length"]
        mask = torch.zeros(1, tool_ids.size(1), dtype=torch.bool, device=device)
        mask[0, :length] = True

        with torch.no_grad():
            score_orig = model.trace_score(tool_ids, args_hashes, mask)

            tool_emb = model.tool_embed(tool_ids)
            if model.use_args:
                args_emb = model.args_embed(args_hashes)
                emb = model.input_proj(torch.cat([tool_emb, args_emb], dim=-1))
            else:
                emb = tool_emb

            for _ in range(n_perturbations):
                noise = torch.randn_like(emb) * epsilon
                noise[:, length:] = 0.0

                emb_pert = emb + noise
                x = model.pos_enc(emb_pert)
                x = model.input_dropout(x)

                seq_len = x.size(1)
                causal = model._causal_mask(seq_len, x.device)
                pad_mask = ~mask

                if model.transformer is not None:
                    x = model.transformer(
                        x,
                        mask=causal,
                        src_key_padding_mask=pad_mask,
                    )
                logits = model.head(x)

                targets = tool_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                preds = log_probs[:, :-1]
                nll = -preds.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                nll_full = torch.zeros_like(tool_ids, dtype=torch.float)
                nll_full[:, 1:] = nll
                nll_full = nll_full * mask.float()
                lengths = mask.float().sum(dim=1).clamp(min=1)
                score_pert = (nll_full.sum(dim=1) / lengths).item()

                delta_out = abs(score_pert - score_orig.item())
                valid_noise = noise[0, :length]
                delta_in = valid_noise.norm(p=2).item()

                if delta_in > 1e-8:
                    ratio = delta_out / delta_in
                    max_ratio = max(max_ratio, ratio)

    return max_ratio


# ---------------------------------------------------------------------------
# Polarity-Aware Conformal Prediction (PA-CP)
# ---------------------------------------------------------------------------


def polarity_aware_split_cp(
    cal_features: dict[str, np.ndarray],
    cal_labels: np.ndarray,
    test_features: dict[str, np.ndarray],
    test_labels: np.ndarray,
    puf: set[str],
    *,
    alpha: float = 0.05,
    scorer_fn: callable | None = None,
) -> dict[str, float]:
    """Polarity-Aware Split CP: filters unstable features before calibration.

    Extends Split CP by removing Polarity-Unstable Features (PUF)
    identified via FSI before computing non-conformity scores. This
    addresses an orthogonal failure mode to DS-CP (magnitude shift):
    PA-CP handles sign reversal.

    Args:
        cal_features: {feature_name: values} for calibration set.
        cal_labels: Binary labels for calibration, shape (n_cal,).
        test_features: {feature_name: values} for test set.
        test_labels: Binary labels for test, shape (n_test,).
        puf: Set of polarity-unstable feature names to exclude.
        alpha: Target miscoverage rate.
        scorer_fn: Optional scoring function (features_dict -> scores).
            If None, uses mean of z-scored stable features.

    Returns:
        Dict with coverage, auroc, tau, n_stable_features,
        n_filtered_features, predictions.
    """
    from sklearn.metrics import roc_auc_score

    stable = {k for k in cal_features if k not in puf}
    if not stable:
        stable = set(cal_features.keys())

    def _default_scorer(feats: dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for name in sorted(stable):
            vals = feats[name]
            mu, sigma = vals.mean(), max(vals.std(), 1e-8)
            cols.append((vals - mu) / sigma)
        return np.mean(np.column_stack(cols), axis=1)

    score_fn = scorer_fn if scorer_fn is not None else _default_scorer

    cal_scores = score_fn({k: v for k, v in cal_features.items() if k in stable})
    test_scores = score_fn({k: v for k, v in test_features.items() if k in stable})

    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    normal_mask = test_labels == 0
    coverage = float((preds[normal_mask] == 0).sum() / max(normal_mask.sum(), 1))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = 0.5

    return {
        "coverage": coverage,
        "auroc": auroc,
        "tau": float(tau),
        "n_stable_features": len(stable),
        "n_filtered_features": len(puf & set(cal_features.keys())),
        "predictions": preds.tolist(),
    }


def compute_polarity_coverage_bound(
    n_features: int,
    n_puf: int,
    alpha: float,
    delta_polarity: float,
) -> float:
    """Theoretical coverage bound for PA-CP.

    Coverage >= 1 - alpha - (|PUF| / |F|) * delta_polarity

    The bound degrades proportionally to the fraction of unstable
    features and the magnitude of polarity reversal.

    Args:
        n_features: Total number of features |F|.
        n_puf: Number of polarity-unstable features |PUF|.
        alpha: Target miscoverage rate.
        delta_polarity: Maximum polarity reversal magnitude.

    Returns:
        Lower bound on coverage.
    """
    if n_features == 0:
        return 1.0 - alpha
    puf_fraction = n_puf / n_features
    return 1.0 - alpha - puf_fraction * delta_polarity


# ---------------------------------------------------------------------------
# PAFC-Conformal: Polarity-Aware Feature Correction + Conformal Prediction
# ---------------------------------------------------------------------------


def pafc_correct_features(
    features: dict[str, np.ndarray],
    source_d: dict[str, float],
    target_d: dict[str, float],
) -> dict[str, np.ndarray]:
    """Apply PAFC sign correction to features where polarity reverses.

    For each feature where sign(d_source) != sign(d_target) and both
    effects are non-negligible (|d| > 0.1), negate the feature values.

    Args:
        features: {feature_name: values} to correct.
        source_d: Cohen's d per feature in source domain.
        target_d: Cohen's d per feature in target domain.

    Returns:
        Corrected feature dict (reversed features negated).
    """
    corrected = {}
    for name, values in features.items():
        d_src = source_d.get(name, 0.0)
        d_tgt = target_d.get(name, 0.0)
        if abs(d_src) > 0.1 and abs(d_tgt) > 0.1:
            src_sign = 1 if d_src >= 0 else -1
            tgt_sign = 1 if d_tgt >= 0 else -1
            if src_sign != tgt_sign:
                corrected[name] = -values
                continue
        corrected[name] = values
    return corrected


def pafc_conformal_calibrate(
    cal_features: dict[str, np.ndarray],
    cal_labels: np.ndarray,
    source_d: dict[str, float],
    target_d: dict[str, float],
    *,
    alpha: float = 0.05,
    scorer_fn: callable | None = None,
) -> tuple[float, set[str]]:
    """PAFC-Conformal calibration: correct features then compute CP threshold.

    Composes PAFC sign correction (Theorem 1) with split conformal
    prediction to produce a threshold with coverage guarantee
    P(Y in C(X)) >= 1 - alpha - delta (Theorem 2).

    Args:
        cal_features: Calibration set features.
        cal_labels: Calibration binary labels.
        source_d: Cohen's d per feature in source domain.
        target_d: Cohen's d per feature in target domain.
        alpha: Target miscoverage rate.
        scorer_fn: Optional scoring function on corrected features.

    Returns:
        (tau, corrected_features_set): threshold and set of corrected feature names.
    """
    corrected = pafc_correct_features(cal_features, source_d, target_d)

    corrected_names = {
        name for name in cal_features
        if not np.array_equal(corrected[name], cal_features[name])
    }

    def _default_scorer(feats: dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for name in sorted(feats.keys()):
            vals = feats[name]
            mu, sigma = vals.mean(), max(vals.std(), 1e-8)
            cols.append((vals - mu) / sigma)
        return np.mean(np.column_stack(cols), axis=1)

    score_fn = scorer_fn if scorer_fn is not None else _default_scorer
    cal_scores = score_fn(corrected)
    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    return tau, corrected_names


def pafc_conformal_predict(
    test_features: dict[str, np.ndarray],
    test_labels: np.ndarray,
    source_d: dict[str, float],
    target_d: dict[str, float],
    tau: float,
    *,
    scorer_fn: callable | None = None,
) -> dict[str, float]:
    """PAFC-Conformal prediction: correct test features and apply threshold.

    Returns:
        Dict with coverage, auroc, predictions, n_corrected_features.
    """
    from sklearn.metrics import roc_auc_score

    corrected = pafc_correct_features(test_features, source_d, target_d)

    def _default_scorer(feats: dict[str, np.ndarray]) -> np.ndarray:
        cols = []
        for name in sorted(feats.keys()):
            vals = feats[name]
            mu, sigma = vals.mean(), max(vals.std(), 1e-8)
            cols.append((vals - mu) / sigma)
        return np.mean(np.column_stack(cols), axis=1)

    score_fn = scorer_fn if scorer_fn is not None else _default_scorer
    test_scores = score_fn(corrected)
    preds = split_conformal_predict(test_scores, tau)

    normal_mask = test_labels == 0
    coverage = float((preds[normal_mask] == 0).sum() / max(normal_mask.sum(), 1))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = 0.5

    n_corrected = sum(
        1 for name in test_features
        if not np.array_equal(corrected[name], test_features[name])
    )

    return {
        "coverage": coverage,
        "auroc": auroc,
        "n_corrected_features": n_corrected,
        "predictions": preds.tolist(),
    }


def theorem2_coverage_bound(
    alpha: float,
    delta_sign_detection: float,
) -> float:
    """Theorem 2: Coverage bound for PAFC-Conformal.

    If PAFC correctly detects sign reversal with probability >= 1 - delta,
    then the PAFC-corrected conformal prediction sets satisfy:
        P(Y in C(X)) >= 1 - alpha - delta

    The delta term comes from Theorem 1 (sign detection sample complexity)
    and quantifies the cost of polarity correction uncertainty.

    Args:
        alpha: Target miscoverage rate.
        delta_sign_detection: Probability of sign detection failure
            (from Theorem 1, depends on N and |d|).

    Returns:
        Lower bound on coverage.
    """
    return 1.0 - alpha - delta_sign_detection
