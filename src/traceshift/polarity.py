"""Feature Stability Index (FSI) for detecting polarity reversal.

Formalizes the observation that certain anomaly-detection features
reverse their correlation with labels across domains. Extends
Eastwood et al. (NeurIPS 2023) stable/unstable feature formalization
from classification to anomaly detection.

Key concepts:
    FSI(f, D_A, D_B) = min(|rho_A|, |rho_B|) * sign(rho_A * rho_B)
    Negative FSI → polarity reversal (a formal Simpson's paradox criterion)
    PUF = {f : FSI(f) < threshold} — Polarity-Unstable Feature set
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_domain_correlations(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    method: str = "spearman",
) -> dict[str, float]:
    """Compute feature-label correlation within a single domain.

    Args:
        features: Mapping of feature name to values array, shape (n,).
        labels: Binary labels array, shape (n,).
        method: Correlation method — "spearman" or "pearson".

    Returns:
        Mapping of feature name to correlation coefficient (rho).
    """
    correlations: dict[str, float] = {}
    for name, values in features.items():
        if len(values) < 3 or np.std(values) < 1e-10:
            correlations[name] = 0.0
            continue
        if method == "spearman":
            rho, _ = stats.spearmanr(values, labels)
        elif method == "pearson":
            rho, _ = stats.pearsonr(values, labels)
        else:
            raise ValueError(f"Unknown method: {method!r}")
        correlations[name] = float(rho) if np.isfinite(rho) else 0.0
    return correlations


def compute_cross_domain_fsi(
    corr_a: dict[str, float],
    corr_b: dict[str, float],
) -> dict[str, float]:
    """Compute cross-domain Feature Stability Index for a pair of domains.

    FSI(f, D_A, D_B) = min(|rho_A|, |rho_B|) * sign(rho_A * rho_B)

    Negative FSI indicates polarity reversal: the feature has opposite
    relationships with the label in the two domains.

    Args:
        corr_a: Feature-label correlations in domain A.
        corr_b: Feature-label correlations in domain B.

    Returns:
        Mapping of feature name to FSI value. Only includes features
        present in both domains.
    """
    common_features = set(corr_a.keys()) & set(corr_b.keys())
    fsi: dict[str, float] = {}
    for f in common_features:
        rho_a, rho_b = corr_a[f], corr_b[f]
        magnitude = min(abs(rho_a), abs(rho_b))
        product = rho_a * rho_b
        sign = 1.0 if product >= 0 else -1.0
        fsi[f] = magnitude * sign
    return fsi


def identify_puf(
    fsi_values: dict[str, float],
    threshold: float = 0.0,
) -> set[str]:
    """Identify Polarity-Unstable Features (PUF).

    Args:
        fsi_values: FSI values per feature from compute_cross_domain_fsi.
        threshold: Features with FSI < threshold are unstable.
            Default 0.0 catches any reversal.

    Returns:
        Set of feature names that are polarity-unstable.
    """
    return {f for f, v in fsi_values.items() if v < threshold}


def compute_polarity_matrix(
    domain_features: dict[str, dict[str, np.ndarray]],
    domain_labels: dict[str, np.ndarray],
    method: str = "spearman",
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute NxN FSI matrix across all domain pairs.

    Args:
        domain_features: {domain_name: {feature_name: values}}.
        domain_labels: {domain_name: labels}.
        method: Correlation method.

    Returns:
        Nested dict: result[domain_a][domain_b] = {feature: fsi_value}.
    """
    domain_corrs: dict[str, dict[str, float]] = {}
    for domain in domain_features:
        domain_corrs[domain] = compute_domain_correlations(
            domain_features[domain],
            domain_labels[domain],
            method=method,
        )

    domains = sorted(domain_features.keys())
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for da in domains:
        matrix[da] = {}
        for db in domains:
            if da == db:
                matrix[da][db] = {
                    f: abs(v) for f, v in domain_corrs[da].items()
                }
            else:
                matrix[da][db] = compute_cross_domain_fsi(
                    domain_corrs[da], domain_corrs[db],
                )
    return matrix


def fsi_significance_test(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    method: str = "spearman",
) -> dict[str, dict[str, float]]:
    """Permutation test for feature-label correlation significance.

    Shuffles labels n_permutations times and computes the fraction of
    permuted correlations that exceed the observed correlation in
    absolute value.

    Args:
        features: Feature arrays for one domain.
        labels: Binary labels.
        n_permutations: Number of permutation iterations.
        seed: Random seed.
        method: Correlation method.

    Returns:
        {feature_name: {"rho": observed, "p_value": p}}.
    """
    rng = np.random.default_rng(seed)
    observed = compute_domain_correlations(features, labels, method=method)

    results: dict[str, dict[str, float]] = {}
    for name, values in features.items():
        obs_rho = observed[name]
        count = 0
        for _ in range(n_permutations):
            perm_labels = rng.permutation(labels)
            if method == "spearman":
                perm_rho, _ = stats.spearmanr(values, perm_labels)
            else:
                perm_rho, _ = stats.pearsonr(values, perm_labels)
            if not np.isfinite(perm_rho):
                continue
            if abs(perm_rho) >= abs(obs_rho):
                count += 1
        p_value = (count + 1) / (n_permutations + 1)
        results[name] = {"rho": obs_rho, "p_value": p_value}

    return results


def aggregate_fsi(
    fsi_values: dict[str, float],
) -> float:
    """Compute aggregate FSI across all features for a domain pair.

    Uses the minimum FSI (weakest-link principle): the most reversed
    feature dominates the aggregate instability.

    Args:
        fsi_values: FSI per feature from compute_cross_domain_fsi.

    Returns:
        Minimum FSI value across features. Negative means reversal.
    """
    if not fsi_values:
        return 0.0
    return min(fsi_values.values())


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size (standardized mean difference).

    Convention: d > 0 means group_a has higher mean than group_b.

    Args:
        group_a: First sample (e.g., anomalous traces).
        group_b: Second sample (e.g., normal traces).

    Returns:
        Cohen's d using pooled standard deviation.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)
