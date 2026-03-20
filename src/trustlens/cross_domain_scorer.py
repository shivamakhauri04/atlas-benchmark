"""Cross-domain scorer using LogisticRegression for transfer experiments.

Trains on source domain features+labels. When applied to a target domain
with reversed feature polarity, the learned weights become counterproductive,
producing measurable AUROC degradation correlated with FSI.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _features_to_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Stack feature dict into (n_samples, n_features) matrix."""
    return np.column_stack([features[k] for k in sorted(features.keys())])


class CrossDomainScorer:
    """LogisticRegression scorer for cross-domain anomaly detection.

    Args:
        seed: Random state for train/cal split and solver.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )

    def fit(
        self,
        features: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> np.ndarray:
        """Train on 70% of source, return calibration scores on held-out 30%.

        Args:
            features: {feature_name: values} for source domain.
            labels: Binary labels, shape (n,).

        Returns:
            Calibration scores (predict_proba[:, 1]) on the 30% held-out split.

        Raises:
            ValueError: If stratified split fails (too few samples in a class).
        """
        X = _features_to_matrix(features)
        X_train, X_cal, y_train, _ = train_test_split(
            X, labels, test_size=0.3, stratify=labels, random_state=self.seed,
        )
        self._model.fit(X_train, y_train)
        return self._model.predict_proba(X_cal)[:, 1]

    def score(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Score target domain using learned source weights.

        Returns:
            P(anomalous) for each sample, shape (n,).
        """
        X = _features_to_matrix(features)
        return self._model.predict_proba(X)[:, 1]
