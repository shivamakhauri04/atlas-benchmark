"""Hybrid scorer: combines LSTM NLL with heuristic features for domain-robust scoring.

Two modes:
- 'equal': Unsupervised equal-weight z-score combination (no labels needed)
- 'fitted': Supervised logistic regression on held-out labeled data

Z-score normalization is fitted on normal (non-anomalous) data only, consistent
with the normal-only training principle used throughout TrustLens.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ("lstm_nll", "trace_length", "low_diversity", "repeat_rate")
VALID_MODES = ("equal", "fitted", "rf", "gbt")
SUPERVISED_MODES = ("fitted", "rf", "gbt")


class HybridScorer:
    """Combines LSTM NLL with heuristic features for domain-robust scoring."""

    def __init__(self, mode: str = "equal"):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.classifier: LogisticRegression | None = None

    def fit_normalizer(self, normal_features: dict[str, np.ndarray]) -> None:
        """Fit z-score normalization on normal (non-anomalous) data."""
        for name, values in normal_features.items():
            self.means[name] = float(values.mean())
            self.stds[name] = max(float(values.std()), 1e-8)

    def fit_classifier(
        self,
        features: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """Fit classifier on labeled data (for supervised modes)."""
        X = self._stack_and_normalize(features)
        if self.mode == "fitted":
            self.classifier = LogisticRegression(max_iter=1000).fit(X, labels)
        elif self.mode == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42,
            ).fit(X, labels)
        elif self.mode == "gbt":
            self.classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42,
            ).fit(X, labels)
        else:
            raise RuntimeError(f"fit_classifier not applicable for mode={self.mode!r}")

    def score(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Produce hybrid anomaly scores (higher = more anomalous)."""
        X = self._stack_and_normalize(features)
        if self.mode == "equal":
            return X.mean(axis=1)
        if self.classifier is None:
            raise RuntimeError(
                f"Call fit_classifier() before scoring in '{self.mode}' mode"
            )
        return self.classifier.predict_proba(X)[:, 1]

    def _stack_and_normalize(
        self, features: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Stack feature arrays and apply z-score normalization."""
        if not self.means:
            raise RuntimeError("Call fit_normalizer() before scoring")
        names = sorted(features.keys())
        cols = []
        for name in names:
            z = (features[name] - self.means[name]) / self.stds[name]
            cols.append(z)
        return np.column_stack(cols)


def build_feature_dict(
    lstm_nll: np.ndarray,
    heuristics: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Merge LSTM NLL scores with heuristic feature arrays."""
    return {"lstm_nll": lstm_nll, **heuristics}
