"""Split-sample polarity estimation and source-dependent transfer for the v50 rerun.

Signs are estimated on one stratified half of a dataset and scored on the other, so a reversal is
never selected and measured on the same labels. Analysis choices are fixed in
`docs/prereg-v50.md`.
"""

from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from traceshift.polarity import cohens_d

Detector = Literal["lr", "knn", "rf"]
Domain = tuple[np.ndarray, np.ndarray]


def cohens_d_anomalous(x: np.ndarray, y: np.ndarray) -> float:
    """Return Cohen's d of anomalous (y=1) minus normal (y=0) traces."""
    return cohens_d(x[y == 1], x[y == 0])


def is_reversed(d_source: float, d_target: float, *, tau: float) -> bool:
    """Return True when both effects exceed `tau` in magnitude and have opposite signs."""
    return abs(d_source) > tau and abs(d_target) > tau and np.sign(d_source) != np.sign(d_target)


def stratified_halves(y: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split indices 50/50 within each class."""
    rng = np.random.default_rng(seed)
    half_a: list[int] = []
    half_b: list[int] = []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        cut = len(idx) // 2
        half_a.extend(idx[:cut].tolist())
        half_b.extend(idx[cut:].tolist())
    return np.array(sorted(half_a)), np.array(sorted(half_b))


def _zscore(x: np.ndarray) -> np.ndarray:
    # Reason: per-domain standardization uses only unlabeled feature statistics, which a deployed
    # monitor has for its new target; no target label enters the transfer score.
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def _make(
    detector: Detector, seed: int, n_train: int
) -> LogisticRegression | KNeighborsClassifier | RandomForestClassifier:
    if detector == "lr":
        return LogisticRegression(class_weight="balanced", random_state=seed, max_iter=500)
    if detector == "knn":
        return KNeighborsClassifier(n_neighbors=min(5, max(1, n_train - 1)))
    return RandomForestClassifier(n_estimators=50, max_depth=3, random_state=seed)


def transfer_auroc(source: Domain, target: Domain, *, detector: Detector, seed: int) -> float:
    """Train on the source domain and return AUROC on the target domain.

    Raises:
        ValueError: If either domain lacks one of the two classes.
    """
    (xs, ys), (xt, yt) = source, target
    if len(set(ys.tolist())) < 2 or len(set(yt.tolist())) < 2:
        raise ValueError("source and target must both contain normal and anomalous traces")
    model = _make(detector, seed, len(ys)).fit(_zscore(xs), ys)
    return float(roc_auc_score(yt, model.predict_proba(_zscore(xt))[:, 1]))
