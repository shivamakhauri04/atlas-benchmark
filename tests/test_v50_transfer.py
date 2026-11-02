"""A transfer score must depend on the source domain.

The earlier version's `heuristic_raw` column was computed from target features only
(`scripts/run_expanded_cross_domain.py:116-118`), so it was identical for every source. This test
fixes the contract for the v50 transfer scorer and documents the old failure.
"""

import numpy as np

from traceshift.v50 import transfer_auroc


def _domain(rng: np.random.Generator, sign: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < 0.3).astype(int)
    x = rng.normal(size=(n, 3))
    x[:, 0] += sign * 1.5 * y
    return x, y


def test_transfer_auroc_changes_with_source() -> None:
    rng = np.random.default_rng(0)
    src_pos, src_neg, tgt = _domain(rng, +1), _domain(rng, -1), _domain(rng, +1)
    a = transfer_auroc(src_pos, tgt, detector="lr", seed=0)
    b = transfer_auroc(src_neg, tgt, detector="lr", seed=0)
    assert a > 0.7
    assert b < 0.3


def test_old_target_only_score_is_source_independent() -> None:
    rng = np.random.default_rng(1)
    tgt = _domain(rng, +1)

    def heuristic_raw(
        _source: tuple[np.ndarray, np.ndarray], target: tuple[np.ndarray, np.ndarray]
    ) -> float:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(target[1], target[0].mean(axis=1)))

    s1, s2 = _domain(rng, +1), _domain(rng, -1)
    assert heuristic_raw(s1, tgt) == heuristic_raw(s2, tgt)
