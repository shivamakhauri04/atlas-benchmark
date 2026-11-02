"""E2: should an engineer reuse an anomaly monitor on a new agent, retrain it, or gate the choice?

For each ordered (source, target) pair, k labeled target traces are drawn (stratified, both classes
present) from target half A; every policy is scored by AUROC on target half B. Gate thresholds were
fixed here on 2026-10-14 before any E2 number existed:
- sign gate: retrain if any feature with |d| > 0.1 in both the source and the k labels flips sign
- direct gate: retrain if the reused detector's AUROC on the k labels is below 0.5
- divergence gate: retrain if the label-free mean per-feature Wasserstein distance exceeds the
  median over all pairs
Regret for a pair = max(AUROC_reuse, AUROC_retrain) - AUROC_chosen, averaged over seeds, then over
sources per target, then over targets.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

from traceshift.v50 import _make, _zscore, cohens_d_anomalous, stratified_halves

import run_v50_grid as grid

OUT = Path("output/tables/v50_reuse_policy.json")
KS = (10, 20, 50, 100)
SEEDS = range(10)
DETECTORS = ("rf", "knn", "lr")
TAU = 0.1


def _draw_k(y: np.ndarray, idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_pos = int(np.clip(round(k * len(pos) / len(idx)), 1, min(len(pos), k - 1)))
    n_neg = min(k - n_pos, len(neg))
    return np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def _pair(xs, ys, xt, yt, det: str, k: int, seed: int, wdist: float, wmed: float) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a_s, _ = stratified_halves(ys, seed=seed)
    a_t, b_t = stratified_halves(yt, seed=seed)
    kidx = _draw_k(yt, a_t, k, rng)
    zs, zt = _zscore(xs), _zscore(xt)
    reuse_model = _make(det, seed, len(a_s)).fit(zs[a_s], ys[a_s])
    retrain_model = _make(det, seed, len(kidx)).fit(zt[kidx], yt[kidx])
    reuse = _auroc(yt[b_t], reuse_model.predict_proba(zt[b_t])[:, 1])
    retrain = _auroc(yt[b_t], retrain_model.predict_proba(zt[b_t])[:, 1])
    d_src = [cohens_d_anomalous(xs[a_s, j], ys[a_s]) for j in range(xs.shape[1])]
    d_k = [cohens_d_anomalous(xt[kidx, j], yt[kidx]) for j in range(xt.shape[1])]
    flip = any(abs(a) > TAU and abs(b) > TAU and np.sign(a) != np.sign(b) for a, b in zip(d_src, d_k))
    direct_k = _auroc(yt[kidx], reuse_model.predict_proba(zt[kidx])[:, 1])
    best = max(reuse, retrain)
    choose = {
        "always_reuse": reuse,
        "always_retrain": retrain,
        "sign_gate": retrain if flip else reuse,
        "direct_gate": retrain if direct_k < 0.5 else reuse,
        "divergence_gate": retrain if wdist > wmed else reuse,
    }
    return {p: best - v for p, v in choose.items()} | {"reuse": reuse, "retrain": retrain}


def main() -> None:
    data = grid.load_primary()
    z = {k: _zscore(v[0]) for k, v in data.items()}
    wd = {(s, t): float(np.mean([wasserstein_distance(z[s][:, j], z[t][:, j]) for j in range(6)]))
          for s, t in itertools.permutations(data, 2)}
    wmed = float(np.median(list(wd.values())))
    result: dict[str, object] = {"wasserstein_median": wmed}
    for det, k in itertools.product(DETECTORS, KS):
        per_target: dict[str, list[dict[str, float]]] = {t: [] for t in data}
        for s, t in itertools.permutations(data, 2):
            (xs, ys), (xt, yt) = data[s], data[t]
            runs = [_pair(xs, ys, xt, yt, det, k, sd, wd[(s, t)], wmed) for sd in SEEDS]
            per_target[t].append({key: float(np.mean([r[key] for r in runs])) for key in runs[0]})
        keys = list(next(iter(per_target.values()))[0])
        by_target = {t: {key: float(np.mean([r[key] for r in rows])) for key in keys}
                     for t, rows in per_target.items()}
        result[f"{det}/k={k}"] = {
            "regret": {key: float(np.mean([v[key] for v in by_target.values()]))
                       for key in keys if key not in ("reuse", "retrain")},
            "mean_auroc": {key: float(np.mean([v[key] for v in by_target.values()]))
                           for key in ("reuse", "retrain")},
            "by_target": by_target,
        }
        print(det, k, json.dumps(result[f"{det}/k={k}"]["regret"]), flush=True)
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
