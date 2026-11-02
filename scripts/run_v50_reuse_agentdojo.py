"""E2b: the reuse decision inside AgentDojo, where source and target share tasks and label meaning.

Unlike E2, the label concept is identical across corpora (injection succeeded vs resisted, one
attack type), so a reused monitor is not doomed by label mismatch and reuse can win. Added after
the post-result adversarial pass (a review of an earlier draft). Gate thresholds are
those of E2, unchanged.

For each ordered pair of the 22 configurations, and for k in {10, 20, 50, 100}:
- source detector trained on the source's task half A
- k stratified target traces drawn from the target's task half A
- every policy scored on the target's task half B (tasks disjoint from training)
Detectors RF, kNN and LR, on all 6 features and on length-free features.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score

import run_v50_cross_agent as e1
from traceshift.v50 import _make, _zscore, cohens_d_anomalous

OUT = Path("output/tables/v50_reuse_agentdojo.json")
KS = (10, 20, 50, 100)
SEEDS = range(5)
DETECTORS = ("rf", "knn", "lr")
TAU = 0.1
FEATURE_SETS = {"all": list(range(6)), "length_free": list(range(5))}


def _draw(y: np.ndarray, idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n_pos = int(np.clip(round(k * len(pos) / len(idx)), 1, min(len(pos), k - 1)))
    return np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, min(k - n_pos, len(neg)), replace=False)])


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir()) if (c := e1._corpus(r))}
    # Reason: the distance gate (added after the venue review) uses only unlabeled features, so it is
    # computed once and leaves the random stream of the other four policies unchanged.
    zs = {r: _zscore(c[0]) for r, c in corpora.items()}
    wd = {(s, t): float(np.mean([wasserstein_distance(zs[s][:, j], zs[t][:, j]) for j in range(6)]))
          for s, t in itertools.permutations(corpora, 2)}
    wmed = float(np.median(list(wd.values())))
    # Reason: a held-out threshold for each target, the median over pairs that do not involve that
    # target, so the gate never sees the pair it is scored on (venue review, A9).
    wmed_loto = {t: float(np.median([v for (a, b), v in wd.items() if t not in (a, b)]))
                 for t in corpora}
    result: dict[str, object] = {"n_configs": len(corpora), "wasserstein_median": wmed}
    for (fs, cols), det, k in itertools.product(FEATURE_SETS.items(), DETECTORS, KS):
        acc: dict[str, list[float]] = {p: [] for p in
                                      ("always_reuse", "always_retrain", "sign_gate", "direct_gate",
                                       "divergence_gate", "divergence_gate_heldout")}
        per_pair: dict[str, dict[str, list[float]]] = {}
        auc_r, auc_t = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            half = {r: e1._task_half(c[2], seed) for r, c in corpora.items()}
            models = {}
            for r, (x, y, _) in corpora.items():
                a = np.flatnonzero(half[r] == 0)
                models[r] = _make(det, seed, len(a)).fit(_zscore(x)[a][:, cols], y[a])
            for s, t in itertools.permutations(corpora, 2):
                (xs, ys, _), (xt, yt, _) = corpora[s], corpora[t]
                a_s = np.flatnonzero(half[s] == 0)
                a_t, b_t = np.flatnonzero(half[t] == 0), np.flatnonzero(half[t] == 1)
                if len(set(yt[b_t])) < 2:
                    continue
                kidx = _draw(yt, a_t, k, rng)
                zt = _zscore(xt)[:, cols]
                reuse_m = models[s]
                retrain_m = _make(det, seed, len(kidx)).fit(zt[kidx], yt[kidx])
                reuse = float(roc_auc_score(yt[b_t], reuse_m.predict_proba(zt[b_t])[:, 1]))
                retrain = float(roc_auc_score(yt[b_t], retrain_m.predict_proba(zt[b_t])[:, 1]))
                d_s = [cohens_d_anomalous(xs[a_s][:, j], ys[a_s]) for j in cols]
                d_k = [cohens_d_anomalous(xt[kidx][:, j], yt[kidx]) for j in cols]
                flip = any(abs(p) > TAU and abs(q) > TAU and np.sign(p) != np.sign(q)
                           for p, q in zip(d_s, d_k))
                direct = float(roc_auc_score(yt[kidx], reuse_m.predict_proba(zt[kidx])[:, 1]))
                best = max(reuse, retrain)
                for p, v in (("always_reuse", reuse), ("always_retrain", retrain),
                             ("sign_gate", retrain if flip else reuse),
                             ("direct_gate", retrain if direct < 0.5 else reuse),
                             ("divergence_gate", retrain if wd[(s, t)] > wmed else reuse),
                             ("divergence_gate_heldout",
                              retrain if wd[(s, t)] > wmed_loto[t] else reuse)):
                    acc[p].append(best - v)
                    per_pair.setdefault(f"{s}|{t}", {}).setdefault(p, []).append(best - v)
                auc_r.append(reuse)
                auc_t.append(retrain)
        key = f"{fs}/{det}/k={k}"
        result[key] = {"regret": {p: float(np.mean(v)) for p, v in acc.items()},
                       "mean_auroc_reuse": float(np.mean(auc_r)),
                       "mean_auroc_retrain": float(np.mean(auc_t)),
                       "reuse_wins_share": float(np.mean(np.array(auc_r) > np.array(auc_t))),
                       "per_pair_regret": {pr: {p: float(np.mean(v)) for p, v in d.items()}
                                           for pr, d in per_pair.items()}}
        print(key, json.dumps(result[key]["regret"]), flush=True)
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
