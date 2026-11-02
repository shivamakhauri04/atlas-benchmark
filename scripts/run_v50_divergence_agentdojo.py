"""Does label-free distribution distance predict transfer within AgentDojo? (prereg-v50 A9, exploratory)

The 12-dataset pool gave negative leave-one-target-out R^2 for transfer AUROC. Here source and target
share tasks and label meaning. For each ordered pair of the 22 configurations: transfer AUROC of a
random forest trained on the source's task half A and tested on the target's task half B (all six
features, 5 splits), and the mean per-feature Wasserstein distance between the two configurations'
z-scored unlabeled features.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

import run_v50_cross_agent as e1
from traceshift.v50 import _zscore, transfer_auroc

OUT = Path("output/tables/v50_divergence_agentdojo.json")
SPLITS = range(5)


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir()) if (c := e1._corpus(r))}
    zs = {r: _zscore(c[0]) for r, c in corpora.items()}
    rows = []
    for s, t in itertools.permutations(corpora, 2):
        (xs, ys, ts), (xt, yt, tt) = corpora[s], corpora[t]
        aurocs = []
        for seed in SPLITS:
            a = np.flatnonzero(e1._task_half(ts, seed) == 0)
            b = np.flatnonzero(e1._task_half(tt, seed) == 1)
            aurocs.append(transfer_auroc((xs[a], ys[a]), (xt[b], yt[b]), detector="rf", seed=seed))
        w = float(np.mean([wasserstein_distance(zs[s][:, j], zs[t][:, j]) for j in range(6)]))
        rows.append({"source": s, "target": t, "auroc": float(np.mean(aurocs)), "wasserstein": w})
    y = np.array([r["auroc"] for r in rows])
    x = np.array([r["wasserstein"] for r in rows])
    pred = np.empty_like(y)
    for t in corpora:
        test = np.array([r["target"] == t for r in rows])
        coef = np.polyfit(x[~test], y[~test], 1)
        pred[test] = np.polyval(coef, x[test])
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rho, p = spearmanr(x, y)
    result = {"n_pairs": len(rows), "loto_r2": r2, "spearman_rho": float(rho), "spearman_p": float(p),
              "mean_auroc": float(y.mean()), "rows": rows}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
