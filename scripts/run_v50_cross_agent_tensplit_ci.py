"""Configuration-level bootstrap interval for E1's ten-split average (exploratory, prereg A17).

The paper's headline is the ten-split average; the earlier interval was computed on one split. This
averages each pair's reversal rate over the ten splits, then resamples configurations with
replacement and weights each pair by how often its two configurations were drawn.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import run_v50_cross_agent_attacks as atk
import run_v50_cross_agent_tensplit_perm as tp

from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_cross_agent_tensplit_ci.json")
N_BOOT = 2000


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir())
               if (c := atk._corpus(r, "important_instructions"))}
    configs = list(corpora)
    pairs = list(itertools.combinations(configs, 2))
    splits = [tp._halves(corpora, s) for s in e1.SEEDS]
    per_pair = {t: {p: [] for p in pairs} for t in tp.TAUS}
    for halves in splits:
        d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
             for r in corpora for s in (0, 1)}
        for t in tp.TAUS:
            for p in pairs:
                per_pair[t][p] += [is_reversed(a, b, tau=t)
                                   for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])]
    idx = {c: i for i, c in enumerate(configs)}
    si = np.array([idx[a] for a, _ in pairs])
    ti = np.array([idx[b] for _, b in pairs])
    rng = np.random.default_rng(0)
    out = {"n_configurations": len(configs), "n_boot": N_BOOT}
    for t in tp.TAUS:
        rates = np.array([np.mean(per_pair[t][p]) for p in pairs])
        boots = []
        for _ in range(N_BOOT):
            c = np.bincount(rng.integers(0, len(configs), len(configs)), minlength=len(configs))
            w = c[si] * c[ti]
            if w.sum():
                boots.append(float(np.average(rates, weights=w)))
        out[f"tau={t}"] = {"cross": float(rates.mean()),
                           "config_bootstrap_ci95": [float(np.percentile(boots, 2.5)),
                                                     float(np.percentile(boots, 97.5))]}
    assert abs(out["tau=0.1"]["cross"] - 0.2248) < 5e-4, out["tau=0.1"]
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
