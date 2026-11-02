"""Permutation null for E1's ten-split average (second adversarial review; exploratory, prereg A10).

The registered permutation test used seed-0 task halves only. The paper quotes the average over ten
task splits, so this test computes the null of that average. Each shuffle pools half A of one
configuration with half B of another, permutes the pooled runs between them (sizes kept), and
recomputes the flip rate. This is repeated on each split's halves and averaged over the ten splits.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import run_v50_cross_agent_attacks as atk

from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_cross_agent_tensplit_perm.json")
TAUS = (0.1, 0.2, 0.3)
N_PERM = 1000
EXPECTED = {0.1: 0.2248, 0.2: 0.1326, 0.3: 0.0747}


def _halves(corpora: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for r, (_, y, tasks) in corpora.items():
        h = e1._task_half(tasks, seed)
        out[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
    return out


def _observed(corpora: dict, pairs: list, splits: list[dict]) -> dict[float, float]:
    flips = {t: [] for t in TAUS}
    for halves in splits:
        d = {
            (r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
            for r in corpora
            for s in (0, 1)
        }
        for t in TAUS:
            flips[t] += [
                is_reversed(a, b, tau=t) for p in pairs for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])
            ]
    return {t: float(np.mean(v)) for t, v in flips.items()}


def _null_draw(corpora: dict, pairs: list, splits: list[dict], rng: np.random.Generator) -> dict:
    flips = {t: [] for t in TAUS}
    for halves in splits:
        for r1, r2 in pairs:
            (x1, y1, _), (x2, y2, _) = corpora[r1], corpora[r2]
            a, b = halves[r1][0], halves[r2][1]
            xp, yp = np.vstack([x1[a], x2[b]]), np.concatenate([y1[a], y2[b]])
            perm = rng.permutation(len(yp))
            da = e1._d(xp, yp, perm[: len(a)])[:5]
            db = e1._d(xp, yp, perm[len(a) :])[:5]
            for t in TAUS:
                flips[t] += [is_reversed(p, q, tau=t) for p, q in zip(da, db)]
    return {t: float(np.mean(v)) for t, v in flips.items()}


def main() -> None:
    corpora = {
        r: c
        for r in sorted(p.name for p in e1.RUNS.iterdir())
        if (c := atk._corpus(r, "important_instructions"))
    }
    pairs = list(itertools.combinations(corpora, 2))
    splits = [_halves(corpora, s) for s in e1.SEEDS]
    obs = _observed(corpora, pairs, splits)
    for t in TAUS:
        assert abs(obs[t] - EXPECTED[t]) < 5e-4, (t, obs[t])
    rng = np.random.default_rng(0)
    null = {t: [] for t in TAUS}
    for i in range(N_PERM):
        draw = _null_draw(corpora, pairs, splits, rng)
        for t in TAUS:
            null[t].append(draw[t])
        if i % 100 == 0:
            print("perm", i, flush=True)
    out = {"n_perm": N_PERM, "n_splits": len(splits), "n_configurations": len(corpora)}
    for t in TAUS:
        out[f"tau={t}"] = {
            "cross": obs[t],
            "perm_null_mean": float(np.mean(null[t])),
            "perm_null_max": float(np.max(null[t])),
            "perm_p": float((1 + sum(n >= obs[t] for n in null[t])) / (1 + N_PERM)),
        }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
