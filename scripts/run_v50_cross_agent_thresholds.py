"""E1 under stricter flip definitions, with a finer permutation null (post-result adversarial pass).

- Thresholds |d| > 0.1, 0.2, 0.3 (0.2 is the conventional "small effect").
- CI-aware flip: both agents' 95% bootstrap CIs for d exclude 0, with opposite signs.
- Permutation null with 2,000 shuffles of agent labels within each pair (sizes kept).
Seed-0 task halves, the same halves as `run_v50_cross_agent_robustness.py`.
"""

import itertools
import json
from pathlib import Path

import numpy as np

import run_v50_cross_agent as e1
from traceshift.v50 import cohens_d_anomalous

OUT = Path("output/tables/v50_cross_agent_thresholds.json")
TAUS = (0.1, 0.2, 0.3)
N_PERM = 2000
N_BOOT = 200


def _flip(a: float, b: float, tau: float) -> bool:
    return abs(a) > tau and abs(b) > tau and np.sign(a) != np.sign(b)


def _ci(x: np.ndarray, y: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """95% bootstrap CI of d per feature, shape (5, 2)."""
    boots = []
    for _ in range(N_BOOT):
        s = rng.choice(idx, len(idx), replace=True)
        boots.append([cohens_d_anomalous(x[s, j], y[s]) for j in range(5)])
    return np.percentile(np.array(boots), [2.5, 97.5], axis=0).T


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir()) if (c := e1._corpus(r))}
    rng = np.random.default_rng(0)
    halves = {}
    for r, (x, y, tasks) in corpora.items():
        h = e1._task_half(tasks, 0)
        halves[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
    d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5] for r in corpora for s in (0, 1)}
    pairs = list(itertools.combinations(corpora, 2))

    out: dict[str, object] = {}
    for tau in TAUS:
        cross = np.mean([_flip(a, b, tau) for r1, r2 in pairs for a, b in zip(d[(r1, 0)], d[(r2, 1)])])
        within = np.mean([_flip(a, b, tau) for r in corpora for a, b in zip(d[(r, 0)], d[(r, 1)])])
        out[f"tau={tau}"] = {"cross": float(cross), "within": float(within)}

    null = {tau: [] for tau in TAUS}
    for _ in range(N_PERM):
        flips = {tau: [] for tau in TAUS}
        for r1, r2 in pairs:
            (x1, y1, _), (x2, y2, _) = corpora[r1], corpora[r2]
            a, b = halves[r1][0], halves[r2][1]
            xp, yp = np.vstack([x1[a], x2[b]]), np.concatenate([y1[a], y2[b]])
            perm = rng.permutation(len(yp))
            da, db = e1._d(xp, yp, perm[: len(a)])[:5], e1._d(xp, yp, perm[len(a):])[:5]
            for tau in TAUS:
                flips[tau] += [_flip(p, q, tau) for p, q in zip(da, db)]
        for tau in TAUS:
            null[tau].append(float(np.mean(flips[tau])))
    for tau in TAUS:
        obs = out[f"tau={tau}"]["cross"]
        out[f"tau={tau}"]["perm_null_mean"] = float(np.mean(null[tau]))
        out[f"tau={tau}"]["perm_p"] = float((1 + sum(n >= obs for n in null[tau])) / (1 + N_PERM))

    ci = {(r, s): _ci(corpora[r][0], corpora[r][1], halves[r][s], rng) for r in corpora for s in (0, 1)}

    def ci_flip(c1: np.ndarray, c2: np.ndarray) -> list[bool]:
        return [(lo1 > 0 and hi2 < 0) or (hi1 < 0 and lo2 > 0)
                for (lo1, hi1), (lo2, hi2) in zip(c1, c2)]

    out["ci_aware"] = {
        "cross": float(np.mean([v for r1, r2 in pairs for v in ci_flip(ci[(r1, 0)], ci[(r2, 1)])])),
        "within": float(np.mean([v for r in corpora for v in ci_flip(ci[(r, 0)], ci[(r, 1)])])),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
