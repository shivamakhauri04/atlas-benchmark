"""Robustness of E1 (cross-agent polarity flips on AgentDojo).

1. Per-feature cross-agent and within-agent flip rates.
2. Pair types: different base model vs same base model with a different scaffold/defense.
3. Permutation null: for each agent pair, pool both agents' traces and reassign agent labels at
   random (sizes kept), so the null has the same sample sizes but no agent difference.
4. Cluster bootstrap over backbones for the cross-agent flip rate.
Seed 0 task halves only; E1's main estimate averages 10 seeds.
"""

import itertools
import json
from pathlib import Path

import numpy as np

import run_v50_cross_agent as e1
from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_cross_agent_robustness.json")
N_PERM = 200
N_BOOT = 1000
BASES = {
    "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "meta-llama_Llama-3.3-70B-Instruct": "meta-llama_Llama-3.3-70B-Instruct",
}


def _base(run: str) -> str:
    return next((b for b in BASES if run.startswith(b)), run)


def _flips(da: np.ndarray, db: np.ndarray) -> list[bool]:
    return [is_reversed(a, b, tau=e1.TAU) for a, b in zip(da[:5], db[:5])]


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir()) if (c := e1._corpus(r))}
    rng = np.random.default_rng(0)
    halves = {}
    for r, (x, y, tasks) in corpora.items():
        h = e1._task_half(tasks, 0)
        halves[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
    d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s]) for r in corpora for s in (0, 1)}

    per_feature = {f: {"cross": [], "within": []} for f in e1.FEATURES}
    for r in corpora:
        for f, fl in zip(e1.FEATURES, _flips(d[(r, 0)], d[(r, 1)])):
            per_feature[f]["within"].append(fl)
    pair_flips: dict[tuple[str, str], list[bool]] = {}
    for r1, r2 in itertools.combinations(corpora, 2):
        fl = _flips(d[(r1, 0)], d[(r2, 1)])
        pair_flips[(r1, r2)] = fl
        for f, v in zip(e1.FEATURES, fl):
            per_feature[f]["cross"].append(v)

    kinds = {"different_base_model": [], "same_model_scaffold": []}
    for (r1, r2), fl in pair_flips.items():
        kinds["same_model_scaffold" if _base(r1) == _base(r2) else "different_base_model"] += fl

    null_rates = []
    pairs = list(pair_flips)
    for _ in range(N_PERM):
        flips = []
        for r1, r2 in pairs:
            (x1, y1, _), (x2, y2, _) = corpora[r1], corpora[r2]
            a, b = halves[r1][0], halves[r2][1]
            xp = np.vstack([x1[a], x2[b]])
            yp = np.concatenate([y1[a], y2[b]])
            perm = rng.permutation(len(yp))
            ia, ib = perm[: len(a)], perm[len(a):]
            flips += _flips(e1._d(xp, yp, ia), e1._d(xp, yp, ib))
        null_rates.append(float(np.mean(flips)))

    runs = list(corpora)
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(runs, len(runs), replace=True)
        vals = [v for r1, r2 in itertools.combinations(pick, 2) if r1 != r2
                for v in pair_flips.get((r1, r2), pair_flips.get((r2, r1), []))]
        if vals:
            boot.append(float(np.mean(vals)))

    observed = float(np.mean([v for fl in pair_flips.values() for v in fl]))
    result = {
        "observed_cross_flip_rate_seed0": observed,
        "per_feature": {f: {k: float(np.mean(v)) for k, v in x.items()} for f, x in per_feature.items()},
        "pair_types": {k: {"flip_rate": float(np.mean(v)) if v else None, "n_cells": len(v)}
                       for k, v in kinds.items()},
        "permutation_null": {"mean": float(np.mean(null_rates)), "p95": float(np.percentile(null_rates, 95)),
                             "p_value": float((1 + sum(n >= observed for n in null_rates)) / (1 + N_PERM))},
        "backbone_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
