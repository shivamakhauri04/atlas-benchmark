"""E1 under other inclusion filters (paper review A8; exploratory, prereg A11).

The main analysis keeps configurations with at least 30 runs per class and runs with at least one
tool call. This reruns the ten-split design with 20 or 40 runs per class, and with at least three
tool calls (the loader default that prereg A3 replaced). The baseline cell must reproduce E1.
"""

import itertools
import json
from pathlib import Path

import numpy as np

import run_v50_cross_agent as e1
from data import domain_loader as dl
from data.trace_loader import load_agentdojo
from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_cross_agent_filters.json")
TAUS = (0.1, 0.2, 0.3)
FILTERS = ((30, 1), (20, 1), (40, 1), (30, 3))
EXPECTED_BASELINE = {0.1: (0.2248, 0.0582)}


def _corpus(run: str, min_class: int, min_calls: int):
    traces = load_agentdojo(e1.RUNS / run, min_tool_calls=min_calls)
    y = np.array([int(t.has_anomaly) for t in traces])
    if len(y) == 0 or min(int(y.sum()), int(len(y) - y.sum())) < min_class:
        return None
    feats = dl._extract_all_features(traces)
    x = np.column_stack([np.asarray(feats[f], float) for f in e1.FEATURES])
    return x, y, [t.task for t in traces]


def _rates(corpora: dict) -> dict:
    pairs = list(itertools.combinations(corpora, 2))
    cross = {t: [] for t in TAUS}
    within = {t: [] for t in TAUS}
    for seed in e1.SEEDS:
        rng = np.random.default_rng(seed)
        halves = {}
        for r, (_, y, tasks) in corpora.items():
            h = e1._task_half(tasks, seed)
            halves[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
        d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
             for r in corpora for s in (0, 1)}
        for t in TAUS:
            within[t] += [is_reversed(a, b, tau=t) for r in corpora
                          for a, b in zip(d[(r, 0)], d[(r, 1)])]
            cross[t] += [is_reversed(a, b, tau=t) for p in pairs
                         for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])]
    return {f"tau={t}": {"cross": float(np.mean(cross[t])), "within": float(np.mean(within[t]))}
            for t in TAUS}


def main() -> None:
    runs = sorted(p.name for p in e1.RUNS.iterdir())
    out = {}
    for min_class, min_calls in FILTERS:
        corpora = {r: c for r in runs if (c := _corpus(r, min_class, min_calls))}
        key = f"min_class={min_class},min_tool_calls={min_calls}"
        out[key] = {"n_configurations": len(corpora), **_rates(corpora)}
        print(key, json.dumps(out[key]), flush=True)
    base = out["min_class=30,min_tool_calls=1"]["tau=0.1"]
    for t, (c, w) in EXPECTED_BASELINE.items():
        assert abs(base["cross"] - c) < 5e-4 and abs(base["within"] - w) < 5e-4, base
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
