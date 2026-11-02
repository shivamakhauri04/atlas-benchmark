"""E1 robustness after the venue review: a second attack, ten-split rates at every threshold,
and bootstrap intervals by pair type (prereg-v50 A9; not a confirmatory test).

Same design as `run_v50_cross_agent.py`: length-free features, task-disjoint halves, class-balanced
subsampling, configurations with at least 30 runs in each class. Pass the attack as argv[1].
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

import run_v50_cross_agent as e1
from data import domain_loader as dl
from data.trace_loader import load_agentdojo
from traceshift.v50 import is_reversed

TAUS = (0.1, 0.2, 0.3)
N_BOOT = 2000
BASES = ("gpt-4o-2024-05-13", "meta-llama_Llama-3.3-70B-Instruct", "Meta-SecAlign-70B",
         "claude-3-sonnet-20240229")


def _base(run: str) -> str:
    return next((b for b in BASES if run.startswith(b)), run)


def _corpus(run: str, attack: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    traces = load_agentdojo(e1.RUNS / run, min_tool_calls=1, attack=attack)
    y = np.array([int(t.has_anomaly) for t in traces])
    if len(y) == 0 or min(int(y.sum()), int(len(y) - y.sum())) < e1.MIN_CLASS:
        return None
    feats = dl._extract_all_features(traces)
    x = np.column_stack([np.asarray(feats[f], float) for f in (*e1.FEATURES, "trace_length")])
    return x, y, [t.task for t in traces]


def _boot_ci(pair_rates: list[float], rng: np.random.Generator) -> list[float]:
    arr = np.array(pair_rates)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main() -> None:
    attack = sys.argv[1]
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir()) if (c := _corpus(r, attack))}
    print(attack, "configurations", len(corpora), flush=True)
    pairs = list(itertools.combinations(corpora, 2))
    cross = {t: [] for t in TAUS}
    within = {t: [] for t in TAUS}
    per_pair = {t: {p: [] for p in pairs} for t in TAUS}
    for seed in e1.SEEDS:
        rng = np.random.default_rng(seed)
        halves = {}
        for r, (x, y, tasks) in corpora.items():
            h = e1._task_half(tasks, seed)
            halves[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
        d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
             for r in corpora for s in (0, 1)}
        for t in TAUS:
            within[t] += [is_reversed(a, b, tau=t) for r in corpora
                          for a, b in zip(d[(r, 0)], d[(r, 1)])]
            for p in pairs:
                fl = [is_reversed(a, b, tau=t) for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])]
                cross[t] += fl
                per_pair[t][p].append(float(np.mean(fl)))
    rng = np.random.default_rng(0)
    out: dict[str, object] = {"attack": attack, "n_configurations": len(corpora),
                              "configurations": list(corpora)}
    for t in TAUS:
        rates = {p: float(np.mean(v)) for p, v in per_pair[t].items()}
        scaffold = [v for p, v in rates.items() if _base(p[0]) == _base(p[1])]
        model = [v for p, v in rates.items() if _base(p[0]) != _base(p[1])]
        out[f"tau={t}"] = {
            "cross": float(np.mean(cross[t])), "within": float(np.mean(within[t])),
            "pair_bootstrap_ci95": _boot_ci(list(rates.values()), rng),
            "different_model": {"rate": float(np.mean(model)) if model else None,
                                "n_pairs": len(model),
                                "ci95": _boot_ci(model, rng) if model else None},
            "same_model_scaffold": {"rate": float(np.mean(scaffold)) if scaffold else None,
                                    "n_pairs": len(scaffold),
                                    "ci95": _boot_ci(scaffold, rng) if len(scaffold) > 1 else None},
        }
    path = Path(f"output/tables/v50_cross_agent_{attack}.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "configurations"}, indent=1))


if __name__ == "__main__":
    main()
