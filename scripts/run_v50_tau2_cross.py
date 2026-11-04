"""E1 on a third benchmark: τ²-bench, frontier models in a tool-calling loop (prereg-v50 A19).

Usage: python scripts/run_v50_tau2_cross.py <cohort.txt> <domain>

Label 1 = task failed (reward < 1). Task-disjoint halves by task id (all trials of a task on one
side), class-balanced, ten splits, five length-free features, tau 0.1 primary. The rate, null, and
bootstrap functions are the SWE-bench script's, unchanged.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import run_v50_swebench_cross as swe

from data import domain_loader as dl
from data.schema import Action, Trace

DATA = Path("data/tau2")


def _corpus(sub: str, domain: str):
    rows = [json.loads(line) for line in (DATA / f"{sub}_{domain}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["actions"] and r["reward"] is not None]
    traces = [Trace(trace_id=f"{r['task_id']}#{r['trial']}", task=r["task_id"], source="tau2",
                    actions=[Action(tool=a["tool"], args=a["args"]) for a in r["actions"]])
              for r in rows]
    y = np.array([1 if r["reward"] < 1 else 0 for r in rows])
    if min(int(y.sum()), int(len(y) - y.sum())) < e1.MIN_CLASS:
        return None
    feats = dl._extract_all_features(traces)
    x = np.column_stack([np.asarray(feats[f], float) for f in (*e1.FEATURES, "trace_length")])
    return x, y, [t.task for t in traces]


def main() -> None:
    subs = Path(sys.argv[1]).read_text().split()
    domain = sys.argv[2]
    corpora = {s: c for s in subs if (c := _corpus(s, domain))}
    systems = list(corpora)
    pairs = list(itertools.combinations(systems, 2))
    splits = [swe._halves(corpora, s) for s in e1.SEEDS]
    r = swe._rates(corpora, pairs, splits)
    rng = np.random.default_rng(0)
    nulls = [swe._null(corpora, pairs, splits, rng) for _ in range(swe.N_PERM)]
    out = {"benchmark": "tau2", "domain": domain, "n_systems": len(systems), "systems": systems,
           "excluded": [s for s in subs if s not in corpora],
           "n_failed": {s: int(corpora[s][1].sum()) for s in systems},
           "n_runs": {s: int(len(corpora[s][1])) for s in systems},
           "length_cross_tau0.1": r["length_cross_tau0.1"]}
    for t in swe.TAUS:
        cross = float(np.mean([v for p in pairs for v in r["per_pair"][t][p]]))
        within = float(np.mean(r["within"][t]))
        null_t = [n[t] for n in nulls]
        out[f"tau={t}"] = {
            "cross": cross, "within": within, "excess": cross - within,
            "perm_null_p95": float(np.percentile(null_t, 95)),
            "perm_p": float((1 + sum(n >= cross for n in null_t)) / (1 + swe.N_PERM)),
            "system_bootstrap_ci95_cross": swe._system_bootstrap(r["per_pair"][t], systems, rng),
        }
    ex = out["tau=0.1"]["excess"]
    out["verdict"] = ("agentdojo_like" if ex >= 0.12 else "swebench_like" if ex <= 0.05
                      else "inconclusive")
    Path(f"output/tables/v50_tau2_cross_{domain}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "systems"}, indent=1))


if __name__ == "__main__":
    main()
