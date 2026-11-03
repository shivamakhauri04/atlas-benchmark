"""E1 on a second benchmark: mini-swe-agent systems on SWE-bench Verified (prereg-v50 A13/A13a).

Usage: python scripts/run_v50_swebench_cross.py <cohort-list> <name>

Label 1 = instance not resolved. Same design as E1: task-disjoint halves (crc32 of seed:instance),
class-balanced, ten splits, five length-free features, tau 0.1 primary. Adds the ten-split
permutation null (runs shuffled between the two systems of a pair) and a bootstrap over systems.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1

from data import domain_loader as dl
from data.schema import Action, Trace
from traceshift.v50 import is_reversed

DATA = Path("data/swebench_mini")
TAUS = (0.1, 0.2, 0.3)
N_PERM = 1000
N_BOOT = 2000


def _corpus(entry: str):
    labels = json.loads((DATA / "labels" / f"{entry}.json").read_text())
    rows = [json.loads(line) for line in (DATA / f"{entry}.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["instance_id"] in labels]
    # Reason: E1's registered run filter keeps runs with at least one tool call.
    rows = [r for r in rows if r["actions"]]
    traces = [
        Trace(
            trace_id=r["instance_id"],
            task=r["instance_id"],
            source="swebench",
            actions=[Action(tool=a["tool"], args=a["args"]) for a in r["actions"]],
        )
        for r in rows
    ]
    y = np.array([0 if labels[r["instance_id"]].get("resolved") else 1 for r in rows])
    if min(int(y.sum()), int(len(y) - y.sum())) < e1.MIN_CLASS:
        return None
    feats = dl._extract_all_features(traces)
    x = np.column_stack([np.asarray(feats[f], float) for f in (*e1.FEATURES, "trace_length")])
    return x, y, [t.task for t in traces]


def _halves(corpora: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for r, (_, y, tasks) in corpora.items():
        h = e1._task_half(tasks, seed)
        out[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
    return out


def _rates(corpora: dict, pairs: list, splits: list) -> dict:
    within = {t: [] for t in TAUS}
    per_pair = {t: {p: [] for p in pairs} for t in TAUS}
    length = []
    for halves in splits:
        d = {
            (r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])
            for r in corpora
            for s in (0, 1)
        }
        for t in TAUS:
            within[t] += [
                is_reversed(a, b, tau=t)
                for r in corpora
                for a, b in zip(d[(r, 0)][:5], d[(r, 1)][:5])
            ]
            for p in pairs:
                per_pair[t][p] += [
                    is_reversed(a, b, tau=t) for a, b in zip(d[(p[0], 0)][:5], d[(p[1], 1)][:5])
                ]
        length += [is_reversed(d[(p[0], 0)][5], d[(p[1], 1)][5], tau=0.1) for p in pairs]
    return {"within": within, "per_pair": per_pair, "length_cross_tau0.1": float(np.mean(length))}


def _null(corpora: dict, pairs: list, splits: list, rng: np.random.Generator) -> dict:
    flips = {t: [] for t in TAUS}
    for halves in splits:
        for r1, r2 in pairs:
            (x1, y1, _), (x2, y2, _) = corpora[r1], corpora[r2]
            a, b = halves[r1][0], halves[r2][1]
            xp, yp = np.vstack([x1[a], x2[b]]), np.concatenate([y1[a], y2[b]])
            perm = rng.permutation(len(yp))
            da, db = e1._d(xp, yp, perm[: len(a)])[:5], e1._d(xp, yp, perm[len(a) :])[:5]
            for t in TAUS:
                flips[t] += [is_reversed(p, q, tau=t) for p, q in zip(da, db)]
    return {t: float(np.mean(v)) for t, v in flips.items()}


def _system_bootstrap(per_pair: dict, systems: list, rng: np.random.Generator) -> list[float]:
    idx = {s: i for i, s in enumerate(systems)}
    pairs = list(per_pair)
    rates = np.array([np.mean(per_pair[p]) for p in pairs])
    si = np.array([idx[a] for a, _ in pairs])
    ti = np.array([idx[b] for _, b in pairs])
    boots = []
    for _ in range(N_BOOT):
        c = np.bincount(rng.integers(0, len(systems), len(systems)), minlength=len(systems))
        w = c[si] * c[ti]
        if w.sum():
            boots.append(float(np.average(rates, weights=w)))
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main() -> None:
    entries = Path(sys.argv[1]).read_text().split()
    name = sys.argv[2]
    corpora = {e: c for e in entries if (c := _corpus(e))}
    systems = list(corpora)
    pairs = list(itertools.combinations(systems, 2))
    splits = [_halves(corpora, s) for s in e1.SEEDS]
    r = _rates(corpora, pairs, splits)
    rng = np.random.default_rng(0)
    nulls = [_null(corpora, pairs, splits, rng) for _ in range(N_PERM)]
    out = {
        "cohort": name,
        "n_systems": len(systems),
        "systems": systems,
        "n_unresolved": {e: int(corpora[e][1].sum()) for e in systems},
        "length_cross_tau0.1": r["length_cross_tau0.1"],
    }
    for t in TAUS:
        cross = float(np.mean([v for p in pairs for v in r["per_pair"][t][p]]))
        null_t = [n[t] for n in nulls]
        out[f"tau={t}"] = {
            "cross": cross,
            "within": float(np.mean(r["within"][t])),
            "perm_null_mean": float(np.mean(null_t)),
            "perm_null_p95": float(np.percentile(null_t, 95)),
            "perm_p": float((1 + sum(n >= cross for n in null_t)) / (1 + N_PERM)),
            "system_bootstrap_ci95": _system_bootstrap(r["per_pair"][t], systems, rng),
        }
    t1 = out["tau=0.1"]
    out["refuted"] = bool(t1["cross"] <= t1["within"] or t1["cross"] <= t1["perm_null_p95"])
    Path(f"output/tables/v50_swebench_cross_{name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "systems"}, indent=1))


if __name__ == "__main__":
    main()
