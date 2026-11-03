"""Configuration-bootstrap intervals for A16's cross-minus-within excess (exploratory, prereg A17).

For each label (security, utility) on the same attacked AgentDojo runs, the per-configuration
within rate and the per-pair cross rate are averaged over ten task splits. A bootstrap over
configurations gives an interval for the excess (cross minus within), and for the difference of
excesses between the two labels on the configurations both labels admit.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import run_v50_cross_agent_attacks as atk
import run_v50_cross_agent_tensplit_perm as tp
import run_v50_utility_label as ul

from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_utility_label_ci.json")
N_BOOT = 2000
TAU = 0.1


def _rates(corpora: dict) -> tuple[dict, dict]:
    pairs = list(itertools.combinations(corpora, 2))
    splits = [tp._halves(corpora, s) for s in e1.SEEDS]
    within = {r: [] for r in corpora}
    cross = {p: [] for p in pairs}
    for halves in splits:
        d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
             for r in corpora for s in (0, 1)}
        for r in corpora:
            within[r] += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(r, 0)], d[(r, 1)])]
        for p in pairs:
            cross[p] += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])]
    return ({r: float(np.mean(v)) for r, v in within.items()},
            {p: float(np.mean(v)) for p, v in cross.items()})


def _excess(within: dict, cross: dict, configs: list[str], counts: np.ndarray) -> float:
    w = {c: n for c, n in zip(configs, counts)}
    pw = np.array([w[a] * w[b] for a, b in cross])
    cw = np.array([w[c] for c in within])
    if pw.sum() == 0 or cw.sum() == 0:
        return float("nan")
    return float(np.average(list(cross.values()), weights=pw)
                 - np.average(list(within.values()), weights=cw))


def main() -> None:
    runs = sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir())
    sec = {r: c for r in runs if (c := atk._corpus(r, "important_instructions"))}
    sec = {r: (c[0][:, :5], c[1], c[2]) for r, c in sec.items()}
    util = {r: c for r in runs if (c := ul._corpus(r, "important_instructions"))}
    common = sorted(set(sec) & set(util))
    out = {"tau": TAU, "n_common_configurations": len(common)}
    rng = np.random.default_rng(0)
    res = {}
    for name, corpora in (("security", sec), ("utility", util)):
        sub = {r: corpora[r] for r in common}
        res[name] = _rates(sub)
    point = {n: _excess(*res[n], common, np.ones(len(common))) for n in res}
    boots = {"security": [], "utility": [], "utility_minus_security": []}
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, len(common), len(common)), minlength=len(common))
        es, eu = (_excess(*res[n], common, counts) for n in ("security", "utility"))
        boots["security"].append(es)
        boots["utility"].append(eu)
        boots["utility_minus_security"].append(eu - es)
    for k, v in boots.items():
        v = np.array(v)[~np.isnan(v)]
        pt = point["utility"] - point["security"] if k == "utility_minus_security" else point[k]
        ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out[k] = {"point": pt, "ci95": ci}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
