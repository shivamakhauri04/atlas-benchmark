"""Capability spread as the source of AgentDojo's reversals (prereg-v50 A18, registered first).

Configurations are ranked by user-task success on their attacked runs (metadata only). The top 11
are "strong". E1's ten-split design is rerun, and pairs are split into strong-strong, weak-weak,
and mixed. Each group's excess is cross minus the mean within rate of the configurations it
involves, with a bootstrap over configurations.
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

OUT = Path("output/tables/v50_capability_split.json")
TAU = 0.1
N_BOOT = 2000


def _capability(run: str) -> float:
    traces = ul.load_agentdojo(e1.RUNS / run, min_tool_calls=1, attack="important_instructions")
    util = ul._utility(e1.RUNS / run, "important_instructions")
    return float(np.mean([util[t.trace_id] for t in traces if t.trace_id in util]))


def main() -> None:
    runs = sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir())
    corpora = {r: c for r in runs if (c := atk._corpus(r, "important_instructions"))}
    cap = {r: _capability(r) for r in corpora}
    ranked = sorted(corpora, key=lambda r: (-cap[r], r))
    strong = set(ranked[:11])
    splits = [tp._halves(corpora, s) for s in e1.SEEDS]
    within = {r: [] for r in corpora}
    pairs = list(itertools.combinations(corpora, 2))
    cross = {p: [] for p in pairs}
    for halves in splits:
        d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
             for r in corpora for s in (0, 1)}
        for r in corpora:
            within[r] += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(r, 0)], d[(r, 1)])]
        for p in pairs:
            cross[p] += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])]
    w = {r: float(np.mean(v)) for r, v in within.items()}
    c = {p: float(np.mean(v)) for p, v in cross.items()}
    assert abs(np.mean(list(c.values())) - 0.2248) < 5e-4
    assert abs(np.mean(list(w.values())) - 0.0582) < 5e-4

    def group(p):
        n = (p[0] in strong) + (p[1] in strong)
        return {2: "strong_strong", 0: "weak_weak", 1: "mixed"}[n]

    configs = list(corpora)
    rng = np.random.default_rng(0)
    out = {"tau": TAU, "capability": cap, "strong": sorted(strong)}
    for g in ("strong_strong", "weak_weak", "mixed"):
        gp = [p for p in pairs if group(p) == g]
        involved = sorted({x for p in gp for x in p})

        def excess(counts: dict) -> float:
            pw = np.array([counts[a] * counts[b] for a, b in gp])
            cw = np.array([counts[r] for r in involved])
            if pw.sum() == 0 or cw.sum() == 0:
                return float("nan")
            return float(np.average([c[p] for p in gp], weights=pw)
                         - np.average([w[r] for r in involved], weights=cw))

        point = excess({r: 1 for r in configs})
        boots = []
        for _ in range(N_BOOT):
            k = np.bincount(rng.integers(0, len(configs), len(configs)), minlength=len(configs))
            boots.append(excess(dict(zip(configs, k))))
        boots = np.array(boots)[~np.isnan(boots)]
        out[g] = {"n_pairs": len(gp), "cross": float(np.mean([c[p] for p in gp])),
                  "within": float(np.mean([w[r] for r in involved])), "excess": point,
                  "excess_ci95": [float(np.percentile(boots, 2.5)),
                                  float(np.percentile(boots, 97.5))]}
    ss = out["strong_strong"]["excess"]
    out["verdict"] = ("refuted" if ss >= 0.12 else "supported" if ss <= 0.05 else "inconclusive")
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("capability",)}, indent=1))


if __name__ == "__main__":
    main()
