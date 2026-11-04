"""Three checks requested by a blind review (exploratory, prereg A20, after the results).

(a) AgentDojo: pairs among the four most recently released configurations (Llama 3.3 and Gemini 2.0
    Flash exp, December 2024; Gemini 2.0 Flash 001 and Claude 3.7 Sonnet, February 2025) against all
    other pairs, E1's ten-split design at tau 0.1.
(b) tau2-bench retail: pairs whose two runs share a tau2 git commit against pairs across commits.
(c) tau2-bench retail: the excess over random configuration orders, which change the balancing draw.
"""

import itertools
import json
import random
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import run_v50_cross_agent_attacks as atk
import run_v50_swebench_cross as swe
import run_v50_tau2_cross as t2

from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_generation_harness_checks.json")
TAU = 0.1
LATEST = {
    "meta-llama_Llama-3.3-70B-Instruct",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-001",
    "claude-3-7-sonnet-20250219",
}
COMMIT = {
    "claude-opus-4-5_sierra_2026-02-26": "7324fb57",
    "claude-sonnet-4-5_sierra_2026-02-26": "7324fb57",
    "gpt-5-2-none_sierra_2026-02-26": "7324fb57",
    "gpt-5-2_sierra_2026-02-26": "7324fb57",
    "gemini-3-flash_sierra_2026-03-02": "ef9454a8",
    "gemini-3-pro_sierra_2026-03-02": "ef9454a8",
    "glm-5-think_sierra_2026-03-02": "unknown",
    "qwen3.5-397b-a17b-think_sierra_2026-03-02": "unknown",
}
N_ORDERS = 20


def _pair_and_within(corpora: dict, splits: list) -> tuple[dict, dict]:
    pairs = list(itertools.combinations(corpora, 2))
    cross = {p: [] for p in pairs}
    within = {r: [] for r in corpora}
    for halves in splits:
        d = {
            (r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
            for r in corpora
            for s in (0, 1)
        }
        for r in corpora:
            within[r] += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(r, 0)], d[(r, 1)])]
        for p in pairs:
            cross[p] += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])]
    return (
        {p: float(np.mean(v)) for p, v in cross.items()},
        {r: float(np.mean(v)) for r, v in within.items()},
    )


def _group(cross: dict, within: dict, keep) -> dict:
    gp = [p for p in cross if keep(p)]
    inv = sorted({x for p in gp for x in p})
    c = float(np.mean([cross[p] for p in gp]))
    w = float(np.mean([within[r] for r in inv]))
    return {"n_pairs": len(gp), "cross": c, "within": w, "excess": c - w}


def _order_spread(corpora: dict) -> dict:
    excess = []
    for k in range(N_ORDERS):
        names = list(corpora)
        random.Random(k).shuffle(names)
        shuffled = {n: corpora[n] for n in names}
        cross, within = _pair_and_within(shuffled, [swe._halves(shuffled, s) for s in e1.SEEDS])
        excess.append(_group(cross, within, lambda p: True)["excess"])
    return {
        "n_orders": N_ORDERS,
        "min": min(excess),
        "max": max(excess),
        "mean": float(np.mean(excess)),
        "sd": float(np.std(excess)),
    }


def main() -> None:
    runs = sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir())
    ad = {r: c for r in runs if (c := atk._corpus(r, "important_instructions"))}
    cross, within = _pair_and_within(ad, [swe._halves(ad, s) for s in e1.SEEDS])
    out = {
        "tau": TAU,
        "agentdojo": {
            "latest_latest": _group(cross, within, lambda p: p[0] in LATEST and p[1] in LATEST),
            "pairs_with_claude_3_7": _group(
                cross, within, lambda p: "claude-3-7-sonnet-20250219" in p
            ),
            "other_pairs": _group(cross, within, lambda p: not (p[0] in LATEST and p[1] in LATEST)),
        },
    }
    # Reason: the balancing draw depends on configuration order, so use the registered order.
    subs = Path("inputs/a19_tau2.txt").read_text().split()
    tau = {s: c for s in subs if (c := t2._corpus(s, "retail"))}
    cross, within = _pair_and_within(tau, [swe._halves(tau, s) for s in e1.SEEDS])
    known = {s for s in tau if COMMIT[s] != "unknown"}
    out["tau2_retail"] = {
        "same_commit": _group(
            cross, within, lambda p: p[0] in known and COMMIT[p[0]] == COMMIT[p[1]]
        ),
        "different_commit": _group(
            cross,
            within,
            lambda p: p[0] in known and p[1] in known and COMMIT[p[0]] != COMMIT[p[1]],
        ),
        "all": _group(cross, within, lambda p: True),
        "excess_over_orders": _order_spread(tau),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
