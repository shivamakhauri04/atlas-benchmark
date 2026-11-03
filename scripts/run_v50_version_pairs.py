"""Version-update pairs within AgentDojo (prereg-v50 A12, exploratory, registered before running).

E1's design on the registered successor pairs: old configuration's task half A against the new
configuration's task half B, ten splits, class-balanced, five length-free features.
"""

import itertools
import json
from pathlib import Path

import numpy as np

import run_v50_cross_agent as e1
import run_v50_cross_agent_attacks as atk
from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_version_pairs.json")
TAUS = (0.1, 0.2, 0.3)
STRICT = (
    ("gemini-1.5-pro-001", "gemini-1.5-pro-002"),
    ("gpt-4-0125-preview", "gpt-4-turbo-2024-04-09"),
    ("gemini-2.0-flash-exp", "gemini-2.0-flash-001"),
)
SUCCESSOR = STRICT + (
    ("gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13"),
    ("claude-3-5-sonnet-20240620", "claude-3-7-sonnet-20250219"),
    ("claude-3-sonnet-20240229", "claude-3-5-sonnet-20240620"),
    ("meta-llama_Llama-3-70b-chat-hf", "meta-llama_Llama-3.3-70B-Instruct"),
    ("gemini-1.5-flash-001", "gemini-2.0-flash-001"),
    ("gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18"),
)
WITHIN_REGISTERED = 0.0582


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir())
               if (c := atk._corpus(r, "important_instructions"))}
    missing = [c for p in SUCCESSOR for c in p if c not in corpora]
    assert not missing, missing
    succ = set(SUCCESSOR)
    flips = {name: {t: [] for t in TAUS} for name in ("strict", "successor", "other_models")}
    for seed in e1.SEEDS:
        rng = np.random.default_rng(seed)
        halves = {}
        for r, (_, y, tasks) in corpora.items():
            h = e1._task_half(tasks, seed)
            halves[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
        d = {(r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])[:5]
             for r in corpora for s in (0, 1)}
        for a, b in itertools.combinations(corpora, 2):
            same_base = atk._base(a) == atk._base(b)
            key = ("strict" if (a, b) in STRICT or (b, a) in STRICT else
                   "successor" if (a, b) in succ or (b, a) in succ else
                   None if same_base else "other_models")
            if key is None:
                continue
            old, new = (a, b) if (a, b) in succ or key == "other_models" else (b, a)
            for t in TAUS:
                fl = [is_reversed(x, z, tau=t) for x, z in zip(d[(old, 0)], d[(new, 1)])]
                flips[key][t] += fl
                if key == "strict":
                    flips["successor"][t] += fl
    out = {"within_registered_tau0.1": WITHIN_REGISTERED,
           "n_pairs": {"strict": len(STRICT), "successor": len(SUCCESSOR)}}
    for name, by_t in flips.items():
        out[name] = {f"tau={t}": float(np.mean(v)) for t, v in by_t.items()}
    out["refuted"] = out["successor"]["tau=0.1"] <= WITHIN_REGISTERED
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
