"""Configuration-level bootstrap of E2b regret differences on AgentDojo.

Paper review A1; exploratory, prereg A9.

Pairs share configurations, so the bootstrap resamples the 22 configurations with replacement and
weights each ordered pair (s, t) by count(s) * count(t). Reported per feature set, detector, and k:
the
mean regret of the held-out distance gate minus each comparator, with a 95% percentile interval.
"""

import json
import sys
from pathlib import Path

import numpy as np

NAME = sys.argv[1] if len(sys.argv) > 1 else "agentdojo"
SRC = Path(f"output/tables/v50_reuse_{NAME}.json")
OUT = Path(
    "output/tables/v50_reuse_bootstrap.json"
    if NAME == "agentdojo"
    else f"output/tables/v50_reuse_bootstrap_{NAME}.json"
)
N_BOOT = 2000
GATE = "divergence_gate_heldout"
COMPARATORS = ("sign_gate", "always_reuse", "always_retrain", "divergence_gate")


def _diffs(cell: dict, configs: list[str], rng: np.random.Generator) -> dict:
    pairs = [k.split("|") for k in cell]
    reg = {p: np.array([cell[k][p] for k in cell]) for p in (GATE, *COMPARATORS)}
    idx = {c: i for i, c in enumerate(configs)}
    s_idx = np.array([idx[s] for s, _ in pairs])
    t_idx = np.array([idx[t] for _, t in pairs])
    boot = {c: [] for c in (*COMPARATORS, "best_fixed")}
    reuse_vs_retrain = []
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, len(configs), len(configs)), minlength=len(configs))
        w = counts[s_idx] * counts[t_idx]
        if w.sum() == 0:
            continue
        mean = {p: float(np.average(v, weights=w)) for p, v in reg.items()}
        for c in COMPARATORS:
            boot[c].append(mean[GATE] - mean[c])
        boot["best_fixed"].append(mean[GATE] - min(mean["always_reuse"], mean["always_retrain"]))
        reuse_vs_retrain.append(mean["always_reuse"] - mean["always_retrain"])
    point = {p: float(v.mean()) for p, v in reg.items()}
    point_best = min(point["always_reuse"], point["always_retrain"])
    out = {"point_gate_regret": point[GATE]}
    for c, vals in boot.items():
        ref = point_best if c == "best_fixed" else point[c]
        out[f"minus_{c}"] = {
            "point": point[GATE] - ref,
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        }
    out["reuse_minus_retrain"] = {
        "point": point["always_reuse"] - point["always_retrain"],
        "ci95": [float(np.percentile(reuse_vs_retrain, 2.5)),
                 float(np.percentile(reuse_vs_retrain, 97.5))],
    }
    return out


def main() -> None:
    data = json.loads(SRC.read_text())
    rng = np.random.default_rng(0)
    out = {}
    for key, cell in data.items():
        if not isinstance(cell, dict) or "per_pair_regret" not in cell:
            continue
        ppr = cell["per_pair_regret"]
        configs = sorted({c for k in ppr for c in k.split("|")})
        out[key] = _diffs(ppr, configs, rng)
    OUT.write_text(json.dumps(out, indent=2))
    for key, v in out.items():
        print(
            key,
            {
                k: (round(x["point"], 4), [round(y, 4) for y in x["ci95"]])
                for k, x in v.items()
                if k.startswith("minus_")
            },
        )


if __name__ == "__main__":
    main()
