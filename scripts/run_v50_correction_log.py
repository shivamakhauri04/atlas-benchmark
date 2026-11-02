"""Recompute the earlier version's (v49) headline numbers under the v50 corrections.

v49 headline: within-group reversal 6.7% vs between-group 26.8% (4.0x, p = 0.020), on 9 domains,
3 structural features, 4-group taxonomy (`scripts/run_taxonomy_permutation.py:33-38`).
Here the same statistic uses v50 d: corrected AgentDojo labels, stratified half A, mean of 10 seeds.
It is also recomputed with construction-tied cells removed, under each provenance classification.
That gives the Role 1 consequence registered in the proposal (§4).
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

import run_v49_reanalysis as v49
import run_v50_ratio_consensus as cons
import run_v50_grid as grid

OUT = Path("output/tables/v50_correction_log.json")
V49_GROUPS = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["agentdojo", "agentharm", "atbench", "injecagent", "rjudge"],
    "api_misuse": ["toolbench"],
}
STRUCT = v49.STRUCTURAL


def gap(domains: list[str], d: dict[str, dict[str, float]], groups: dict[str, list[str]],
        drop: set[tuple[str, str]], n_perm: int = 20000, thr: float = 0.1) -> dict[str, float]:
    """Within- vs between-group reversal rate, skipping (dataset, feature) cells in `drop`."""
    def rates(assign: dict[str, str]) -> tuple[float, float]:
        w = wt = b = bt = 0
        for x, y in combinations(domains, 2):
            for f in STRUCT:
                if (x, f) in drop or (y, f) in drop:
                    continue
                r = v49._rev(d[x][f], d[y][f], thr)
                if assign[x] == assign[y]:
                    wt, w = wt + 1, w + r
                else:
                    bt, b = bt + 1, b + r
        return (w / wt if wt else 0.0), (b / bt if bt else 0.0)

    groups = {g: [x for x in xs if x in domains] for g, xs in groups.items()}
    groups = {g: xs for g, xs in groups.items() if xs}
    assign = {x: g for g, xs in groups.items() for x in xs}
    within, between = rates(assign)
    rng = np.random.RandomState(0)
    sizes = [len(v) for v in groups.values()]
    count = 0
    for _ in range(n_perm):
        perm = list(domains)
        rng.shuffle(perm)
        it = iter(perm)
        a = {next(it): g for g, sz in zip(groups, sizes) for _ in range(sz)}
        w, b = rates(a)
        count += (b - w) >= (between - within)
    return {"within": within, "between": between,
            "ratio": between / within if within else None, "p_value": (count + 1) / (n_perm + 1)}


def main() -> None:
    d = json.loads(grid.OUT.read_text())["d_half_a"]
    v49_domains = [x for g in V49_GROUPS.values() for x in g]
    tied_cons = {k for k, v in cons.consensus(strict=False).items() if v}
    tied_auth = {k for k, v in grid.provenance({"R1", "R2", "R3"}).items() if v}
    result = {
        "v49_reported": {"within": 0.067, "between": 0.268, "ratio": 4.0, "p_value": 0.020,
                         "domains": 9},
        "v50_same_9_domains": gap(v49_domains, d, V49_GROUPS, set()),
        "v50_same_9_drop_tied_consensus": gap(v49_domains, d, V49_GROUPS, tied_cons),
        "v50_same_9_drop_tied_authors": gap(v49_domains, d, V49_GROUPS, tied_auth),
        "v50_12_domains_groups4": gap(list(grid.PRIMARY), d, v49.GROUPS_4, set()),
    }
    rows = json.loads(grid.OUT.read_text())["grid"]
    a = np.array([r["auroc_mean"] for r in rows if r["features"] == "all" and r["detector"] == "rf"])
    result["anti_prediction_rate_rf"] = {"v49_reported": 0.328, "v50": float((a < 0.5).mean()),
                                         "v50_n_pairs": int(len(a))}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
