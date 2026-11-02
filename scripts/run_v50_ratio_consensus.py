"""Secondary reversal ratio under the annotator-consensus provenance (prereg-v50 A6, primary arm).

A cell is tied when annotators A and B give the same non-zero sign. The strict arm additionally
requires both to name the definition rule (A: R1, B: definition).
"""

import csv
import json
from pathlib import Path

import numpy as np

import run_v50_grid as grid

NOTES = Path("inputs/annotators")
OUT = Path("output/tables/v50_ratio_consensus.json")


def _rows(name: str) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["dataset"].strip().lower(), r["feature"].strip()): r
            for r in csv.DictReader((NOTES / f"annotator_{name}.csv").open())}


def consensus(strict: bool) -> dict[tuple[str, str], int]:
    a, b = _rows("A"), _rows("B")
    out: dict[tuple[str, str], int] = {}
    for key, ra in a.items():
        if key[0] not in grid.PRIMARY or key[0] in grid.EXCLUDED:
            continue
        rb = b[key]
        sa, sb = int(float(ra["predicted_sign"])), int(float(rb["predicted_sign"]))
        tied = sa == sb != 0
        if strict:
            tied = tied and ra["rule"].strip() == "R1" and rb["entanglement"].strip() == "definition"
        out[key] = int(tied)
    return out


def main() -> None:
    r = json.loads(grid.OUT.read_text())
    d = {k: np.array([v[f] for f in grid.FEATURES]) for k, v in r["d_half_a"].items()}
    rng = np.random.default_rng(0)
    result = {
        f"tau={t}/{arm}": grid.reversal_ratio(d, consensus(arm == "annotators_strict_R1"), t, rng)
        for t in grid.TAUS for arm in ("annotators_lenient_R1-R3", "annotators_strict_R1")
    }
    result["n_tied_cells"] = {arm: sum(consensus(arm == "annotators_strict_R1").values())
                              for arm in ("annotators_lenient_R1-R3", "annotators_strict_R1")}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
