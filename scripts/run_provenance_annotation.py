"""Fleiss' kappa between the authors' provenance classification and two blind annotators.

Raters: authors (`prereg-v50-sign-predictions.csv`, primary rows), annotator A, annotator B
(`notes/annotators/`). Each rater gives a predicted sign in {-1, 0, +1} per (dataset, feature) cell;
kappa is reported on that three-way sign and on the binary tied-vs-free split. Per prereg-v50 §2,
kappa < 0.6 means the headline uses the annotators' majority classification.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

NOTES = Path("inputs")
OUT = Path("output/tables/v50_provenance.json")
PRIMARY_SUB = {"agentdojo": "gpt-4o-2024-05-13", "stabletoolbench": "chatgpt_cot", "trail": "all"}

Cell = tuple[str, str]


def _authors() -> dict[Cell, int]:
    out: dict[Cell, int] = {}
    for r in csv.DictReader((NOTES / "prereg-v50-sign-predictions.csv").open()):
        if r["stratum"] in ("ii_external", "unavailable"):
            continue
        if r["sub_corpus"] not in ("", PRIMARY_SUB.get(r["dataset"], "")):
            continue
        out[(r["dataset"], r["feature"])] = int(r["predicted_sign"])
    return out


def _annotator(name: str) -> dict[Cell, int]:
    rows = csv.DictReader((NOTES / "annotators" / f"annotator_{name}.csv").open())
    return {(r["dataset"].strip().lower(), r["feature"].strip()): int(float(r["predicted_sign"]))
            for r in rows}


def fleiss_kappa(ratings: list[list[int]], categories: list[int]) -> float:
    """Fleiss' kappa for items rated by the same number of raters."""
    counts = np.array([[Counter(item)[c] for c in categories] for item in ratings], dtype=float)
    n = counts.sum(axis=1)[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j**2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def main() -> None:
    raters = {"authors": _authors(), "A": _annotator("A"), "B": _annotator("B")}
    cells = sorted(set.intersection(*(set(r) for r in raters.values())))
    missing = {k: sorted(set(raters["authors"]) - set(v)) for k, v in raters.items()}
    signs = [[raters[r][c] for r in raters] for c in cells]
    tied = [[int(v != 0) for v in row] for row in signs]
    majority_ab = {f"{c[0]}/{c[1]}": Counter([raters["A"][c], raters["B"][c]]).most_common(1)[0][0]
                   if raters["A"][c] == raters["B"][c] else 0 for c in cells}
    disagreements = [{"cell": f"{c[0]}/{c[1]}", **{r: raters[r][c] for r in raters}}
                     for c in cells if len({raters[r][c] for r in raters}) > 1]
    k_sign = fleiss_kappa(signs, [-1, 0, 1])
    result = {
        "n_cells": len(cells),
        "missing_cells": missing,
        "fleiss_kappa_sign": k_sign,
        "fleiss_kappa_tied": fleiss_kappa(tied, [0, 1]),
        "headline_uses": "authors" if k_sign >= 0.6 else "annotator_majority",
        "n_tied": {r: sum(v[c] != 0 for c in cells) for r, v in raters.items()},
        "annotator_consensus_sign": majority_ab,
        "disagreements": disagreements,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("annotator_consensus_sign",)},
                     indent=1))


if __name__ == "__main__":
    main()
