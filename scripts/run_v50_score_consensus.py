"""Score sign predictions under the annotator-consensus classification (prereg-v50 A6).

Fleiss' kappa was 0.33 < 0.6, so the headline uses cells where annotators A and B agree on a
non-zero sign. The consensus sign of a (dataset, feature) applies to every sub-corpus of that
dataset. The external AgentDrift stratum keeps the authors' predictions, because the annotators
classified only the 12 datasets of the earlier benchmark.
"""

import csv
import json
from pathlib import Path

import run_v50_decomposition as dec

OUT = Path("output/tables/v50_decomposition_consensus.json")


def main() -> None:
    decomp = json.loads(dec.OUT.read_text())
    consensus = json.loads(Path("output/tables/v50_provenance.json").read_text())[
        "annotator_consensus_sign"
    ]
    d = {tuple(k.split("/", 1)): v for k, v in decomp["d_half_a"].items()}
    d = {(k[0], k[1] if len(k) > 1 else ""): v for k, v in d.items()}
    preds = []
    for r in csv.DictReader(dec.PRED.open()):
        row = dict(r)
        if r["stratum"] in ("i_executed", "iii_injected"):
            row["predicted_sign"] = str(consensus.get(f"{r['dataset']}/{r['feature']}", 0))
        preds.append(row)
    result = {
        f"tau={t}": {
            s: dec.score(preds, d, s, t) for s in ("i_executed", "ii_external", "iii_injected")
        }
        for t in dec.TAUS
    }
    OUT.write_text(json.dumps(result, indent=2))
    summary = {
        k: {s: {x: v[s].get(x) for x in ("n_cells", "accuracy", "pooled_sign_baseline", "refuted")}
            for s in v}
        for k, v in result.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
