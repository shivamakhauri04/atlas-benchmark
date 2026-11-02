"""Holm correction across the registered family of confirmatory tests (prereg-v50 §4, A7).

Family: primary sign test on the executed and external strata (annotator consensus), and E1's
cross-configuration flip test at the registered threshold tau = 0.1 (2,000-shuffle null).
"""

import json
from pathlib import Path

from scipy.stats import binomtest

T = Path("output/tables")
OUT = T / "v50_holm.json"


def main() -> None:
    cons = json.loads((T / "v50_decomposition_consensus.json").read_text())["tau=0.1"]
    e1 = json.loads((T / "v50_cross_agent_thresholds.json").read_text())["tau=0.1"]
    raw = {
        "primary_executed": cons["i_executed"]["binom_p_vs_baseline_plus_10"],
        # Reason: the registered external condition (prereg A1/A2) is accuracy > 60%, not the
        # +10-over-baseline rule used for the executed stratum.
        "primary_external": float(binomtest(sum(c["hit"] for c in cons["ii_external"]["cells"]),
                                            cons["ii_external"]["n_cells"], 0.6,
                                            alternative="greater").pvalue),
        "e1_cross_config_flip": e1["perm_p"],
    }
    order = sorted(raw, key=raw.get)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * raw[k]))
        adj[k] = running
    OUT.write_text(json.dumps({"raw": raw, "holm": adj, "alpha": 0.05,
                               "reject": {k: v < 0.05 for k, v in adj.items()}}, indent=2))
    print(json.dumps({"raw": raw, "holm": adj}, indent=1))


if __name__ == "__main__":
    main()
