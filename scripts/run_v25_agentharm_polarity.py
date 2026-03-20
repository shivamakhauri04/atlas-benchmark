"""v25: AgentHarm polarity check — predictive validation of feature polarity reversal.

AgentHarm has paired benign/harmful traces using the SAME target tool sequences.
If polarity reversal depends on dataset construction (injected vs emergent anomalies),
then AgentHarm should NOT show trace_length inversion (same tools → similar lengths).

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_v25_agentharm_polarity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from data.trace_loader import load_agentharm

OUTPUT_DIR = Path("output/tables")


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def d_magnitude(d: float) -> str:
    if np.isnan(d):
        return "undefined"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AgentHarm traces...")
    raw_traces = load_agentharm()
    print(f"  Loaded {len(raw_traces)} traces")

    traces = []
    for t in raw_traces:
        actions = [a.tool for a in t.actions]
        label = 1 if t.has_anomaly else 0
        traces.append({"actions": actions, "label": label})

    n = len(traces)
    lengths = np.zeros(n)
    diversities = np.zeros(n)
    repeat_rates = np.zeros(n)
    labels = np.zeros(n, dtype=np.int32)

    for i, t in enumerate(traces):
        actions = t["actions"]
        na = len(actions)
        lengths[i] = float(na)
        unique = len(set(actions))
        diversities[i] = 1.0 - (unique / max(na, 1))
        if na > 1:
            reps = sum(1 for j in range(1, na) if actions[j] == actions[j - 1])
            repeat_rates[i] = reps / (na - 1)
        labels[i] = t["label"]

    normal_mask = labels == 0
    anomalous_mask = labels == 1
    n_normal = int(normal_mask.sum())
    n_anom = int(anomalous_mask.sum())
    print(f"  Normal: {n_normal}, Harmful: {n_anom}")

    result = {
        "domain": "agentharm",
        "n_normal": n_normal,
        "n_anomalous": n_anom,
        "description": "Paired benign/harmful traces using same target tool sequences",
        "prediction": "trace_length should NOT invert (same tools -> similar lengths)",
        "properties": {},
    }

    print("\n--- AgentHarm Feature Polarity Check ---")
    print(f"{'Feature':<18} {'Normal mean':>12} {'Harmful mean':>13} {'Cohen d':>10} {'Magnitude':>12} {'KS p':>12}")
    print("-" * 80)

    for prop_name, values in [
        ("trace_length", lengths),
        ("low_diversity", diversities),
        ("repeat_rate", repeat_rates),
    ]:
        normal_vals = values[normal_mask]
        anom_vals = values[anomalous_mask]
        d = cohens_d(anom_vals, normal_vals)
        ks_stat, ks_p = stats.ks_2samp(normal_vals, anom_vals)

        result["properties"][prop_name] = {
            "normal_mean": float(normal_vals.mean()),
            "normal_std": float(normal_vals.std()),
            "anomalous_mean": float(anom_vals.mean()),
            "anomalous_std": float(anom_vals.std()),
            "cohens_d": d,
            "cohens_d_magnitude": d_magnitude(d),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
        }

        print(
            f"{prop_name:<18} {normal_vals.mean():>12.3f} {anom_vals.mean():>13.3f} "
            f"{d:>+10.3f} {d_magnitude(d):>12} {ks_p:>12.2e}"
        )

    # Vocabulary analysis
    all_tools_normal = []
    all_tools_harmful = []
    for t in traces:
        if t["label"] == 0:
            all_tools_normal.extend(t["actions"])
        else:
            all_tools_harmful.extend(t["actions"])

    vocab_normal = set(all_tools_normal)
    vocab_harmful = set(all_tools_harmful)
    overlap = vocab_normal & vocab_harmful
    result["vocabulary"] = {
        "unique_normal": len(vocab_normal),
        "unique_harmful": len(vocab_harmful),
        "overlap": len(overlap),
        "overlap_pct": len(overlap) / len(vocab_normal | vocab_harmful) * 100,
    }
    print(f"\nVocabulary: {len(vocab_normal)} normal tools, {len(vocab_harmful)} harmful tools, "
          f"{len(overlap)} overlap ({result['vocabulary']['overlap_pct']:.1f}%)")

    # Prediction validation
    d_trace = result["properties"]["trace_length"]["cohens_d"]
    if abs(d_trace) < 0.2:
        verdict = "CONFIRMED: trace_length does NOT invert (negligible d)"
    elif d_trace > 0:
        verdict = "PARTIALLY CONFIRMED: trace_length d>0 (same direction as SWE-agent/ATBench, not synthetic)"
    else:
        verdict = "DISCONFIRMED: trace_length inverts (d<0, same as synthetic)"

    result["prediction_verdict"] = verdict
    print(f"\nPrediction verdict: {verdict}")

    elapsed = time.time() - t0
    result["elapsed_seconds"] = round(elapsed, 1)
    print(f"\nCompleted in {elapsed:.1f}s")

    out_path = OUTPUT_DIR / "v25_agentharm_polarity.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
