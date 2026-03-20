"""TRAIL-excluded sensitivity analysis.

Addresses reviewer concern: "TRAIL's 148 traces introduce high variance
on TRAIL-involving pairs."

Recomputes all polarity statistics (structural heuristic features)
with TRAIL excluded. If the within/between gap persists, the finding
is not driven by TRAIL's small size.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_trail_excluded_sensitivity.py
    PYTHONPATH=../../lib:src uv run python scripts/run_trail_excluded_sensitivity.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research_utils.reproducibility import set_seeds
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

# 4-group taxonomy
SEMANTIC_GROUPS_FULL: dict[str, list[str]] = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "rjudge", "agentharm", "injecagent"],
    "api_misuse": ["toolbench"],
}

# Same taxonomy but without trail
SEMANTIC_GROUPS_NO_TRAIL: dict[str, list[str]] = {
    "injected": ["synthetic"],
    "failure": ["swe_agent"],  # singleton now
    "safety": ["atbench", "rjudge", "agentharm", "injecagent"],
    "api_misuse": ["toolbench"],
}

ALL_DOMAINS = [
    "synthetic", "swe_agent", "trail", "atbench",
    "rjudge", "agentharm", "injecagent", "toolbench",
]

DOMAINS_NO_TRAIL = [d for d in ALL_DOMAINS if d != "trail"]

HEURISTIC_FEATURES = ["trace_length", "distinct_tools", "repetition_rate"]


def load_domain_traces(domain_name: str, *, max_traces: int = 5000) -> list:
    import random
    from data.trace_loader import (
        _inject_anomaly, download_trail, generate_synthetic_traces,
        load_agentharm, load_atbench, load_injecagent,
        load_rjudge, load_swe_agent, load_toolbench, load_trail,
    )

    if domain_name == "synthetic":
        return generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42)
    elif domain_name == "atbench":
        return load_atbench(max_traces=max_traces)
    elif domain_name == "agentharm":
        return load_agentharm(max_traces=max_traces)
    elif domain_name == "injecagent":
        return load_injecagent(Path("data"), max_traces=max_traces)
    elif domain_name == "rjudge":
        return load_rjudge(Path("data"), max_traces=max_traces)
    elif domain_name == "swe_agent":
        return load_swe_agent(max_traces=max_traces)
    elif domain_name == "toolbench":
        raw_traces = load_toolbench(Path("data"), max_traces=max_traces)
        rng = random.Random(42)
        traces = []
        for trace in raw_traces:
            if rng.random() < 0.10:
                from data.schema import Trace
                new_actions = _inject_anomaly(list(trace.actions), rng)
                traces.append(Trace(
                    trace_id=trace.trace_id, actions=new_actions,
                    source=trace.source, task=trace.task,
                ))
            else:
                traces.append(trace)
        return traces
    elif domain_name == "trail":
        trail_dir = Path("data/trail/TRAIL-Benchmark")
        if not trail_dir.exists():
            trail_dir = download_trail(trail_dir.parent)
        return load_trail(trail_dir)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def compute_heuristic_features(traces: list) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3 heuristic features for all traces."""
    features = []
    labels = []
    for trace in traces:
        n_actions = len(trace.actions)
        tools = [a.tool for a in trace.actions]
        distinct = len(set(tools))
        rep_rate = 1.0 - (distinct / max(n_actions, 1))
        features.append([n_actions, distinct, rep_rate])
        labels.append(1 if trace.has_anomaly else 0)
    return np.array(features), np.array(labels)


def compute_d_vector(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cohen's d per feature dimension."""
    d_vec = np.zeros(features.shape[1])
    normal = labels == 0
    anomalous = labels == 1
    if normal.sum() < 2 or anomalous.sum() < 2:
        return d_vec * float("nan")
    for dim in range(features.shape[1]):
        d_vec[dim] = cohens_d(features[anomalous, dim], features[normal, dim])
    return d_vec


def analyze_polarity(
    domain_d: dict[str, np.ndarray],
    domains: list[str],
    group_map: dict[str, str],
) -> dict:
    """Compute polarity stats for a set of domains and group mapping."""
    pairs = [(da, db) for i, da in enumerate(domains) for db in domains[i+1:]]

    within_flips = 0
    within_total = 0
    between_flips = 0
    between_total = 0
    pair_details = {}

    for da, db in pairs:
        ga = group_map.get(da, "?")
        gb = group_map.get(db, "?")
        is_within = ga == gb

        d_a, d_b = domain_d[da], domain_d[db]

        n_flipped = 0
        n_comparable = 0
        for k in range(len(d_a)):
            if np.isnan(d_a[k]) or np.isnan(d_b[k]):
                continue
            if abs(d_a[k]) > 0.1 and abs(d_b[k]) > 0.1:
                n_comparable += 1
                if np.sign(d_a[k]) != np.sign(d_b[k]):
                    n_flipped += 1

        pair_details[f"{da}-{db}"] = {
            "is_within": is_within,
            "n_flipped": n_flipped,
            "n_comparable": n_comparable,
            "flip_rate": n_flipped / max(n_comparable, 1),
        }

        if is_within:
            within_flips += n_flipped
            within_total += n_comparable
        else:
            between_flips += n_flipped
            between_total += n_comparable

    within_rate = within_flips / max(within_total, 1)
    between_rate = between_flips / max(between_total, 1)
    gap = between_rate / max(within_rate, 0.001) if within_rate > 0 else float("inf")

    return {
        "within_flips": within_flips,
        "within_total": within_total,
        "within_rate": within_rate,
        "between_flips": between_flips,
        "between_total": between_total,
        "between_rate": between_rate,
        "gap_ratio": gap,
        "pair_details": pair_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRAIL-excluded sensitivity analysis")
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.max_traces = 500

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    set_seeds(42)

    print("=" * 70)
    print("  TRAIL-EXCLUDED SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Load all domains including trail
    domain_d: dict[str, np.ndarray] = {}
    domain_stats: dict[str, dict] = {}

    for domain in ALL_DOMAINS:
        print(f"\n  Loading {domain}...")
        try:
            traces = load_domain_traces(domain, max_traces=args.max_traces)
            features, labels = compute_heuristic_features(traces)
            d_vec = compute_d_vector(features, labels)
            domain_d[domain] = d_vec

            n_norm = int((labels == 0).sum())
            n_anom = int((labels == 1).sum())
            domain_stats[domain] = {
                "n_traces": len(traces), "n_normal": n_norm, "n_anomalous": n_anom,
                "d_vector": [float(x) for x in d_vec],
            }
            print(f"    {len(traces)} traces, d = [{', '.join(f'{x:+.2f}' for x in d_vec)}]")
        except Exception as e:
            print(f"    FAILED: {e}")

    loaded_full = sorted(d for d in ALL_DOMAINS if d in domain_d)
    loaded_no_trail = [d for d in loaded_full if d != "trail"]

    group_map_full = {d: g for g, ds in SEMANTIC_GROUPS_FULL.items() for d in ds}
    group_map_no_trail = {d: g for g, ds in SEMANTIC_GROUPS_NO_TRAIL.items() for d in ds}

    # Analysis 1: Full (with TRAIL)
    print(f"\n{'='*70}")
    print("  ANALYSIS 1: ALL DOMAINS (with TRAIL)")
    print(f"{'='*70}")

    full_results = analyze_polarity(domain_d, loaded_full, group_map_full)
    print(f"  Within-group reversal:  {full_results['within_flips']}/{full_results['within_total']} "
          f"= {full_results['within_rate']:.1%}")
    print(f"  Between-group reversal: {full_results['between_flips']}/{full_results['between_total']} "
          f"= {full_results['between_rate']:.1%}")
    print(f"  Gap ratio: {full_results['gap_ratio']:.1f}x")

    # Analysis 2: Without TRAIL
    print(f"\n{'='*70}")
    print("  ANALYSIS 2: TRAIL EXCLUDED")
    print(f"{'='*70}")

    no_trail_results = analyze_polarity(domain_d, loaded_no_trail, group_map_no_trail)
    print(f"  Within-group reversal:  {no_trail_results['within_flips']}/{no_trail_results['within_total']} "
          f"= {no_trail_results['within_rate']:.1%}")
    print(f"  Between-group reversal: {no_trail_results['between_flips']}/{no_trail_results['between_total']} "
          f"= {no_trail_results['between_rate']:.1%}")
    print(f"  Gap ratio: {no_trail_results['gap_ratio']:.1f}x")

    # Analysis 3: TRAIL-involving vs non-TRAIL pairs (within full analysis)
    print(f"\n{'='*70}")
    print("  ANALYSIS 3: TRAIL-INVOLVING vs NON-TRAIL PAIRS")
    print(f"{'='*70}")

    trail_pairs_flips = 0
    trail_pairs_total = 0
    non_trail_pairs_flips = 0
    non_trail_pairs_total = 0

    for pair_key, pd in full_results["pair_details"].items():
        involves_trail = "trail" in pair_key.split("-")
        if involves_trail:
            trail_pairs_flips += pd["n_flipped"]
            trail_pairs_total += pd["n_comparable"]
        else:
            non_trail_pairs_flips += pd["n_flipped"]
            non_trail_pairs_total += pd["n_comparable"]

    trail_rate = trail_pairs_flips / max(trail_pairs_total, 1)
    non_trail_rate = non_trail_pairs_flips / max(non_trail_pairs_total, 1)

    print(f"  TRAIL-involving pairs:     {trail_pairs_flips}/{trail_pairs_total} = {trail_rate:.1%}")
    print(f"  Non-TRAIL pairs:           {non_trail_pairs_flips}/{non_trail_pairs_total} = {non_trail_rate:.1%}")

    # Per-feature analysis
    print(f"\n{'='*70}")
    print("  PER-FEATURE D-VECTOR COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Domain':<15s} {'trace_len':>10s} {'distinct':>10s} {'rep_rate':>10s}")
    print("  " + "-" * 48)
    for domain in loaded_full:
        d = domain_d[domain]
        print(f"  {domain:<15s} {d[0]:>+10.2f} {d[1]:>+10.2f} {d[2]:>+10.2f}")

    elapsed = time.time() - t0

    # Comparison summary
    print(f"\n{'='*70}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<25s} {'Within%':>10s} {'Between%':>10s} {'Gap':>8s}")
    print("  " + "-" * 55)
    print(f"  {'All domains':<25s} {full_results['within_rate']:>10.1%} "
          f"{full_results['between_rate']:>10.1%} {full_results['gap_ratio']:>8.1f}x")
    print(f"  {'TRAIL excluded':<25s} {no_trail_results['within_rate']:>10.1%} "
          f"{no_trail_results['between_rate']:>10.1%} {no_trail_results['gap_ratio']:>8.1f}x")

    output = {
        "experiment": "v31_trail_excluded_sensitivity",
        "features": HEURISTIC_FEATURES,
        "domain_stats": domain_stats,
        "full_analysis": {
            k: v for k, v in full_results.items() if k != "pair_details"
        },
        "full_pair_details": full_results["pair_details"],
        "no_trail_analysis": {
            k: v for k, v in no_trail_results.items() if k != "pair_details"
        },
        "no_trail_pair_details": no_trail_results["pair_details"],
        "trail_sensitivity": {
            "trail_involving_flip_rate": trail_rate,
            "non_trail_flip_rate": non_trail_rate,
        },
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "v31_trail_excluded_sensitivity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
