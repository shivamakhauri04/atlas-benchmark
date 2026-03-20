"""v26: Feature Stability Index (FSI) analysis across all domains.

Computes per-domain Spearman correlations and cross-domain FSI matrix
for trace_length, low_diversity, repeat_rate features.
Runs permutation tests for significance.

Domains: Synthetic, SWE-agent, ATBench, AgentHarm, R-Judge, InjecAgent

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_fsi_analysis.py
    PYTHONPATH=../../lib:src uv run python scripts/run_fsi_analysis.py --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds

from data.domain_loader import load_all_domains
from trustlens.polarity import (
    aggregate_fsi,
    compute_domain_correlations,
    compute_polarity_matrix,
    fsi_significance_test,
    identify_puf,
)
from trustlens.stats import bootstrap_ci

OUTPUT_DIR = Path("output/tables")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v26: FSI analysis")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--max-swe-traces", type=int, default=5000)
    p.add_argument("--max-rjudge", type=int, default=None)
    p.add_argument("--max-injecagent", type=int, default=None)
    return p.parse_args()


def run_single_seed(
    seed: int,
    domain_data: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
    n_permutations: int,
) -> dict:
    """Run FSI analysis for one seed."""
    set_seeds(seed)

    domain_features = {name: data[0] for name, data in domain_data.items()}
    domain_labels = {name: data[1] for name, data in domain_data.items()}

    per_domain_corr = {}
    for name in domain_features:
        corr = compute_domain_correlations(domain_features[name], domain_labels[name])
        per_domain_corr[name] = corr

    fsi_matrix = compute_polarity_matrix(domain_features, domain_labels)

    puf_by_pair: dict[str, list[str]] = {}
    agg_fsi_by_pair: dict[str, float] = {}
    domains = sorted(domain_features.keys())
    for da in domains:
        for db in domains:
            if da == db:
                continue
            key = f"{da}_vs_{db}"
            puf = identify_puf(fsi_matrix[da][db])
            puf_by_pair[key] = sorted(puf)
            agg_fsi_by_pair[key] = aggregate_fsi(fsi_matrix[da][db])

    sig_results = {}
    for name in domain_features:
        sig = fsi_significance_test(
            domain_features[name],
            domain_labels[name],
            n_permutations=n_permutations,
            seed=seed,
        )
        sig_results[name] = sig

    return {
        "seed": seed,
        "per_domain_correlations": per_domain_corr,
        "fsi_matrix": {
            da: {db: fsi_matrix[da][db] for db in domains}
            for da in domains
        },
        "puf_by_pair": puf_by_pair,
        "aggregate_fsi_by_pair": agg_fsi_by_pair,
        "significance": sig_results,
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("v26: Feature Stability Index (FSI) Analysis")
    print("=" * 60)

    print("\nLoading domains...")
    domain_data = load_all_domains(
        seeds=args.seeds,
        max_swe_traces=args.max_swe_traces,
        max_rjudge=args.max_rjudge,
        max_injecagent=args.max_injecagent,
    )
    print(f"\nLoaded {len(domain_data)} domains: {sorted(domain_data.keys())}")

    print(f"\nRunning FSI analysis ({len(args.seeds)} seeds)...")
    all_seed_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_single_seed(seed, domain_data, args.n_permutations)
        all_seed_results.append(result)

    domains = sorted(domain_data.keys())
    feature_names = sorted(next(iter(domain_data.values()))[0].keys())

    aggregate_fsi_values: dict[str, dict[str, list[float]]] = {}
    for da in domains:
        for db in domains:
            if da == db:
                continue
            key = f"{da}_vs_{db}"
            aggregate_fsi_values[key] = {}
            for f in feature_names:
                vals = []
                for r in all_seed_results:
                    v = r["fsi_matrix"].get(da, {}).get(db, {}).get(f, 0.0)
                    vals.append(v)
                aggregate_fsi_values[key][f] = vals

    fsi_summary: dict[str, dict[str, dict]] = {}
    for pair_key, feat_vals in aggregate_fsi_values.items():
        fsi_summary[pair_key] = {}
        for f, vals in feat_vals.items():
            ci = bootstrap_ci(np.array(vals))
            fsi_summary[pair_key][f] = ci

    output = {
        "version": "v26",
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "domains": domains,
        "features": feature_names,
        "n_permutations": args.n_permutations,
        "domain_sizes": {
            name: int(len(data[1])) for name, data in domain_data.items()
        },
        "fsi_summary": fsi_summary,
        "per_seed_results": all_seed_results,
        "elapsed_seconds": time.time() - t0,
    }

    out_path = OUTPUT_DIR / "fsi_analysis.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    print("\n--- FSI Summary ---")
    for pair_key in sorted(fsi_summary.keys()):
        feats = fsi_summary[pair_key]
        reversed_feats = [f for f, ci in feats.items() if ci["mean"] < 0]
        if reversed_feats:
            print(f"  {pair_key}: REVERSED features = {reversed_feats}")
        else:
            print(f"  {pair_key}: all features stable")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
