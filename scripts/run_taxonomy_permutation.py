"""Permutation test for taxonomy validity.

Tests whether the observed within/between-group reversal rate gap is
statistically significant under the null hypothesis that the semantic
group assignments are arbitrary (random permutation of datasets to groups
preserving group sizes).

Addresses DA reviewer Major #1: "Taxonomy validity is asserted, not demonstrated."

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_taxonomy_permutation.py
    PYTHONPATH=../../lib:src uv run python scripts/run_taxonomy_permutation.py --n-permutations 50000
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

from data.domain_loader import load_all_domains
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

STRUCTURAL_FEATURES = ["trace_length", "low_diversity", "repeat_rate"]

# Actual taxonomy from the paper
SEMANTIC_GROUPS = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["agentdojo", "agentharm", "atbench", "injecagent", "rjudge"],
    "api_misuse": ["toolbench"],
}

# Alternative: 5-group taxonomy (splitting safety into prompt-injection + safety)
SEMANTIC_GROUPS_5 = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "prompt_injection": ["agentdojo", "injecagent"],
    "safety": ["atbench", "agentharm", "rjudge"],
    "api_misuse": ["toolbench"],
}


def compute_d_vectors(
    domain_data: dict,
    features: list[str],
) -> dict[str, np.ndarray]:
    """Compute Cohen's d vector for each domain."""
    d_vectors: dict[str, np.ndarray] = {}
    for domain_name, (feats, labels) in domain_data.items():
        normal_mask = labels == 0
        anomalous_mask = labels == 1
        d_vec = []
        for f in features:
            if f not in feats:
                d_vec.append(0.0)
                continue
            d = cohens_d(feats[f][anomalous_mask], feats[f][normal_mask])
            d_vec.append(d)
        d_vectors[domain_name] = np.array(d_vec)
    return d_vectors


def check_sign_reversal(d_a: float, d_b: float, threshold: float = 0.1) -> bool:
    """Check if two Cohen's d values have opposite signs (with min effect)."""
    if abs(d_a) < threshold or abs(d_b) < threshold:
        return False
    return (d_a > 0) != (d_b > 0)


def compute_gap(
    domains: list[str],
    d_vectors: dict[str, np.ndarray],
    domain_to_group: dict[str, str],
    features: list[str],
) -> dict:
    """Compute within/between reversal rates and their gap."""
    within_reversals = 0
    within_total = 0
    between_reversals = 0
    between_total = 0

    for da, db in combinations(domains, 2):
        ga = domain_to_group.get(da, "unknown")
        gb = domain_to_group.get(db, "unknown")
        n_rev = sum(
            1
            for i, f in enumerate(features)
            if check_sign_reversal(d_vectors[da][i], d_vectors[db][i])
        )
        if ga == gb:
            within_reversals += n_rev
            within_total += len(features)
        else:
            between_reversals += n_rev
            between_total += len(features)

    within_rate = within_reversals / max(within_total, 1)
    between_rate = between_reversals / max(between_total, 1)
    gap = between_rate / max(within_rate, 1e-10)

    return {
        "within_rate": within_rate,
        "between_rate": between_rate,
        "gap": gap,
        "within_reversals": within_reversals,
        "within_total": within_total,
        "between_reversals": between_reversals,
        "between_total": between_total,
    }


def run_permutation_test(
    domains: list[str],
    d_vectors: dict[str, np.ndarray],
    group_assignment: dict[str, str],
    n_permutations: int,
    features: list[str],
    seed: int = 42,
) -> dict:
    """Permutation test: randomly reassign datasets to groups, compute null gap distribution."""
    rng = np.random.default_rng(seed)

    # Observed gap
    observed = compute_gap(domains, d_vectors, group_assignment, features)
    observed_gap = observed["gap"]
    observed_diff = observed["between_rate"] - observed["within_rate"]

    # Get group sizes from original assignment
    group_names = sorted(set(group_assignment.values()))
    group_sizes = []
    for g in group_names:
        size = sum(1 for d in domains if group_assignment.get(d) == g)
        group_sizes.append(size)

    null_gaps = []
    null_diffs = []

    for _ in range(n_permutations):
        # Randomly permute dataset-to-group assignments (preserving group sizes)
        perm = rng.permutation(domains)
        perm_assignment = {}
        idx = 0
        for g, size in zip(group_names, group_sizes):
            for d in perm[idx : idx + size]:
                perm_assignment[d] = g
            idx += size

        perm_result = compute_gap(domains, d_vectors, perm_assignment, features)
        null_gaps.append(perm_result["gap"])
        null_diffs.append(perm_result["between_rate"] - perm_result["within_rate"])

    null_gaps = np.array(null_gaps)
    null_diffs = np.array(null_diffs)

    # p-value: fraction of null gaps >= observed
    p_gap = float(np.mean(null_gaps >= observed_gap))
    p_diff = float(np.mean(null_diffs >= observed_diff))

    return {
        "observed_gap": float(observed_gap),
        "observed_diff": float(observed_diff),
        "observed_within_rate": float(observed["within_rate"]),
        "observed_between_rate": float(observed["between_rate"]),
        "null_gap_mean": float(np.mean(null_gaps)),
        "null_gap_std": float(np.std(null_gaps)),
        "null_gap_median": float(np.median(null_gaps)),
        "null_gap_95pct": float(np.percentile(null_gaps, 95)),
        "null_diff_mean": float(np.mean(null_diffs)),
        "null_diff_std": float(np.std(null_diffs)),
        "p_value_gap": p_gap,
        "p_value_diff": p_diff,
        "n_permutations": n_permutations,
        "group_sizes": dict(zip(group_names, group_sizes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation test for taxonomy validity")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-swe-traces", type=int, default=5000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Taxonomy Validity Permutation Test")
    print("=" * 70)

    print("\nLoading all domains...")
    domain_data = load_all_domains(seeds=[42], max_swe_traces=args.max_swe_traces)
    domains = sorted(domain_data.keys())
    print(f"  Loaded {len(domains)} domains: {domains}")

    print("\nComputing Cohen's d vectors (structural features)...")
    d_vectors = compute_d_vectors(domain_data, STRUCTURAL_FEATURES)

    for d in domains:
        d_str = " ".join(f"{v:+.3f}" for v in d_vectors[d])
        print(f"  {d:15s}: [{d_str}]")

    results = {}

    # Test 1: 4-group taxonomy
    print("\n" + "=" * 70)
    print("  Test 1: 4-Group Taxonomy (structural features)")
    print("=" * 70)

    domain_to_group_4 = {}
    for group, members in SEMANTIC_GROUPS.items():
        for d in members:
            if d in domains:
                domain_to_group_4[d] = group

    result_4g = run_permutation_test(
        domains, d_vectors, domain_to_group_4, args.n_permutations,
        STRUCTURAL_FEATURES, seed=args.seed,
    )
    results["4_group_structural"] = result_4g
    print(f"\n  Observed gap: {result_4g['observed_gap']:.2f}x "
          f"(within={result_4g['observed_within_rate']:.3f}, "
          f"between={result_4g['observed_between_rate']:.3f})")
    print(f"  Null gap:     {result_4g['null_gap_mean']:.2f}x "
          f"(std={result_4g['null_gap_std']:.2f})")
    print(f"  p-value (gap >= observed): {result_4g['p_value_gap']:.4f}")
    print(f"  p-value (diff >= observed): {result_4g['p_value_diff']:.4f}")

    # Test 2: 5-group taxonomy
    print("\n" + "=" * 70)
    print("  Test 2: 5-Group Taxonomy (structural features)")
    print("=" * 70)

    domain_to_group_5 = {}
    for group, members in SEMANTIC_GROUPS_5.items():
        for d in members:
            if d in domains:
                domain_to_group_5[d] = group

    result_5g = run_permutation_test(
        domains, d_vectors, domain_to_group_5, args.n_permutations,
        STRUCTURAL_FEATURES, seed=args.seed,
    )
    results["5_group_structural"] = result_5g
    print(f"\n  Observed gap: {result_5g['observed_gap']:.2f}x "
          f"(within={result_5g['observed_within_rate']:.3f}, "
          f"between={result_5g['observed_between_rate']:.3f})")
    print(f"  Null gap:     {result_5g['null_gap_mean']:.2f}x "
          f"(std={result_5g['null_gap_std']:.2f})")
    print(f"  p-value (gap >= observed): {result_5g['p_value_gap']:.4f}")
    print(f"  p-value (diff >= observed): {result_5g['p_value_diff']:.4f}")

    # Test 3: 4-group taxonomy with ALL 6 features (structural + argument)
    print("\n" + "=" * 70)
    print("  Test 3: 4-Group Taxonomy (all 6 features)")
    print("=" * 70)

    all_features = STRUCTURAL_FEATURES + ["mean_arg_length", "arg_diversity", "arg_empty_rate"]
    d_vectors_all = compute_d_vectors(domain_data, all_features)

    result_4g_all = run_permutation_test(
        domains, d_vectors_all, domain_to_group_4, args.n_permutations,
        all_features, seed=args.seed,
    )
    results["4_group_all_features"] = result_4g_all
    print(f"\n  Observed gap: {result_4g_all['observed_gap']:.2f}x "
          f"(within={result_4g_all['observed_within_rate']:.3f}, "
          f"between={result_4g_all['observed_between_rate']:.3f})")
    print(f"  Null gap:     {result_4g_all['null_gap_mean']:.2f}x "
          f"(std={result_4g_all['null_gap_std']:.2f})")
    print(f"  p-value (gap >= observed): {result_4g_all['p_value_gap']:.4f}")
    print(f"  p-value (diff >= observed): {result_4g_all['p_value_diff']:.4f}")

    # Test 4: Leave-one-out sensitivity (drop each dataset, re-test)
    print("\n" + "=" * 70)
    print("  Test 4: Leave-One-Out Sensitivity (4-group, structural)")
    print("=" * 70)

    loo_results = {}
    for drop in domains:
        remaining = [d for d in domains if d != drop]
        remaining_d = {d: d_vectors[d] for d in remaining}
        remaining_assign = {d: g for d, g in domain_to_group_4.items() if d != drop}

        # Skip if any group becomes empty
        remaining_groups = set(remaining_assign.values())
        if len(remaining_groups) < 2:
            continue

        loo_result = compute_gap(remaining, remaining_d, remaining_assign, STRUCTURAL_FEATURES)
        loo_results[drop] = {
            "within_rate": float(loo_result["within_rate"]),
            "between_rate": float(loo_result["between_rate"]),
            "gap": float(loo_result["gap"]),
        }
        print(f"  Drop {drop:15s}: gap={loo_result['gap']:.2f}x "
              f"(W={loo_result['within_rate']:.3f}, B={loo_result['between_rate']:.3f})")

    results["leave_one_out"] = loo_results

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"\n  4-group structural: p={result_4g['p_value_diff']:.4f} "
          f"(gap={result_4g['observed_gap']:.2f}x)")
    print(f"  5-group structural: p={result_5g['p_value_diff']:.4f} "
          f"(gap={result_5g['observed_gap']:.2f}x)")
    print(f"  4-group all feats:  p={result_4g_all['p_value_diff']:.4f} "
          f"(gap={result_4g_all['observed_gap']:.2f}x)")
    loo_gap_strs = [f"{v['gap']:.1f}x" for v in loo_results.values()]
    print(f"\n  Leave-one-out gaps: {loo_gap_strs}")
    min_gap = min(v["gap"] for v in loo_results.values())
    print(f"  Minimum LOO gap: {min_gap:.2f}x")

    # Save
    out_path = OUTPUT_DIR / "v31_taxonomy_permutation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
