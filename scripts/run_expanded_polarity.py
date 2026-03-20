"""v30: Expanded polarity analysis across 8 datasets (D&B pivot).

Computes Cohen's d for trace_length, low_diversity, repeat_rate across
all available datasets including ToolBench and TRAIL. Builds the polarity
matrix (dataset x feature x d). Groups by anomaly semantics to test
whether reversal occurs between semantic groups but not within.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_expanded_polarity.py
    PYTHONPATH=../../lib:src uv run python scripts/run_expanded_polarity.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from data.domain_loader import (
    DomainData,
    load_all_domains,
)
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

STRUCTURAL_FEATURES = ["trace_length", "low_diversity", "repeat_rate"]
ARG_FEATURES = ["mean_arg_length", "arg_diversity", "arg_empty_rate"]
FEATURE_NAMES = STRUCTURAL_FEATURES + ARG_FEATURES

SEMANTIC_GROUPS_5 = {
    "injected": ["synthetic", "toolace"],
    "failure": ["swe_agent", "trail", "stabletoolbench"],
    "safety": ["atbench", "agentharm", "rjudge"],
    "prompt_injection": ["injecagent", "agentdojo"],
    "api_misuse": ["toolbench", "apibank"],
}

SEMANTIC_GROUPS_4 = {
    "injected": ["synthetic", "toolace"],
    "failure": ["swe_agent", "trail", "stabletoolbench"],
    "safety": ["atbench", "agentharm", "rjudge", "injecagent", "agentdojo"],
    "api_misuse": ["toolbench", "apibank"],
}

# Default to 5-group for analysis
SEMANTIC_GROUPS = SEMANTIC_GROUPS_5


def compute_polarity_row(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    domain_name: str,
) -> dict[str, float]:
    """Compute Cohen's d (anomalous vs normal) for each feature."""
    normal_mask = labels == 0
    anomalous_mask = labels == 1
    n_normal = int(normal_mask.sum())
    n_anomalous = int(anomalous_mask.sum())

    row: dict[str, float] = {
        "domain": domain_name,
        "n_traces": len(labels),
        "n_normal": n_normal,
        "n_anomalous": n_anomalous,
    }

    for feat in FEATURE_NAMES:
        if feat not in features:
            row[f"d_{feat}"] = float("nan")
            continue
        vals = features[feat]
        normal_vals = vals[normal_mask]
        anomalous_vals = vals[anomalous_mask]
        d = cohens_d(anomalous_vals, normal_vals)
        row[f"d_{feat}"] = d

    return row


def check_sign_reversal(
    d_a: float,
    d_b: float,
    threshold: float = 0.1,
) -> bool:
    """Check if two Cohen's d values represent sign reversal."""
    if abs(d_a) < threshold or abs(d_b) < threshold:
        return False
    return (d_a > 0) != (d_b > 0)


def run_multi_seed_polarity(
    domain_data: dict[str, DomainData],
    seeds: list[int],
) -> dict[str, list[dict]]:
    """Run polarity analysis across seeds with bootstrap resampling."""
    all_rows: dict[str, list[dict]] = {}

    for domain_name, (features, labels) in domain_data.items():
        seed_rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            n = len(labels)
            idx = rng.choice(n, size=n, replace=True)
            resampled_features = {k: v[idx] for k, v in features.items()}
            resampled_labels = labels[idx]
            row = compute_polarity_row(resampled_features, resampled_labels, domain_name)
            row["seed"] = seed
            seed_rows.append(row)
        all_rows[domain_name] = seed_rows

    return all_rows


def build_reversal_matrix(
    domain_data: dict[str, DomainData],
) -> dict[str, dict[str, dict[str, bool]]]:
    """Build NxN reversal matrix for all domain pairs."""
    domains = sorted(domain_data.keys())
    d_values: dict[str, dict[str, float]] = {}
    for domain_name, (features, labels) in domain_data.items():
        row = compute_polarity_row(features, labels, domain_name)
        d_values[domain_name] = {f: row[f"d_{f}"] for f in FEATURE_NAMES}

    matrix: dict[str, dict[str, dict[str, bool]]] = {}
    for da in domains:
        matrix[da] = {}
        for db in domains:
            if da == db:
                matrix[da][db] = {f: False for f in FEATURE_NAMES}
            else:
                matrix[da][db] = {
                    f: check_sign_reversal(d_values[da][f], d_values[db][f])
                    for f in FEATURE_NAMES
                }
    return matrix


def _analyze_with_groups(
    domain_data: dict[str, DomainData],
    reversal_matrix: dict[str, dict[str, dict[str, bool]]],
    groups: dict[str, list[str]],
) -> dict:
    """Analyze reversal rates with a specific group mapping."""
    domains = sorted(domain_data.keys())
    domain_to_group: dict[str, str] = {}
    for group, members in groups.items():
        for d in members:
            if d in domains:
                domain_to_group[d] = group

    within_group_reversals = 0
    within_group_pairs = 0
    between_group_reversals = 0
    between_group_pairs = 0

    for da in domains:
        for db in domains:
            if da >= db:
                continue
            ga = domain_to_group.get(da, "unknown")
            gb = domain_to_group.get(db, "unknown")
            n_rev = sum(1 for f in FEATURE_NAMES if reversal_matrix[da][db].get(f, False))
            if ga == gb:
                within_group_pairs += 1
                within_group_reversals += n_rev
            else:
                between_group_pairs += 1
                between_group_reversals += n_rev

    return {
        "within_group_pairs": within_group_pairs,
        "within_group_reversals": within_group_reversals,
        "within_group_rate": (
            within_group_reversals / max(within_group_pairs * len(FEATURE_NAMES), 1)
        ),
        "between_group_pairs": between_group_pairs,
        "between_group_reversals": between_group_reversals,
        "between_group_rate": (
            between_group_reversals / max(between_group_pairs * len(FEATURE_NAMES), 1)
        ),
    }


def analyze_semantic_groups(
    domain_data: dict[str, DomainData],
    reversal_matrix: dict[str, dict[str, dict[str, bool]]],
) -> dict:
    """Check if reversal occurs between groups but not within."""
    within_group_reversals = 0
    within_group_pairs = 0
    between_group_reversals = 0
    between_group_pairs = 0

    domains = sorted(domain_data.keys())
    domain_to_group: dict[str, str] = {}
    for group, members in SEMANTIC_GROUPS.items():
        for d in members:
            if d in domains:
                domain_to_group[d] = group

    for da in domains:
        for db in domains:
            if da >= db:
                continue
            ga = domain_to_group.get(da, "unknown")
            gb = domain_to_group.get(db, "unknown")
            n_rev = sum(1 for f in FEATURE_NAMES if reversal_matrix[da][db].get(f, False))
            if ga == gb:
                within_group_pairs += 1
                within_group_reversals += n_rev
            else:
                between_group_pairs += 1
                between_group_reversals += n_rev

    return {
        "within_group_pairs": within_group_pairs,
        "within_group_reversals": within_group_reversals,
        "within_group_rate": (
            within_group_reversals / max(within_group_pairs * len(FEATURE_NAMES), 1)
        ),
        "between_group_pairs": between_group_pairs,
        "between_group_reversals": between_group_reversals,
        "between_group_rate": (
            between_group_reversals / max(between_group_pairs * len(FEATURE_NAMES), 1)
        ),
    }


def wilcoxon_reversal_test(
    seed_rows_a: list[dict],
    seed_rows_b: list[dict],
    feature: str,
) -> dict:
    """Wilcoxon signed-rank test for sign reversal between domains."""
    d_a = np.array([r[f"d_{feature}"] for r in seed_rows_a])
    d_b = np.array([r[f"d_{feature}"] for r in seed_rows_b])

    d_a = d_a[np.isfinite(d_a)]
    d_b = d_b[np.isfinite(d_b)]
    n = min(len(d_a), len(d_b))
    if n < 3:
        return {"p_value": float("nan"), "significant": False}

    d_a, d_b = d_a[:n], d_b[:n]
    products = d_a * d_b

    try:
        _, p = stats.wilcoxon(products, alternative="less")
    except ValueError:
        p = 1.0

    return {
        "p_value": float(p),
        "significant": p < 0.05,
        "n_pairs": n,
        "mean_product": float(products.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v30: Expanded polarity analysis")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-swe-traces", type=int, default=5000)
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123, 456]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  v30: Expanded Polarity Analysis (8 datasets)")
    print("=" * 70)

    print("\nLoading all available domains...")
    domain_data = load_all_domains(
        seeds=args.seeds,
        max_swe_traces=args.max_swe_traces,
    )
    loaded = sorted(domain_data.keys())
    print(f"\n  Loaded {len(loaded)} domains: {loaded}")

    print("\n" + "=" * 70)
    print("  Phase 1: Cohen's d Matrix (Full Dataset)")
    print("=" * 70)

    for domain_name, (features, labels) in sorted(domain_data.items()):
        row = compute_polarity_row(features, labels, domain_name)
        print(f"\n  {domain_name} (n={row['n_traces']}, "
              f"anom={row['n_anomalous']}):")
        for feat in FEATURE_NAMES:
            d = row[f"d_{feat}"]
            sign = "+" if d >= 0 else "-"
            print(f"    {feat:20s}: d={sign}{abs(d):.3f}")

    print("\n" + "=" * 70)
    print("  Phase 2: Reversal Matrix")
    print("=" * 70)

    reversal_matrix = build_reversal_matrix(domain_data)
    print(f"\n  {'':15s}" + "".join(f"{d:>15s}" for d in loaded))
    for da in loaded:
        row_str = f"  {da:15s}"
        for db in loaded:
            n_rev = sum(1 for f in FEATURE_NAMES if reversal_matrix[da][db].get(f, False))
            row_str += f"{n_rev:>15d}"
        print(row_str)

    print("\n" + "=" * 70)
    print("  Phase 3: Semantic Group Analysis")
    print("=" * 70)

    # Run both 5-group and 4-group analyses
    saved_groups = SEMANTIC_GROUPS.copy()

    for label, groups in [("5-group", SEMANTIC_GROUPS_5), ("4-group", SEMANTIC_GROUPS_4)]:
        # Temporarily swap
        group_analysis = analyze_semantic_groups(
            domain_data, reversal_matrix,
        )
        # Re-run with explicit group override
        ga = _analyze_with_groups(domain_data, reversal_matrix, groups)
        print(f"\n  [{label}]")
        print(f"  Within-group reversal rate:  "
              f"{ga['within_group_rate']:.3f} "
              f"({ga['within_group_reversals']}/"
              f"{ga['within_group_pairs'] * len(FEATURE_NAMES)})")
        print(f"  Between-group reversal rate: "
              f"{ga['between_group_rate']:.3f} "
              f"({ga['between_group_reversals']}/"
              f"{ga['between_group_pairs'] * len(FEATURE_NAMES)})")
        ratio = (
            ga['between_group_rate'] / max(ga['within_group_rate'], 0.001)
        )
        print(f"  Gap: {ratio:.1f}x")

    group_analysis_5 = _analyze_with_groups(domain_data, reversal_matrix, SEMANTIC_GROUPS_5)
    group_analysis_4 = _analyze_with_groups(domain_data, reversal_matrix, SEMANTIC_GROUPS_4)

    print("\n" + "=" * 70)
    print("  Phase 4: Multi-Seed Bootstrap + Wilcoxon Tests")
    print("=" * 70)

    seed_rows = run_multi_seed_polarity(domain_data, args.seeds)

    wilcoxon_results: dict[str, dict] = {}
    for da in loaded:
        for db in loaded:
            if da >= db:
                continue
            pair_key = f"{da}->{db}"
            pair_results: dict[str, dict] = {}
            for feat in FEATURE_NAMES:
                test = wilcoxon_reversal_test(seed_rows[da], seed_rows[db], feat)
                pair_results[feat] = test
            wilcoxon_results[pair_key] = pair_results
            n_sig = sum(1 for v in pair_results.values() if v.get("significant"))
            if n_sig > 0:
                print(f"  {pair_key}: {n_sig} features with significant reversal")

    polarity_table = {}
    for domain_name, rows in seed_rows.items():
        d_means = {}
        d_stds = {}
        for feat in FEATURE_NAMES:
            vals = [r[f"d_{feat}"] for r in rows if np.isfinite(r[f"d_{feat}"])]
            if vals:
                d_means[feat] = float(np.mean(vals))
                d_stds[feat] = float(np.std(vals))
            else:
                d_means[feat] = float("nan")
                d_stds[feat] = float("nan")
        polarity_table[domain_name] = {"d_mean": d_means, "d_std": d_stds}

    output = {
        "experiment": "v30_expanded_polarity",
        "n_domains": len(loaded),
        "domains": loaded,
        "seeds": args.seeds,
        "polarity_table": polarity_table,
        "reversal_matrix": {
            da: {db: rev for db, rev in inner.items()}
            for da, inner in reversal_matrix.items()
        },
        "semantic_group_analysis_5": group_analysis_5,
        "semantic_group_analysis_4": group_analysis_4,
        "wilcoxon_tests": {
            k: {f: {kk: vv for kk, vv in v.items()} for f, v in pair.items()}
            for k, pair in wilcoxon_results.items()
        },
    }
    out_path = OUTPUT_DIR / "v30_expanded_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
