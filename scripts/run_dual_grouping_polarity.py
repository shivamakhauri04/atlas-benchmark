"""E1: Dual-grouping polarity analysis (4-group vs 5-group).

Recomputes within/between reversal rates under both grouping schemes:
  - 4-group: InjecAgent merged into Safety (matches v30 code)
  - 5-group: InjecAgent as separate Prompt-injection group (matches paper taxonomy)

Reports reversal rates, Wilcoxon tests, and a comparison table.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_dual_grouping_polarity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from data.domain_loader import DomainData, load_all_domains
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")
FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]

GROUPINGS = {
    "4group": {
        "injected": ["synthetic"],
        "failure": ["swe_agent", "trail"],
        "safety": ["atbench", "injecagent", "agentdojo", "agentharm", "rjudge"],
        "api_misuse": ["toolbench"],
    },
    "5group": {
        "injected": ["synthetic"],
        "failure": ["swe_agent", "trail"],
        "safety": ["atbench", "agentharm", "rjudge"],
        "prompt_injection": ["injecagent", "agentdojo"],
        "api_misuse": ["toolbench"],
    },
}


def compute_cohens_d_per_domain(
    domain_data: dict[str, DomainData],
) -> dict[str, dict[str, float]]:
    """Compute Cohen's d (anomalous - normal) for each domain and feature."""
    result: dict[str, dict[str, float]] = {}
    for name, (features, labels) in domain_data.items():
        normal_mask = labels == 0
        anom_mask = labels == 1
        d_vals: dict[str, float] = {}
        for feat in FEATURE_NAMES:
            if feat not in features:
                d_vals[feat] = 0.0
                continue
            d_vals[feat] = cohens_d(
                features[feat][anom_mask], features[feat][normal_mask],
            )
        result[name] = d_vals
    return result


def check_sign_reversal(d_a: float, d_b: float, threshold: float = 0.1) -> bool:
    """Check if two Cohen's d values have opposite signs (ignoring near-zero)."""
    if abs(d_a) < threshold or abs(d_b) < threshold:
        return False
    return (d_a > 0) != (d_b > 0)


def analyze_grouping(
    d_values: dict[str, dict[str, float]],
    grouping: dict[str, list[str]],
    domains: list[str],
) -> dict:
    """Compute within/between reversal stats for one grouping scheme."""
    domain_to_group: dict[str, str] = {}
    for group, members in grouping.items():
        for d in members:
            if d in domains:
                domain_to_group[d] = group

    within_reversals = 0
    within_total = 0
    between_reversals = 0
    between_total = 0
    within_pairs: list[dict] = []
    between_pairs: list[dict] = []

    for i, da in enumerate(domains):
        for db in domains[i + 1:]:
            ga = domain_to_group.get(da, "unknown")
            gb = domain_to_group.get(db, "unknown")
            n_rev = sum(
                1 for f in FEATURE_NAMES
                if check_sign_reversal(d_values[da][f], d_values[db][f])
            )
            pair_info = {
                "source": da,
                "target": db,
                "source_group": ga,
                "target_group": gb,
                "n_reversed": n_rev,
                "features": {
                    f: {
                        "d_source": d_values[da][f],
                        "d_target": d_values[db][f],
                        "reversed": check_sign_reversal(
                            d_values[da][f], d_values[db][f],
                        ),
                    }
                    for f in FEATURE_NAMES
                },
            }
            if ga == gb:
                within_reversals += n_rev
                within_total += len(FEATURE_NAMES)
                within_pairs.append(pair_info)
            else:
                between_reversals += n_rev
                between_total += len(FEATURE_NAMES)
                between_pairs.append(pair_info)

    within_rate = within_reversals / max(within_total, 1)
    between_rate = between_reversals / max(between_total, 1)
    gap_ratio = between_rate / max(within_rate, 0.001)

    return {
        "within_group": {
            "n_pairs": len(within_pairs),
            "n_feature_comparisons": within_total,
            "n_reversals": within_reversals,
            "rate": round(within_rate, 4),
            "pairs": within_pairs,
        },
        "between_group": {
            "n_pairs": len(between_pairs),
            "n_feature_comparisons": between_total,
            "n_reversals": between_reversals,
            "rate": round(between_rate, 4),
            "pairs": between_pairs,
        },
        "gap_ratio": round(gap_ratio, 1),
    }


def run_bootstrap_wilcoxon(
    domain_data: dict[str, DomainData],
    seeds: list[int],
    grouping: dict[str, list[str]],
) -> dict:
    """Run bootstrap Wilcoxon tests for sign reversal significance."""
    domains = sorted(domain_data.keys())
    domain_to_group: dict[str, str] = {}
    for group, members in grouping.items():
        for d in members:
            if d in domains:
                domain_to_group[d] = group

    boot_d: dict[str, dict[str, list[float]]] = {
        d: {f: [] for f in FEATURE_NAMES} for d in domains
    }

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for name, (features, labels) in domain_data.items():
            n = len(labels)
            idx = rng.choice(n, size=n, replace=True)
            for feat in FEATURE_NAMES:
                if feat not in features:
                    boot_d[name][feat].append(0.0)
                    continue
                vals = features[feat][idx]
                labs = labels[idx]
                d = cohens_d(vals[labs == 1], vals[labs == 0])
                boot_d[name][feat].append(d)

    n_significant = 0
    n_tested = 0
    pair_results: list[dict] = []

    for i, da in enumerate(domains):
        for db in domains[i + 1:]:
            ga = domain_to_group.get(da, "unknown")
            gb = domain_to_group.get(db, "unknown")
            if ga == gb:
                continue

            for feat in FEATURE_NAMES:
                d_a = np.array(boot_d[da][feat])
                d_b = np.array(boot_d[db][feat])
                products = d_a * d_b
                n_tested += 1
                try:
                    _, p = stats.wilcoxon(products, alternative="less")
                except ValueError:
                    p = 1.0
                significant = p < 0.05
                if significant:
                    n_significant += 1
                pair_results.append({
                    "source": da,
                    "target": db,
                    "feature": feat,
                    "p_value": round(float(p), 6),
                    "significant": significant,
                })

    return {
        "n_tested": n_tested,
        "n_significant": n_significant,
        "results": pair_results,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  E1: Dual-Grouping Polarity Analysis")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]

    print("\nLoading all domains...")
    domain_data = load_all_domains(max_swe_traces=5000)
    domains = sorted(domain_data.keys())
    print(f"  Loaded {len(domains)} domains: {domains}")

    print("\nComputing Cohen's d per domain...")
    d_values = compute_cohens_d_per_domain(domain_data)
    for name, d_dict in sorted(d_values.items()):
        vals = "  ".join(
            f"{f}: {'+' if d >= 0 else ''}{d:.3f}" for f, d in d_dict.items()
        )
        print(f"  {name:15s}: {vals}")

    results: dict[str, dict] = {}

    for grouping_name, grouping in GROUPINGS.items():
        print(f"\n{'=' * 70}")
        print(f"  Grouping: {grouping_name}")
        print(f"{'=' * 70}")

        n_groups = len(grouping)
        multi_dataset_groups = [
            g for g, members in grouping.items()
            if sum(1 for m in members if m in domains) >= 2
        ]
        print(f"  Groups: {n_groups}")
        print(f"  Multi-dataset groups: {multi_dataset_groups}")

        analysis = analyze_grouping(d_values, grouping, domains)
        w = analysis["within_group"]
        b = analysis["between_group"]

        print(f"\n  Within-group:  {w['rate']:.1%} "
              f"({w['n_reversals']}/{w['n_feature_comparisons']} "
              f"across {w['n_pairs']} pairs)")
        print(f"  Between-group: {b['rate']:.1%} "
              f"({b['n_reversals']}/{b['n_feature_comparisons']} "
              f"across {b['n_pairs']} pairs)")
        print(f"  Gap ratio:     {analysis['gap_ratio']}x")

        if w['n_pairs'] > 0:
            print(f"\n  Within-group pairs:")
            for p in w['pairs']:
                rev_feats = [
                    f for f in FEATURE_NAMES if p['features'][f]['reversed']
                ]
                status = f"REVERSED: {rev_feats}" if rev_feats else "aligned"
                print(f"    {p['source']} <-> {p['target']} "
                      f"({p['source_group']}): {status}")

        print(f"\n  Running bootstrap Wilcoxon tests ({len(seeds)} seeds)...")
        wilcoxon = run_bootstrap_wilcoxon(domain_data, seeds, grouping)
        print(f"  Significant reversals: {wilcoxon['n_significant']}"
              f"/{wilcoxon['n_tested']} pairs")

        results[grouping_name] = {
            "grouping": grouping,
            "n_groups": n_groups,
            "multi_dataset_groups": multi_dataset_groups,
            "analysis": {
                "within_group": {
                    k: v for k, v in w.items() if k != "pairs"
                },
                "between_group": {
                    k: v for k, v in b.items() if k != "pairs"
                },
                "gap_ratio": analysis["gap_ratio"],
            },
            "within_pairs_detail": w["pairs"],
            "between_pairs_detail": b["pairs"],
            "wilcoxon": {
                "n_tested": wilcoxon["n_tested"],
                "n_significant": wilcoxon["n_significant"],
            },
        }

    print(f"\n{'=' * 70}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<30s} {'4-group':>12s} {'5-group':>12s}")
    print(f"  {'-' * 54}")
    for metric_name, key_path in [
        ("Within-group pairs", "analysis.within_group.n_pairs"),
        ("Within feature comparisons", "analysis.within_group.n_feature_comparisons"),
        ("Within reversals", "analysis.within_group.n_reversals"),
        ("Within reversal rate", "analysis.within_group.rate"),
        ("Between-group pairs", "analysis.between_group.n_pairs"),
        ("Between feature comparisons", "analysis.between_group.n_feature_comparisons"),
        ("Between reversals", "analysis.between_group.n_reversals"),
        ("Between reversal rate", "analysis.between_group.rate"),
        ("Gap ratio", "analysis.gap_ratio"),
        ("Wilcoxon significant", "wilcoxon.n_significant"),
    ]:
        vals = []
        for gname in ["4group", "5group"]:
            obj = results[gname]
            for part in key_path.split("."):
                obj = obj[part]
            vals.append(obj)
        if isinstance(vals[0], float) and vals[0] < 1:
            print(f"  {metric_name:<30s} {vals[0]:>11.1%} {vals[1]:>11.1%}")
        else:
            print(f"  {metric_name:<30s} {str(vals[0]):>12s} {str(vals[1]):>12s}")

    output = {
        "experiment": "e1_dual_grouping_polarity",
        "n_domains": len(domains),
        "domains": domains,
        "seeds": seeds,
        "d_values": {
            d: {f: round(v, 4) for f, v in fv.items()}
            for d, fv in d_values.items()
        },
        "results": results,
    }
    out_path = OUTPUT_DIR / "e1_dual_grouping_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
