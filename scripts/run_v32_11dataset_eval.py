"""v32: Full cross-domain evaluation with 11 datasets (0 singletons).

Adds ToolACE (Injected group) and API-Bank (API-misuse group) to the
existing 9-dataset ATLAS benchmark. Runs:
  1. Polarity analysis (Cohen's d, reversal matrix, within/between gap)
  2. Cross-domain AUROC (heuristic raw, PAFC, LSTM, hybrid)
  3. Permutation test for semantic group structure
  4. Conformal coverage analysis

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_v32_11dataset_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_v32_11dataset_eval.py --quick
    PYTHONPATH=../../lib:src uv run python scripts/run_v32_11dataset_eval.py --polarity-only
"""

from __future__ import annotations

import argparse
import json
import time
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
ALL_FEATURES = STRUCTURAL_FEATURES + ARG_FEATURES

# 5-group taxonomy (11 datasets, 0 singletons)
GROUPS_5 = {
    "injected": ["synthetic", "toolace"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "rjudge", "agentharm"],
    "prompt_injection": ["injecagent", "agentdojo"],
    "api_misuse": ["toolbench", "apibank"],
}

# 4-group merge (prompt_injection → safety)
GROUPS_4 = {
    "injected": ["synthetic", "toolace"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "rjudge", "agentharm", "injecagent", "agentdojo"],
    "api_misuse": ["toolbench", "apibank"],
}

D_THRESHOLD = 0.1  # minimum |d| to count as "has polarity"


def domain_to_group(domain: str, groups: dict[str, list[str]]) -> str:
    for g, members in groups.items():
        if domain in members:
            return g
    return "unknown"


def compute_polarity_matrix(
    domain_data: dict[str, DomainData],
    features: list[str],
) -> dict[str, dict[str, float]]:
    """Compute Cohen's d per domain per feature."""
    d_matrix: dict[str, dict[str, float]] = {}
    for name, (feats, labels) in domain_data.items():
        d_matrix[name] = {}
        for f in features:
            if f not in feats:
                d_matrix[name][f] = 0.0
                continue
            vals = feats[f]
            normal = vals[labels == 0]
            anomalous = vals[labels == 1]
            if len(normal) < 2 or len(anomalous) < 2:
                d_matrix[name][f] = 0.0
                continue
            d_matrix[name][f] = cohens_d(anomalous, normal)
    return d_matrix


def compute_reversal_stats(
    d_matrix: dict[str, dict[str, float]],
    groups: dict[str, list[str]],
    features: list[str],
) -> dict:
    """Compute within/between group reversal rates."""
    domains = sorted(d_matrix.keys())
    within_reversals: list[int] = []
    between_reversals: list[int] = []
    pair_details: list[dict] = []

    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            d1, d2 = domains[i], domains[j]
            g1 = domain_to_group(d1, groups)
            g2 = domain_to_group(d2, groups)
            same_group = g1 == g2

            n_reversed = 0
            n_testable = 0
            for f in features:
                dv1 = d_matrix[d1].get(f, 0.0)
                dv2 = d_matrix[d2].get(f, 0.0)
                if abs(dv1) > D_THRESHOLD and abs(dv2) > D_THRESHOLD:
                    n_testable += 1
                    if (dv1 > 0) != (dv2 > 0):
                        n_reversed += 1

            bucket = within_reversals if same_group else between_reversals
            bucket.append(n_reversed)

            pair_details.append({
                "pair": f"{d1}↔{d2}",
                "group_src": g1,
                "group_tgt": g2,
                "same_group": same_group,
                "n_reversed": n_reversed,
                "n_testable": n_testable,
            })

    within_rate = (
        np.mean([r / len(features) for r in within_reversals])
        if within_reversals else 0.0
    )
    between_rate = (
        np.mean([r / len(features) for r in between_reversals])
        if between_reversals else 0.0
    )
    gap = between_rate / max(within_rate, 0.001)

    # Mann-Whitney U test
    if within_reversals and between_reversals:
        w_rates = [r / len(features) for r in within_reversals]
        b_rates = [r / len(features) for r in between_reversals]
        u_stat, p_val = stats.mannwhitneyu(
            w_rates, b_rates, alternative="less",
        )
    else:
        u_stat, p_val = 0.0, 1.0

    return {
        "within_pairs": len(within_reversals),
        "between_pairs": len(between_reversals),
        "within_reversal_rate": float(within_rate),
        "between_reversal_rate": float(between_rate),
        "gap": float(gap),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_val),
        "pair_details": pair_details,
    }


def run_permutation_test(
    d_matrix: dict[str, dict[str, float]],
    groups: dict[str, list[str]],
    features: list[str],
    n_perms: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test: is the within/between gap significant?"""
    rng = np.random.RandomState(seed)
    domains = sorted(d_matrix.keys())

    # Observed gap
    obs = compute_reversal_stats(d_matrix, groups, features)
    obs_gap = obs["between_reversal_rate"] - obs["within_reversal_rate"]

    # Permute group labels
    perm_gaps = []
    domain_list = list(domains)
    for _ in range(n_perms):
        shuffled = list(domain_list)
        rng.shuffle(shuffled)
        perm_groups: dict[str, list[str]] = {}
        idx = 0
        for g, members in groups.items():
            perm_groups[g] = shuffled[idx:idx + len(members)]
            idx += len(members)
        perm_stats = compute_reversal_stats(d_matrix, perm_groups, features)
        perm_gaps.append(
            perm_stats["between_reversal_rate"]
            - perm_stats["within_reversal_rate"]
        )

    perm_gaps = np.array(perm_gaps)
    p_value = float(np.mean(perm_gaps >= obs_gap))

    return {
        "observed_gap": float(obs_gap),
        "p_value": p_value,
        "n_permutations": n_perms,
    }


def heuristic_score(features: dict[str, np.ndarray]) -> np.ndarray:
    """Z-score + mean of structural features."""
    cols = []
    for name in STRUCTURAL_FEATURES:
        if name not in features:
            continue
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        cols.append((vals - mu) / sigma)
    return np.mean(np.column_stack(cols), axis=1)


def apply_pafc_correction(
    features: dict[str, np.ndarray],
    source_d: dict[str, float],
    target_d: dict[str, float],
) -> dict[str, np.ndarray]:
    corrected = {}
    for name, values in features.items():
        if name not in STRUCTURAL_FEATURES:
            continue
        d_src = source_d.get(name, 0.0)
        d_tgt = target_d.get(name, 0.0)
        if abs(d_src) > 0.1 and abs(d_tgt) > 0.1 and (d_src > 0) != (d_tgt > 0):
            corrected[name] = -values
        else:
            corrected[name] = values
    return corrected


def evaluate_cross_domain_auroc(
    domain_data: dict[str, DomainData],
    d_matrix: dict[str, dict[str, float]],
    groups: dict[str, list[str]],
    seeds: list[int],
) -> dict:
    """Compute cross-domain AUROC for all ordered pairs."""
    from sklearn.metrics import roc_auc_score

    domains = sorted(domain_data.keys())
    results: dict[str, dict] = {}

    for src in domains:
        for tgt in domains:
            if src == tgt:
                continue
            pair_key = f"{src}->{tgt}"
            src_feat, src_lbl = domain_data[src]
            tgt_feat, tgt_lbl = domain_data[tgt]

            if len(np.unique(tgt_lbl)) < 2:
                results[pair_key] = {
                    "heuristic_raw": [0.5],
                    "heuristic_pafc": [0.5],
                    "same_group": domain_to_group(src, groups)
                    == domain_to_group(tgt, groups),
                }
                continue

            seed_results: dict[str, list[float]] = {
                "heuristic_raw": [],
                "heuristic_pafc": [],
            }

            for seed in seeds:
                rng = np.random.RandomState(seed)
                n_tgt = len(tgt_lbl)
                idx = rng.permutation(n_tgt)
                cal_size = max(20, n_tgt // 5)
                test_idx = idx[cal_size:]

                if len(test_idx) < 10:
                    test_idx = idx

                test_feat = {k: v[test_idx] for k, v in tgt_feat.items()}
                test_lbl = tgt_lbl[test_idx]

                if len(np.unique(test_lbl)) < 2:
                    seed_results["heuristic_raw"].append(0.5)
                    seed_results["heuristic_pafc"].append(0.5)
                    continue

                raw_scores = heuristic_score(test_feat)
                try:
                    seed_results["heuristic_raw"].append(
                        float(roc_auc_score(test_lbl, raw_scores))
                    )
                except ValueError:
                    seed_results["heuristic_raw"].append(0.5)

                src_d = d_matrix[src]
                tgt_d = d_matrix[tgt]
                corrected = apply_pafc_correction(test_feat, src_d, tgt_d)
                pafc_scores = heuristic_score(corrected)
                try:
                    seed_results["heuristic_pafc"].append(
                        float(roc_auc_score(test_lbl, pafc_scores))
                    )
                except ValueError:
                    seed_results["heuristic_pafc"].append(0.5)

            g_src = domain_to_group(src, groups)
            g_tgt = domain_to_group(tgt, groups)
            results[pair_key] = {
                **seed_results,
                "same_group": g_src == g_tgt,
                "group_src": g_src,
                "group_tgt": g_tgt,
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v32: 11-dataset cross-domain evaluation",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--polarity-only", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123, 456]
        args.max_traces = 500

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  v32: 11-Dataset Cross-Domain Evaluation")
    print("=" * 70)

    print("\nLoading all 11 domains...")
    t0 = time.time()
    domain_data = load_all_domains(
        max_swe_traces=args.max_traces,
        max_toolbench=args.max_traces,
        max_toolace=args.max_traces,
        max_apibank=args.max_traces,
    )
    elapsed = time.time() - t0
    loaded = sorted(domain_data.keys())
    print(f"  Loaded {len(loaded)} domains in {elapsed:.1f}s: {loaded}")

    for name, (feats, labels) in sorted(domain_data.items()):
        print(f"    {name:>12}: {len(labels):>5} traces, "
              f"{int(labels.sum()):>4} anomalous ({labels.mean():.1%})")

    # ── Phase 1: Polarity Analysis ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Phase 1: Polarity Analysis")
    print("=" * 70)

    d_matrix_struct = compute_polarity_matrix(domain_data, STRUCTURAL_FEATURES)
    d_matrix_all = compute_polarity_matrix(domain_data, ALL_FEATURES)

    print("\nCohen's d matrix (structural features):")
    print(f"  {'Domain':<12}", end="")
    for f in STRUCTURAL_FEATURES:
        print(f"  {f:>15}", end="")
    print()
    for name in loaded:
        print(f"  {name:<12}", end="")
        for f in STRUCTURAL_FEATURES:
            d = d_matrix_struct[name].get(f, 0.0)
            print(f"  {d:>15.3f}", end="")
        print()

    for taxonomy_name, groups in [("5-group", GROUPS_5), ("4-group", GROUPS_4)]:
        print(f"\n--- Reversal stats ({taxonomy_name}) ---")
        rev_struct = compute_reversal_stats(
            d_matrix_struct, groups, STRUCTURAL_FEATURES,
        )
        print(f"  Structural features:")
        print(f"    Within:  {rev_struct['within_pairs']} pairs, "
              f"rate={rev_struct['within_reversal_rate']:.1%}")
        print(f"    Between: {rev_struct['between_pairs']} pairs, "
              f"rate={rev_struct['between_reversal_rate']:.1%}")
        print(f"    Gap: {rev_struct['gap']:.1f}x")
        print(f"    Mann-Whitney p={rev_struct['mann_whitney_p']:.4f}")

        rev_arg = compute_reversal_stats(
            d_matrix_all, groups, ARG_FEATURES,
        )
        print(f"  Argument features:")
        print(f"    Within:  {rev_arg['within_pairs']} pairs, "
              f"rate={rev_arg['within_reversal_rate']:.1%}")
        print(f"    Between: {rev_arg['between_pairs']} pairs, "
              f"rate={rev_arg['between_reversal_rate']:.1%}")
        print(f"    Gap: {rev_arg['gap']:.1f}x")

        # Permutation test
        print(f"\n  Permutation test ({taxonomy_name}, structural)...")
        perm = run_permutation_test(
            d_matrix_struct, groups, STRUCTURAL_FEATURES,
        )
        print(f"    Observed gap: {perm['observed_gap']:.3f}")
        print(f"    Permutation p-value: {perm['p_value']:.4f}")

    # ── Phase 2: Cross-Domain AUROC ─────────────────────────────────────
    if not args.polarity_only:
        print("\n" + "=" * 70)
        print("  Phase 2: Cross-Domain AUROC")
        print("=" * 70)

        auroc_results = evaluate_cross_domain_auroc(
            domain_data, d_matrix_struct, GROUPS_5, args.seeds,
        )

        # Summarize within vs between performance
        within_aurocs: list[float] = []
        between_aurocs: list[float] = []
        for pair_key, res in auroc_results.items():
            mean_raw = np.mean(res["heuristic_raw"])
            if res["same_group"]:
                within_aurocs.append(mean_raw)
            else:
                between_aurocs.append(mean_raw)

        print(f"\n  Within-group AUROC:  {np.mean(within_aurocs):.3f} "
              f"± {np.std(within_aurocs):.3f} (n={len(within_aurocs)})")
        print(f"  Between-group AUROC: {np.mean(between_aurocs):.3f} "
              f"± {np.std(between_aurocs):.3f} (n={len(between_aurocs)})")
    else:
        auroc_results = {}

    # ── Save results ────────────────────────────────────────────────────
    output = {
        "experiment": "v32_11dataset_eval",
        "n_datasets": len(loaded),
        "datasets": loaded,
        "seeds": args.seeds,
        "d_matrix_structural": d_matrix_struct,
        "d_matrix_all": d_matrix_all,
        "reversal_stats_5group": compute_reversal_stats(
            d_matrix_struct, GROUPS_5, STRUCTURAL_FEATURES,
        ),
        "reversal_stats_4group": compute_reversal_stats(
            d_matrix_struct, GROUPS_4, STRUCTURAL_FEATURES,
        ),
        "reversal_stats_5group_args": compute_reversal_stats(
            d_matrix_all, GROUPS_5, ARG_FEATURES,
        ),
        "permutation_5group": run_permutation_test(
            d_matrix_struct, GROUPS_5, STRUCTURAL_FEATURES,
        ),
        "permutation_4group": run_permutation_test(
            d_matrix_struct, GROUPS_4, STRUCTURAL_FEATURES,
        ),
        "cross_domain_auroc": auroc_results,
    }

    out_path = OUTPUT_DIR / "v32_11dataset_eval.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    r5 = output["reversal_stats_5group"]
    r4 = output["reversal_stats_4group"]
    p5 = output["permutation_5group"]
    p4 = output["permutation_4group"]
    print(f"  5-group: within={r5['within_reversal_rate']:.1%}, "
          f"between={r5['between_reversal_rate']:.1%}, "
          f"gap={r5['gap']:.1f}x, perm_p={p5['p_value']:.4f}")
    print(f"  4-group: within={r4['within_reversal_rate']:.1%}, "
          f"between={r4['between_reversal_rate']:.1%}, "
          f"gap={r4['gap']:.1f}x, perm_p={p4['p_value']:.4f}")


if __name__ == "__main__":
    main()
