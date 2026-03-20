"""Base-rate confound analysis for polarity reversal.

Tests whether anomaly base rate differences alone explain polarity reversal,
or whether the semantic grouping adds explanatory power beyond base rate.

Analyses:
1. Logistic regression: reversal ~ |base_rate_src - base_rate_tgt|
2. Logistic regression: reversal ~ |base_rate_diff| + same_group
3. Partial correlation: reversal vs same_group, controlling for base_rate_diff
4. Stratified analysis: reversal rates within base-rate-similar vs dissimilar pairs
5. Per-feature breakdown
"""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

DATASET_BASE_RATES = {
    "synthetic": 0.143,
    "swe_agent": 0.798,
    "trail": 0.203,
    "atbench": 0.501,
    "rjudge": 0.510,
    "agentharm": 0.500,
    "injecagent": 0.992,
    "agentdojo": 0.578,
    "toolbench": 0.098,
}

SEMANTIC_GROUPS_5 = {
    "synthetic": "injected",
    "swe_agent": "failure",
    "trail": "failure",
    "atbench": "safety",
    "rjudge": "safety",
    "agentharm": "safety",
    "injecagent": "prompt_injection",
    "agentdojo": "prompt_injection",
    "toolbench": "api_misuse",
}

SEMANTIC_GROUPS_4 = {
    "synthetic": "injected",
    "swe_agent": "failure",
    "trail": "failure",
    "atbench": "safety",
    "rjudge": "safety",
    "agentharm": "safety",
    "injecagent": "safety",
    "agentdojo": "safety",
    "toolbench": "api_misuse",
}

STRUCTURAL_FEATURES = ["trace_length", "low_diversity", "repeat_rate"]


def load_polarity_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_pair_table(polarity_data: dict) -> list[dict]:
    """Build a table of all unordered dataset pairs with reversal and base-rate info."""
    reversal_matrix = polarity_data["reversal_matrix"]
    domains = sorted(reversal_matrix.keys())
    rows = []

    for d1, d2 in itertools.combinations(domains, 2):
        base_rate_diff = abs(DATASET_BASE_RATES[d1] - DATASET_BASE_RATES[d2])
        same_group_5 = SEMANTIC_GROUPS_5[d1] == SEMANTIC_GROUPS_5[d2]
        same_group_4 = SEMANTIC_GROUPS_4[d1] == SEMANTIC_GROUPS_4[d2]

        reversals_structural = []
        for feat in STRUCTURAL_FEATURES:
            rev = reversal_matrix[d1][d2][feat]
            reversals_structural.append(rev)

        n_reversed = sum(reversals_structural)
        any_reversed = n_reversed > 0
        frac_reversed = n_reversed / len(STRUCTURAL_FEATURES)

        rows.append({
            "d1": d1,
            "d2": d2,
            "base_rate_diff": base_rate_diff,
            "base_rate_d1": DATASET_BASE_RATES[d1],
            "base_rate_d2": DATASET_BASE_RATES[d2],
            "same_group_5": same_group_5,
            "same_group_4": same_group_4,
            "n_reversed": n_reversed,
            "any_reversed": any_reversed,
            "frac_reversed": frac_reversed,
            "reversals": {f: r for f, r in zip(STRUCTURAL_FEATURES, reversals_structural)},
        })

    return rows


def analysis_1_baserate_vs_reversal(rows: list[dict]) -> dict:
    """Point-biserial correlation: base_rate_diff vs any_reversed."""
    base_diffs = np.array([r["base_rate_diff"] for r in rows])
    reversed_binary = np.array([int(r["any_reversed"]) for r in rows])

    rho_pb, p_pb = stats.pointbiserialr(reversed_binary, base_diffs)

    rho_spearman, p_spearman = stats.spearmanr(base_diffs, [r["frac_reversed"] for r in rows])

    return {
        "point_biserial_r": float(rho_pb),
        "point_biserial_p": float(p_pb),
        "spearman_rho": float(rho_spearman),
        "spearman_p": float(p_spearman),
        "n_pairs": len(rows),
    }


def analysis_2_group_vs_baserate(rows: list[dict]) -> dict:
    """Compare predictive power: same_group vs base_rate_diff for reversal."""
    results = {}

    for grouping in ["same_group_4", "same_group_5"]:
        same = np.array([int(r[grouping]) for r in rows])
        reversed_bin = np.array([int(r["any_reversed"]) for r in rows])
        base_diffs = np.array([r["base_rate_diff"] for r in rows])

        rho_group, p_group = stats.pointbiserialr(reversed_bin, same)

        # Partial correlation: reversal vs same_group, controlling for base_rate_diff
        # Using residual method
        # Regress reversed on base_rate_diff, get residuals
        slope_rb, intercept_rb, _, _, _ = stats.linregress(base_diffs, reversed_bin.astype(float))
        resid_reversed = reversed_bin - (slope_rb * base_diffs + intercept_rb)

        slope_gb, intercept_gb, _, _, _ = stats.linregress(base_diffs, same.astype(float))
        resid_group = same - (slope_gb * base_diffs + intercept_gb)

        rho_partial, p_partial = stats.pearsonr(resid_reversed, resid_group)

        results[grouping] = {
            "rho_group_reversal": float(rho_group),
            "p_group_reversal": float(p_group),
            "partial_r_group_given_baserate": float(rho_partial),
            "partial_p_group_given_baserate": float(p_partial),
        }

    return results


def analysis_3_stratified(rows: list[dict]) -> dict:
    """Stratify pairs by base-rate similarity, check reversal within strata."""
    base_diffs = [r["base_rate_diff"] for r in rows]
    median_diff = float(np.median(base_diffs))

    strata = {"similar_baserate": [], "different_baserate": []}
    for r in rows:
        if r["base_rate_diff"] <= median_diff:
            strata["similar_baserate"].append(r)
        else:
            strata["different_baserate"].append(r)

    results = {"median_baserate_diff": median_diff}
    for stratum_name, stratum_rows in strata.items():
        n = len(stratum_rows)
        within = [r for r in stratum_rows if r["same_group_4"]]
        between = [r for r in stratum_rows if not r["same_group_4"]]

        within_rev = sum(r["frac_reversed"] for r in within) / len(within) if within else float("nan")
        between_rev = sum(r["frac_reversed"] for r in between) / len(between) if between else float("nan")

        within_any = sum(r["any_reversed"] for r in within) / len(within) if within else float("nan")
        between_any = sum(r["any_reversed"] for r in between) / len(between) if between else float("nan")

        results[stratum_name] = {
            "n_pairs": n,
            "n_within": len(within),
            "n_between": len(between),
            "within_frac_reversed": float(within_rev),
            "between_frac_reversed": float(between_rev),
            "within_any_reversed_rate": float(within_any),
            "between_any_reversed_rate": float(between_any),
            "gap": float(between_rev / within_rev) if within_rev > 0 else float("inf"),
        }

    return results


def analysis_4_per_feature(rows: list[dict]) -> dict:
    """Per-feature: does base_rate_diff predict reversal?"""
    results = {}
    for feat in STRUCTURAL_FEATURES:
        reversed_feat = np.array([int(r["reversals"][feat]) for r in rows])
        base_diffs = np.array([r["base_rate_diff"] for r in rows])
        same_4 = np.array([int(r["same_group_4"]) for r in rows])

        if reversed_feat.sum() == 0 or reversed_feat.sum() == len(reversed_feat):
            results[feat] = {"note": f"no variance in reversal (all={reversed_feat[0]})"}
            continue

        rho_base, p_base = stats.pointbiserialr(reversed_feat, base_diffs)
        rho_group, p_group = stats.pointbiserialr(reversed_feat, same_4)

        results[feat] = {
            "n_reversed": int(reversed_feat.sum()),
            "n_total": len(reversed_feat),
            "reversal_rate": float(reversed_feat.mean()),
            "r_baserate_diff": float(rho_base),
            "p_baserate_diff": float(p_base),
            "r_same_group": float(rho_group),
            "p_same_group": float(p_group),
        }

    return results


def analysis_5_multivariate(rows: list[dict]) -> dict:
    """Simple OLS: frac_reversed ~ base_rate_diff + same_group."""
    from numpy.linalg import lstsq

    y = np.array([r["frac_reversed"] for r in rows])
    results = {}

    for grouping in ["same_group_4", "same_group_5"]:
        base_diffs = np.array([r["base_rate_diff"] for r in rows])
        same = np.array([int(r[grouping]) for r in rows])

        # Model 1: frac_reversed ~ base_rate_diff
        X1 = np.column_stack([np.ones(len(rows)), base_diffs])
        beta1, res1, _, _ = lstsq(X1, y, rcond=None)
        ss_res1 = float(np.sum((y - X1 @ beta1) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_baserate = 1 - ss_res1 / ss_tot if ss_tot > 0 else 0

        # Model 2: frac_reversed ~ same_group
        X2 = np.column_stack([np.ones(len(rows)), same.astype(float)])
        beta2, res2, _, _ = lstsq(X2, y, rcond=None)
        ss_res2 = float(np.sum((y - X2 @ beta2) ** 2))
        r2_group = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0

        # Model 3: frac_reversed ~ base_rate_diff + same_group
        X3 = np.column_stack([np.ones(len(rows)), base_diffs, same.astype(float)])
        beta3, res3, _, _ = lstsq(X3, y, rcond=None)
        ss_res3 = float(np.sum((y - X3 @ beta3) ** 2))
        r2_both = 1 - ss_res3 / ss_tot if ss_tot > 0 else 0

        # Incremental R^2 of group over baserate
        delta_r2 = r2_both - r2_baserate

        results[grouping] = {
            "r2_baserate_only": float(r2_baserate),
            "r2_group_only": float(r2_group),
            "r2_both": float(r2_both),
            "delta_r2_group_over_baserate": float(delta_r2),
            "beta_intercept": float(beta3[0]),
            "beta_baserate_diff": float(beta3[1]),
            "beta_same_group": float(beta3[2]),
        }

    return results


def analysis_6_matched_baserate_pairs(rows: list[dict]) -> dict:
    """Find pairs with similar base rates but different groups, and vice versa."""
    # Pairs with similar base rates (diff < 0.15) but different groups
    similar_br_diff_group = [
        r for r in rows
        if r["base_rate_diff"] < 0.15 and not r["same_group_4"]
    ]
    # Pairs with different base rates (diff > 0.30) but same group
    diff_br_same_group = [
        r for r in rows
        if r["base_rate_diff"] > 0.30 and r["same_group_4"]
    ]

    def summarize(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0, "note": "no pairs in this category"}
        return {
            "n": len(subset),
            "pairs": [(r["d1"], r["d2"], f"br_diff={r['base_rate_diff']:.3f}", f"rev={r['n_reversed']}/3") for r in subset],
            "mean_frac_reversed": float(np.mean([r["frac_reversed"] for r in subset])),
            "mean_base_rate_diff": float(np.mean([r["base_rate_diff"] for r in subset])),
        }

    return {
        "similar_baserate_different_group": summarize(similar_br_diff_group),
        "different_baserate_same_group": summarize(diff_br_same_group),
        "interpretation": (
            "If base rate drives reversal, similar_br_diff_group should have LOW reversal "
            "and diff_br_same_group should have HIGH reversal. "
            "If semantic group drives reversal, the opposite pattern should hold."
        ),
    }


def main() -> None:
    output_dir = Path("output/tables")
    polarity_path = output_dir / "v30_expanded_polarity.json"

    polarity_data = load_polarity_data(str(polarity_path))
    rows = build_pair_table(polarity_data)

    print(f"Built {len(rows)} unordered dataset pairs")
    print(f"Base rates: {DATASET_BASE_RATES}")
    print()

    # Analysis 1
    print("=" * 60)
    print("ANALYSIS 1: Base-rate difference vs reversal")
    print("=" * 60)
    a1 = analysis_1_baserate_vs_reversal(rows)
    print(f"  Point-biserial r(any_reversed, base_rate_diff) = {a1['point_biserial_r']:.3f}, p = {a1['point_biserial_p']:.3f}")
    print(f"  Spearman rho(frac_reversed, base_rate_diff) = {a1['spearman_rho']:.3f}, p = {a1['spearman_p']:.3f}")
    print()

    # Analysis 2
    print("=" * 60)
    print("ANALYSIS 2: Group membership vs base-rate as reversal predictors")
    print("=" * 60)
    a2 = analysis_2_group_vs_baserate(rows)
    for grouping, vals in a2.items():
        print(f"  [{grouping}]")
        print(f"    r(reversed, same_group) = {vals['rho_group_reversal']:.3f}, p = {vals['p_group_reversal']:.3f}")
        print(f"    Partial r(reversed, same_group | base_rate_diff) = {vals['partial_r_group_given_baserate']:.3f}, p = {vals['partial_p_group_given_baserate']:.3f}")
    print()

    # Analysis 3
    print("=" * 60)
    print("ANALYSIS 3: Stratified by base-rate similarity")
    print("=" * 60)
    a3 = analysis_3_stratified(rows)
    print(f"  Median base-rate diff: {a3['median_baserate_diff']:.3f}")
    for stratum, vals in a3.items():
        if stratum == "median_baserate_diff":
            continue
        print(f"  [{stratum}] (n={vals['n_pairs']}, within={vals['n_within']}, between={vals['n_between']})")
        print(f"    Within-group frac_reversed: {vals['within_frac_reversed']:.3f}")
        print(f"    Between-group frac_reversed: {vals['between_frac_reversed']:.3f}")
        print(f"    Gap: {vals['gap']:.1f}x")
    print()

    # Analysis 4
    print("=" * 60)
    print("ANALYSIS 4: Per-feature base-rate vs group prediction")
    print("=" * 60)
    a4 = analysis_4_per_feature(rows)
    for feat, vals in a4.items():
        if "note" in vals:
            print(f"  {feat}: {vals['note']}")
            continue
        print(f"  {feat}: reversed {vals['n_reversed']}/{vals['n_total']} ({vals['reversal_rate']:.1%})")
        print(f"    r(reversed, base_rate_diff) = {vals['r_baserate_diff']:.3f}, p = {vals['p_baserate_diff']:.3f}")
        print(f"    r(reversed, same_group) = {vals['r_same_group']:.3f}, p = {vals['p_same_group']:.3f}")
    print()

    # Analysis 5
    print("=" * 60)
    print("ANALYSIS 5: OLS R^2 comparison")
    print("=" * 60)
    a5 = analysis_5_multivariate(rows)
    for grouping, vals in a5.items():
        print(f"  [{grouping}]")
        print(f"    R^2 (base_rate_diff only): {vals['r2_baserate_only']:.3f}")
        print(f"    R^2 (same_group only): {vals['r2_group_only']:.3f}")
        print(f"    R^2 (both): {vals['r2_both']:.3f}")
        print(f"    Delta R^2 (group over baserate): {vals['delta_r2_group_over_baserate']:.3f}")
        print(f"    Betas: intercept={vals['beta_intercept']:.3f}, baserate_diff={vals['beta_baserate_diff']:.3f}, same_group={vals['beta_same_group']:.3f}")
    print()

    # Analysis 6
    print("=" * 60)
    print("ANALYSIS 6: Matched-condition pairs")
    print("=" * 60)
    a6 = analysis_6_matched_baserate_pairs(rows)
    for condition, vals in a6.items():
        if condition == "interpretation":
            print(f"  Interpretation: {vals}")
            continue
        print(f"  [{condition}]")
        if vals.get("n", 0) == 0:
            print(f"    {vals.get('note', 'no pairs')}")
        else:
            print(f"    n={vals['n']}, mean_frac_reversed={vals['mean_frac_reversed']:.3f}, mean_br_diff={vals['mean_base_rate_diff']:.3f}")
            for pair in vals["pairs"]:
                print(f"      {pair}")
    print()

    # Save all results
    all_results = {
        "analysis_1_baserate_correlation": a1,
        "analysis_2_group_vs_baserate": a2,
        "analysis_3_stratified": a3,
        "analysis_4_per_feature": a4,
        "analysis_5_ols_comparison": a5,
        "analysis_6_matched_pairs": a6,
    }

    out_path = output_dir / "v31_baserate_confound_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
