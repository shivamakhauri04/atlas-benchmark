"""Merge original and expanded LLM judge results and run significance test.

Reads the original v31_llm_judge*.json and expanded v31_llm_judge_within*.json
files, merges within-group pairs, and runs Mann-Whitney U test comparing
within-group vs between-group AUROC distributions.

Usage:
    uv run python scripts/analyze_llm_within.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

OUTPUT_DIR = Path("output/tables")


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def merge_and_analyze(model_name: str, original_path: Path, within_path: Path):
    """Merge original + expanded within-group results and test significance."""
    original = load_json(original_path)
    expanded = load_json(within_path)

    if original is None:
        print(f"  {model_name}: no original results at {original_path}")
        return

    # Gather all few-shot pairs from original
    orig_fs = original["aggregated"]["few_shot"]
    within_pairs = {
        k: v for k, v in orig_fs.items() if v["transfer_type"] == "within"
    }
    between_pairs = {
        k: v for k, v in orig_fs.items() if v["transfer_type"] == "between"
    }

    # Add expanded within-group pairs
    if expanded is not None:
        exp_fs = expanded["aggregated"]
        for k, v in exp_fs.items():
            if v["transfer_type"] == "within" and k not in within_pairs:
                within_pairs[k] = v
    else:
        print(f"  {model_name}: no expanded results at {within_path}")

    # Extract AUROC arrays
    within_aurocs = [v["auroc_mean"] for v in within_pairs.values()]
    between_aurocs = [v["auroc_mean"] for v in between_pairs.values()]

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")

    print(f"\n  Within-group pairs ({len(within_pairs)}):")
    for k, v in sorted(within_pairs.items(), key=lambda x: x[1]["auroc_mean"]):
        print(f"    {k:30s}  {v['auroc_mean']:.3f}±{v.get('auroc_std', 0):.3f}")

    print(f"\n  Between-group pairs ({len(between_pairs)}):")
    for k, v in sorted(between_pairs.items(), key=lambda x: x[1]["auroc_mean"]):
        print(f"    {k:30s}  {v['auroc_mean']:.3f}±{v.get('auroc_std', 0):.3f}")

    within_mean = np.mean(within_aurocs)
    between_mean = np.mean(between_aurocs)
    within_std = np.std(within_aurocs)
    between_std = np.std(between_aurocs)

    print(f"\n  Within mean:  {within_mean:.3f} ± {within_std:.3f}")
    print(f"  Between mean: {between_mean:.3f} ± {between_std:.3f}")
    print(f"  Delta:        {within_mean - between_mean:+.3f}")
    print(f"  Direction:    {'within < between (INVERTED)' if within_mean < between_mean else 'within >= between'}")

    # Mann-Whitney U test (one-sided: within < between)
    if len(within_aurocs) >= 2 and len(between_aurocs) >= 2:
        u_stat, p_two = stats.mannwhitneyu(
            within_aurocs, between_aurocs, alternative="two-sided",
        )
        _, p_less = stats.mannwhitneyu(
            within_aurocs, between_aurocs, alternative="less",
        )
        print(f"\n  Mann-Whitney U test:")
        print(f"    U = {u_stat}, p (two-sided) = {p_two:.4f}")
        print(f"    p (one-sided, within < between) = {p_less:.4f}")

        # Effect size (rank-biserial correlation)
        n1, n2 = len(within_aurocs), len(between_aurocs)
        r = 1 - (2 * u_stat) / (n1 * n2)
        print(f"    Rank-biserial r = {r:.3f}")
    else:
        print(f"\n  Insufficient pairs for Mann-Whitney test")

    # Per-seed analysis (using individual seed AUROCs if available)
    print(f"\n  Per-seed within-group AUROCs (all seeds pooled):")
    all_within_per_seed = []
    all_between_per_seed = []

    for v in within_pairs.values():
        if "aurocs" in v:
            all_within_per_seed.extend(v["aurocs"])
    for v in between_pairs.values():
        if "aurocs" in v:
            all_between_per_seed.extend(v["aurocs"])

    if all_within_per_seed and all_between_per_seed:
        print(f"    Within:  n={len(all_within_per_seed)}, "
              f"mean={np.mean(all_within_per_seed):.3f}±{np.std(all_within_per_seed):.3f}")
        print(f"    Between: n={len(all_between_per_seed)}, "
              f"mean={np.mean(all_between_per_seed):.3f}±{np.std(all_between_per_seed):.3f}")

        u2, p2_two = stats.mannwhitneyu(
            all_within_per_seed, all_between_per_seed, alternative="two-sided",
        )
        _, p2_less = stats.mannwhitneyu(
            all_within_per_seed, all_between_per_seed, alternative="less",
        )
        print(f"    Mann-Whitney (per-seed pooled): U={u2}, "
              f"p(two)={p2_two:.4f}, p(less)={p2_less:.4f}")

    return {
        "model": model_name,
        "n_within": len(within_pairs),
        "n_between": len(between_pairs),
        "within_mean": round(within_mean, 4),
        "within_std": round(within_std, 4),
        "between_mean": round(between_mean, 4),
        "between_std": round(between_std, 4),
        "delta": round(within_mean - between_mean, 4),
        "inverted": within_mean < between_mean,
    }


def main():
    models = [
        (
            "Claude Haiku 4.5",
            OUTPUT_DIR / "v31_llm_judge.json",
            OUTPUT_DIR / "v31_llm_judge_within_claude_haiku_4_5_20251001.json",
        ),
        (
            "Llama 3.3 70B",
            OUTPUT_DIR / "v31_llm_judge_llama3_3_70b_instruct.json",
            OUTPUT_DIR / "v31_llm_judge_within_llama3_3_70b_instruct.json",
        ),
    ]

    results = []
    for name, orig, exp in models:
        r = merge_and_analyze(name, orig, exp)
        if r:
            results.append(r)

    # Cross-model summary
    if len(results) == 2:
        print(f"\n{'=' * 60}")
        print(f"  CROSS-MODEL SUMMARY")
        print(f"{'=' * 60}")
        for r in results:
            direction = "INVERTED" if r["inverted"] else "normal"
            print(f"  {r['model']:20s}: within={r['within_mean']:.3f}, "
                  f"between={r['between_mean']:.3f}, "
                  f"delta={r['delta']:+.3f} ({direction})")

    # Save merged analysis
    out_path = OUTPUT_DIR / "v31_llm_judge_merged_analysis.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
