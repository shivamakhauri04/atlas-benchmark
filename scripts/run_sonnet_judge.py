"""Run Claude Sonnet 4.6 as LLM judge on all 17 pairs (8 within + 9 between).

Combines the pairs from run_llm_judge.py and run_llm_judge_within.py
into a single run for a stronger LLM baseline.

Usage:
    cd experiments/trustlens-auditable-agent-trust
    PYTHONPATH=../../lib:src uv run python scripts/run_sonnet_judge.py
    PYTHONPATH=../../lib:src uv run python scripts/run_sonnet_judge.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

OUTPUT_DIR = Path("output/tables")

ALL_PAIRS = [
    # Within-group (8 pairs)
    ("swe_agent", "trail"),          # Failure→Failure
    ("trail", "swe_agent"),          # Failure→Failure (reverse)
    ("atbench", "rjudge"),           # Safety→Safety
    ("rjudge", "atbench"),           # Safety→Safety (reverse)
    ("rjudge", "agentharm"),         # Safety→Safety
    ("agentharm", "rjudge"),         # Safety→Safety
    ("agentharm", "atbench"),        # Safety→Safety
    ("atbench", "agentharm"),        # Safety→Safety
    # Between-group (9 pairs)
    ("synthetic", "atbench"),        # Injected→Safety
    ("synthetic", "swe_agent"),      # Injected→Failure
    ("swe_agent", "synthetic"),      # Failure→Injected
    ("rjudge", "toolbench"),         # Safety→API
    ("toolbench", "synthetic"),      # API→Injected
    ("atbench", "swe_agent"),        # Safety→Failure
    ("swe_agent", "toolbench"),      # Failure→API
    ("injecagent", "atbench"),       # PI→Safety
    ("synthetic", "toolbench"),      # Injected→API
]

MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def main():
    parser = argparse.ArgumentParser(description="Sonnet 4.6 LLM judge — all 17 pairs")
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--quick", action="store_true", help="10 traces for smoke test")
    args = parser.parse_args()

    if args.quick:
        args.n_eval = 10
        args.n_examples = 3

    import run_llm_judge as llm
    llm._model_id = args.model

    t0 = time.time()

    print("=" * 60)
    print("  Sonnet 4.6 LLM Judge — All 17 Pairs")
    print(f"  Model: {args.model}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Traces/domain: {args.n_eval}")
    print(f"  Pairs: {len(ALL_PAIRS)} (8 within + 9 between)")
    print("=" * 60)

    needed = set()
    for s, t in ALL_PAIRS:
        needed.add(s)
        needed.add(t)

    print("\nLoading domains...")
    domain_traces = {}
    for domain in sorted(needed):
        try:
            traces = llm.load_domain_traces(domain)
            normal = sum(1 for t in traces if not t.has_anomaly)
            anom = sum(1 for t in traces if t.has_anomaly)
            if normal < 5 or anom < 5:
                print(f"  {domain}: skipping (insufficient balance)")
                continue
            domain_traces[domain] = traces
            print(f"  {domain}: {len(traces)} traces ({normal}N/{anom}A)")
        except Exception as e:
            print(f"  {domain}: FAILED ({e})")

    # Also run zero-shot on all loaded domains
    all_zero_shot: list[dict] = []
    all_few_shot: list[dict] = []

    for seed in args.seeds:
        print(f"\n{'#' * 60}")
        print(f"  SEED: {seed}")
        print(f"{'#' * 60}")

        zs = llm.run_zero_shot(domain_traces, n_eval=args.n_eval, seed=seed)
        all_zero_shot.append(zs)

        pairs = [
            (s, t) for s, t in ALL_PAIRS
            if s in domain_traces and t in domain_traces
        ]
        fs = llm.run_few_shot_cross_domain(
            domain_traces, pairs,
            n_eval=args.n_eval, n_examples=args.n_examples,
            seed=seed,
        )
        all_few_shot.append(fs)

    # Aggregate zero-shot
    all_domains_seen = set()
    for run in all_zero_shot:
        all_domains_seen.update(run.keys())

    zs_agg = {}
    for domain in sorted(all_domains_seen):
        aurocs = [run[domain]["auroc"] for run in all_zero_shot if domain in run]
        zs_agg[domain] = {
            "auroc_mean": round(np.mean(aurocs), 4),
            "auroc_std": round(np.std(aurocs), 4),
            "aurocs": [round(a, 4) for a in aurocs],
            "n_seeds": len(aurocs),
            "group": llm.DOMAIN_TO_GROUP.get(domain, "unknown"),
        }

    # Aggregate few-shot
    all_pairs_seen = set()
    for run in all_few_shot:
        all_pairs_seen.update(run.keys())

    fs_agg = {}
    for pair_key in sorted(all_pairs_seen):
        aurocs = [run[pair_key]["auroc"] for run in all_few_shot if pair_key in run]
        meta = next(run[pair_key] for run in all_few_shot if pair_key in run)
        anti_counts = sum(1 for a in aurocs if a < 0.5)
        fs_agg[pair_key] = {
            "source": meta["source"],
            "target": meta["target"],
            "source_group": meta["source_group"],
            "target_group": meta["target_group"],
            "transfer_type": meta["transfer_type"],
            "auroc_mean": round(np.mean(aurocs), 4),
            "auroc_std": round(np.std(aurocs), 4),
            "aurocs": [round(a, 4) for a in aurocs],
            "n_seeds": len(aurocs),
            "anti_predict_rate": round(anti_counts / len(aurocs), 4),
        }

    # Compute within vs between statistics
    within = {k: v for k, v in fs_agg.items() if v["transfer_type"] == "within"}
    between = {k: v for k, v in fs_agg.items() if v["transfer_type"] == "between"}

    within_means = [v["auroc_mean"] for v in within.values()]
    between_means = [v["auroc_mean"] for v in between.values()]

    # Per-seed pooled analysis (more statistical power)
    within_all_seeds = []
    between_all_seeds = []
    for run in all_few_shot:
        for pair_key, result in run.items():
            if result["transfer_type"] == "within":
                within_all_seeds.append(result["auroc"])
            else:
                between_all_seeds.append(result["auroc"])

    # Mann-Whitney on per-seed pooled AUROCs
    mw_stat, mw_p = (None, None)
    if within_all_seeds and between_all_seeds:
        mw_stat, mw_p = mannwhitneyu(
            within_all_seeds, between_all_seeds, alternative="two-sided",
        )

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print("  SONNET 4.6 RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  Zero-shot per domain:")
    for d, r in sorted(zs_agg.items(), key=lambda x: -x[1]["auroc_mean"]):
        print(f"    {d:15s}  {r['auroc_mean']:.3f}±{r['auroc_std']:.3f}")

    print(f"\n  Few-shot within-group ({len(within)} pairs):")
    for p, r in sorted(within.items(), key=lambda x: x[1]["auroc_mean"]):
        print(f"    {p:30s}  {r['auroc_mean']:.3f}±{r['auroc_std']:.3f}")
    if within_means:
        print(f"    MEAN: {np.mean(within_means):.3f}±{np.std(within_means):.3f}")

    print(f"\n  Few-shot between-group ({len(between)} pairs):")
    for p, r in sorted(between.items(), key=lambda x: x[1]["auroc_mean"]):
        print(f"    {p:30s}  {r['auroc_mean']:.3f}±{r['auroc_std']:.3f}")
    if between_means:
        print(f"    MEAN: {np.mean(between_means):.3f}±{np.std(between_means):.3f}")

    print(f"\n  Within vs Between:")
    print(f"    Within:  {np.mean(within_means):.3f}±{np.std(within_means):.3f}"
          f" (n={len(within_means)} pairs, {len(within_all_seeds)} seed-pairs)")
    print(f"    Between: {np.mean(between_means):.3f}±{np.std(between_means):.3f}"
          f" (n={len(between_means)} pairs, {len(between_all_seeds)} seed-pairs)")
    if mw_p is not None:
        print(f"    Mann-Whitney U (pooled per-seed): p={mw_p:.4f}")
    inverted = np.mean(within_means) < np.mean(between_means)
    print(f"    Pattern: {'INVERTED (within < between)' if inverted else 'STANDARD (within > between)'}")

    print(f"\n  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "within_mean": round(float(np.mean(within_means)), 4) if within_means else None,
        "within_std": round(float(np.std(within_means)), 4) if within_means else None,
        "between_mean": round(float(np.mean(between_means)), 4) if between_means else None,
        "between_std": round(float(np.std(between_means)), 4) if between_means else None,
        "mann_whitney_p": round(float(mw_p), 4) if mw_p is not None else None,
        "inverted": bool(inverted),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
        "n_within_seed_obs": len(within_all_seeds),
        "n_between_seed_obs": len(between_all_seeds),
    }

    output = {
        "model": args.model,
        "n_eval": args.n_eval,
        "n_examples": args.n_examples,
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "aggregated": {
            "zero_shot": zs_agg,
            "few_shot": fs_agg,
            "summary": summary,
        },
        "per_seed": [
            {
                "seed": s,
                "zero_shot": all_zero_shot[i],
                "few_shot": {k: v for k, v in all_few_shot[i].items()},
            }
            for i, s in enumerate(args.seeds)
        ],
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path = OUTPUT_DIR / "v31_llm_judge_sonnet_4_6.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
