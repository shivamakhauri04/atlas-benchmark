"""Run all 3 LLM judges at 200 traces per domain (scaled from 50).

Runs Sonnet 4.6, Haiku 4.5, and Llama 3.3 70B on the same 17 pairs
(8 within + 9 between) with 200 traces per domain per seed, 3 seeds.

Usage:
    cd experiments/trustlens-auditable-agent-trust
    PYTHONPATH=../../lib:src uv run python scripts/run_llm_judge_scaled.py
    PYTHONPATH=../../lib:src uv run python scripts/run_llm_judge_scaled.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

OUTPUT_DIR = Path("output/tables")

ALL_PAIRS = [
    # Within-group (8 pairs)
    ("swe_agent", "trail"),
    ("trail", "swe_agent"),
    ("atbench", "rjudge"),
    ("rjudge", "atbench"),
    ("rjudge", "agentharm"),
    ("agentharm", "rjudge"),
    ("agentharm", "atbench"),
    ("atbench", "agentharm"),
    # Between-group (9 pairs)
    ("synthetic", "atbench"),
    ("synthetic", "swe_agent"),
    ("swe_agent", "synthetic"),
    ("rjudge", "toolbench"),
    ("toolbench", "synthetic"),
    ("atbench", "swe_agent"),
    ("swe_agent", "toolbench"),
    ("injecagent", "atbench"),
    ("synthetic", "toolbench"),
]

MODELS = {
    "sonnet": "us.anthropic.claude-sonnet-4-6",
    "haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "llama": "us.meta.llama3-3-70b-instruct-v1:0",
}


def run_model(
    model_name: str,
    model_id: str,
    domain_traces: dict,
    n_eval: int,
    n_examples: int,
    seeds: list[int],
    llm_module,
) -> dict:
    """Run zero-shot + few-shot for a single model across all pairs and seeds."""
    llm_module._model_id = model_id
    print(f"\n{'#' * 60}")
    print(f"  MODEL: {model_name} ({model_id})")
    print(f"  n_eval={n_eval}, seeds={seeds}")
    print(f"{'#' * 60}")

    all_zero_shot = []
    all_few_shot = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        zs = llm_module.run_zero_shot(domain_traces, n_eval=n_eval, seed=seed)
        all_zero_shot.append(zs)

        pairs = [
            (s, t) for s, t in ALL_PAIRS
            if s in domain_traces and t in domain_traces
        ]
        fs = llm_module.run_few_shot_cross_domain(
            domain_traces, pairs,
            n_eval=n_eval, n_examples=n_examples, seed=seed,
        )
        all_few_shot.append(fs)

    return aggregate_results(model_name, model_id, all_zero_shot, all_few_shot, seeds, n_eval, n_examples, llm_module)


def aggregate_results(
    model_name: str,
    model_id: str,
    all_zero_shot: list[dict],
    all_few_shot: list[dict],
    seeds: list[int],
    n_eval: int,
    n_examples: int,
    llm_module,
) -> dict:
    """Aggregate zero-shot and few-shot results across seeds."""
    # Zero-shot aggregation
    all_domains = set()
    for run in all_zero_shot:
        all_domains.update(run.keys())

    zs_agg = {}
    for domain in sorted(all_domains):
        aurocs = [run[domain]["auroc"] for run in all_zero_shot if domain in run]
        zs_agg[domain] = {
            "auroc_mean": round(np.mean(aurocs), 4),
            "auroc_std": round(np.std(aurocs), 4),
            "aurocs": [round(a, 4) for a in aurocs],
            "group": llm_module.DOMAIN_TO_GROUP.get(domain, "unknown"),
        }

    # Few-shot aggregation
    all_pairs_seen = set()
    for run in all_few_shot:
        all_pairs_seen.update(run.keys())

    fs_agg = {}
    for pair_key in sorted(all_pairs_seen):
        aurocs = [run[pair_key]["auroc"] for run in all_few_shot if pair_key in run]
        meta = next(run[pair_key] for run in all_few_shot if pair_key in run)
        fs_agg[pair_key] = {
            "source": meta["source"],
            "target": meta["target"],
            "source_group": meta["source_group"],
            "target_group": meta["target_group"],
            "transfer_type": meta["transfer_type"],
            "auroc_mean": round(np.mean(aurocs), 4),
            "auroc_std": round(np.std(aurocs), 4),
            "aurocs": [round(a, 4) for a in aurocs],
            "anti_predict_rate": round(sum(1 for a in aurocs if a < 0.5) / len(aurocs), 4),
        }

    # Within vs between
    within = {k: v for k, v in fs_agg.items() if v["transfer_type"] == "within"}
    between = {k: v for k, v in fs_agg.items() if v["transfer_type"] == "between"}

    within_means = [v["auroc_mean"] for v in within.values()]
    between_means = [v["auroc_mean"] for v in between.values()]

    within_all = []
    between_all = []
    for run in all_few_shot:
        for pair_key, result in run.items():
            if result["transfer_type"] == "within":
                within_all.append(result["auroc"])
            else:
                between_all.append(result["auroc"])

    mw_p = None
    if within_all and between_all:
        _, mw_p = mannwhitneyu(within_all, between_all, alternative="two-sided")

    summary = {
        "within_mean": round(float(np.mean(within_means)), 4) if within_means else None,
        "within_std": round(float(np.std(within_means)), 4) if within_means else None,
        "between_mean": round(float(np.mean(between_means)), 4) if between_means else None,
        "between_std": round(float(np.std(between_means)), 4) if between_means else None,
        "mann_whitney_p": round(float(mw_p), 4) if mw_p is not None else None,
        "inverted": bool(np.mean(within_means) < np.mean(between_means)) if within_means and between_means else None,
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
        "n_within_seed_obs": len(within_all),
        "n_between_seed_obs": len(between_all),
    }

    # Print
    print(f"\n  {model_name} SUMMARY:")
    print(f"    Within:  {summary['within_mean']:.3f}±{summary['within_std']:.3f} ({len(within)} pairs)")
    print(f"    Between: {summary['between_mean']:.3f}±{summary['between_std']:.3f} ({len(between)} pairs)")
    if mw_p is not None:
        print(f"    Mann-Whitney p={mw_p:.4f}")
    print(f"    Pattern: {'INVERTED' if summary['inverted'] else 'STANDARD'}")

    return {
        "model": model_id,
        "model_name": model_name,
        "n_eval": n_eval,
        "n_examples": n_examples,
        "seeds": seeds,
        "zero_shot": zs_agg,
        "few_shot": fs_agg,
        "summary": summary,
        "per_seed_zero_shot": all_zero_shot,
        "per_seed_few_shot": [{k: v for k, v in run.items()} for run in all_few_shot],
    }


def main():
    parser = argparse.ArgumentParser(description="Scaled LLM judge — 200 traces, 3 models")
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--quick", action="store_true", help="10 traces for smoke test")
    args = parser.parse_args()

    if args.quick:
        args.n_eval = 10
        args.n_examples = 3

    sys.path.insert(0, "scripts")
    import run_llm_judge as llm

    t0 = time.time()

    print("=" * 60)
    print("  SCALED LLM JUDGE EVALUATION")
    print(f"  Models: {args.models}")
    print(f"  Traces/domain: {args.n_eval}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Pairs: {len(ALL_PAIRS)} (8 within + 9 between)")
    print("=" * 60)

    # Load domains once (shared across models)
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

    # Run each model
    all_model_results = {}
    for model_name in args.models:
        model_id = MODELS[model_name]
        result = run_model(
            model_name, model_id, domain_traces,
            args.n_eval, args.n_examples, args.seeds, llm,
        )
        all_model_results[model_name] = result

    elapsed = time.time() - t0

    # Final comparison
    print(f"\n{'=' * 60}")
    print("  FINAL COMPARISON (200 traces/domain)")
    print(f"{'=' * 60}")
    for name, res in all_model_results.items():
        s = res["summary"]
        print(f"  {name:8s}: within {s['within_mean']:.3f}±{s['within_std']:.3f} | "
              f"between {s['between_mean']:.3f}±{s['between_std']:.3f} | "
              f"p={s['mann_whitney_p']:.4f} | "
              f"{'INVERTED' if s['inverted'] else 'STANDARD'}")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "scaled_llm_judge_200_traces",
        "n_eval": args.n_eval,
        "n_examples": args.n_examples,
        "seeds": args.seeds,
        "models": all_model_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    model_suffix = "_".join(args.models)
    out_path = OUTPUT_DIR / f"v31_llm_judge_scaled_200_{model_suffix}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
