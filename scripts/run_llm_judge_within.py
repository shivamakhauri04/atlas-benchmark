"""Supplementary LLM judge: expanded within-group pairs only.

Runs additional within-group few-shot pairs to strengthen the
"inverted transfer" claim. Reuses functions from run_llm_judge.py.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_llm_judge_within.py \
        --model us.anthropic.claude-haiku-4-5-20251001-v1:0 --seeds 42 123 456
    PYTHONPATH=../../lib:src uv run python scripts/run_llm_judge_within.py \
        --model meta.llama3-3-70b-instruct-v1:0 --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("output/tables")

WITHIN_PAIRS = [
    ("trail", "swe_agent"),        # Failure→Failure (reverse)
    ("rjudge", "atbench"),         # Safety→Safety (reverse)
    ("rjudge", "agentharm"),       # Safety→Safety
    ("agentharm", "rjudge"),       # Safety→Safety
    ("agentharm", "atbench"),      # Safety→Safety
    ("atbench", "agentharm"),      # Safety→Safety
]


def main():
    parser = argparse.ArgumentParser(
        description="Expanded within-group LLM judge pairs",
    )
    parser.add_argument("--n-eval", type=int, default=50)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    args = parser.parse_args()

    # Import from main script (shares call_bedrock, loaders, etc.)
    import run_llm_judge as llm
    llm._model_id = args.model

    t0 = time.time()

    print("=" * 60)
    print("  Expanded Within-Group LLM Judge")
    print(f"  Model: {args.model}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Pairs: {len(WITHIN_PAIRS)}")
    print("=" * 60)

    # Load domains needed for within-group pairs
    needed = set()
    for s, t in WITHIN_PAIRS:
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

    all_few_shot: list[dict] = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        pairs = [
            (s, t) for s, t in WITHIN_PAIRS
            if s in domain_traces and t in domain_traces
        ]
        fs = llm.run_few_shot_cross_domain(
            domain_traces, pairs,
            n_eval=args.n_eval, n_examples=args.n_examples,
            seed=seed,
        )
        all_few_shot.append(fs)

    # Aggregate
    all_pairs_seen = set()
    for run in all_few_shot:
        all_pairs_seen.update(run.keys())

    agg = {}
    for pair_key in sorted(all_pairs_seen):
        aurocs = [
            run[pair_key]["auroc"]
            for run in all_few_shot if pair_key in run
        ]
        meta = next(run[pair_key] for run in all_few_shot if pair_key in run)
        anti_counts = sum(1 for a in aurocs if a < 0.5)
        agg[pair_key] = {
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

    elapsed = time.time() - t0

    # Print summary
    within_aurocs = [v["auroc_mean"] for v in agg.values()]
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {len(agg)} within-group pairs")
    for pair, r in sorted(agg.items(), key=lambda x: x[1]["auroc_mean"]):
        print(f"    {pair:25s}  {r['auroc_mean']:.3f}±{r['auroc_std']:.3f}")
    if within_aurocs:
        print(f"\n  Mean within-group AUROC: {np.mean(within_aurocs):.3f}"
              f"±{np.std(within_aurocs):.3f}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.split(".")[-1].split("-v")[0].replace("-", "_")
    out_path = OUTPUT_DIR / f"v31_llm_judge_within_{model_slug}.json"

    output = {
        "model": args.model,
        "n_eval": args.n_eval,
        "n_examples": args.n_examples,
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "pairs_evaluated": [list(p) for p in WITHIN_PAIRS],
        "aggregated": agg,
        "per_seed": [
            {"seed": s, "few_shot": fs}
            for s, fs in zip(args.seeds, all_few_shot)
        ],
        "elapsed_seconds": round(elapsed, 1),
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
