"""E24: RAC evaluation on SWE-agent — real-world three-way decisions.

Addresses M5 (RAC evaluated only on synthetic) by running the Risk-Averse
Calibration two-threshold evaluation on SWE-agent trajectories.

Scorers: LSTM (trained on SWE-agent) + heuristic baselines.
Key question: Does the defer zone add value with weak-to-moderate scorers
on real-world data?

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_rac_swe_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_rac_swe_eval.py --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds

# Reuse SWE-agent data loading and model infrastructure
from run_swe_agent_eval import (
    build_vocabulary,
    compute_ncs,
    compute_simple_baselines,
    load_swe_agent_data,
    train_scorer,
)

from trustlens.rac import evaluate_rac, rac_sensitivity_sweep
from trustlens.stats import bootstrap_ci

OUTPUT_DIR = Path("output/tables")


def run_rac_seed(
    traces: list[dict],
    vocab: dict[str, int],
    seed: int,
    epochs: int,
    device: torch.device,
    alpha: float,
    d_model: int,
    cost_ratios: list[float],
    defer_budgets: list[float],
) -> dict:
    """Run RAC evaluation for one seed on SWE-agent data."""
    set_seeds(seed)
    rng = np.random.RandomState(seed)

    solved = [t for t in traces if t["solved"]]
    failed = [t for t in traces if not t["solved"]]
    rng.shuffle(solved)
    rng.shuffle(failed)

    n_s, n_f = len(solved), len(failed)
    n_train_s, n_cal_s = int(0.6 * n_s), int(0.2 * n_s)
    n_train_f, n_cal_f = int(0.6 * n_f), int(0.2 * n_f)

    train_traces = solved[:n_train_s] + failed[:n_train_f]
    cal_traces = solved[n_train_s : n_train_s + n_cal_s] + failed[n_train_f : n_train_f + n_cal_f]
    test_traces = solved[n_train_s + n_cal_s :] + failed[n_train_f + n_cal_f :]

    rng.shuffle(train_traces)
    rng.shuffle(cal_traces)
    rng.shuffle(test_traces)

    cal_labels = np.array([0 if t["solved"] else 1 for t in cal_traces])
    test_labels = np.array([0 if t["solved"] else 1 for t in test_traces])

    train_normal = [t for t in train_traces if t["solved"]]
    print(
        f"\n  Seed {seed}: train={len(train_traces)} "
        f"({len(train_normal)} normal for LSTM), "
        f"cal={len(cal_traces)} ({(cal_labels == 0).sum()} normal), "
        f"test={len(test_traces)} ({(test_labels == 0).sum()} normal)"
    )

    # --- LSTM ---
    model = train_scorer(train_normal, vocab, device, epochs=epochs, d_model=d_model)
    cal_scores = compute_ncs(model, cal_traces, vocab, device)
    test_scores = compute_ncs(model, test_traces, vocab, device)

    lstm_rac = evaluate_rac(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        cost_ratio=5.0,
        defer_budget=0.15,
    )
    lstm_sweep = rac_sensitivity_sweep(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        cost_ratios=cost_ratios,
        defer_budgets=defer_budgets,
    )

    # --- Heuristic baselines ---
    bl_cal = compute_simple_baselines(cal_traces)
    bl_test = compute_simple_baselines(test_traces)

    baseline_rac = {}
    baseline_sweeps = {}
    for name in bl_test:
        bl_result = evaluate_rac(
            bl_cal[name],
            cal_labels,
            bl_test[name],
            test_labels,
            alpha=alpha,
            cost_ratio=5.0,
            defer_budget=0.15,
        )
        bl_sweep = rac_sensitivity_sweep(
            bl_cal[name],
            cal_labels,
            bl_test[name],
            test_labels,
            alpha=alpha,
            cost_ratios=cost_ratios,
            defer_budgets=defer_budgets,
        )
        baseline_rac[name] = bl_result
        baseline_sweeps[name] = bl_sweep

    return {
        "seed": seed,
        "lstm_rac": lstm_rac,
        "lstm_sweep": lstm_sweep,
        "baseline_rac": baseline_rac,
        "baseline_sweeps": baseline_sweeps,
        "split_sizes": {
            "train": len(train_traces),
            "cal": len(cal_traces),
            "test": len(test_traces),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E24: RAC evaluation on SWE-agent",
    )
    parser.add_argument("--max-traces", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024],
    )
    parser.add_argument(
        "--cost-ratios",
        type=float,
        nargs="+",
        default=[2.0, 5.0, 10.0],
    )
    parser.add_argument(
        "--defer-budgets",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    t0 = time.time()
    traces = load_swe_agent_data(args.max_traces)
    vocab = build_vocabulary(traces)

    results = []
    for seed in args.seeds:
        r = run_rac_seed(
            traces,
            vocab,
            seed,
            args.epochs,
            device,
            args.alpha,
            d_model=args.d_model,
            cost_ratios=args.cost_ratios,
            defer_budgets=args.defer_budgets,
        )
        results.append(r)

        lr = r["lstm_rac"]
        print(
            f"  -> LSTM RAC: VaR={lr['var_cost']:.4f}, "
            f"binary_VaR={lr['binary_var_cost']:.4f}, "
            f"improvement={lr['var_improvement']:.4f}, "
            f"defer={lr['defer_rate']:.3f}"
        )
        for bn, bv in r["baseline_rac"].items():
            print(
                f"     {bn} RAC: VaR={bv['var_cost']:.4f}, "
                f"improvement={bv['var_improvement']:.4f}, "
                f"defer={bv['defer_rate']:.3f}"
            )

    # --- Aggregate ---
    lstm_vars = [r["lstm_rac"]["var_cost"] for r in results]
    lstm_binary = [r["lstm_rac"]["binary_var_cost"] for r in results]
    lstm_improvements = [r["lstm_rac"]["var_improvement"] for r in results]
    lstm_defers = [r["lstm_rac"]["defer_rate"] for r in results]

    bl_names = list(results[0]["baseline_rac"].keys())
    bl_agg = {}
    for bn in bl_names:
        bl_vars = [r["baseline_rac"][bn]["var_cost"] for r in results]
        bl_imps = [r["baseline_rac"][bn]["var_improvement"] for r in results]
        bl_defs = [r["baseline_rac"][bn]["defer_rate"] for r in results]
        bl_agg[bn] = {
            "var_cost_mean": float(np.mean(bl_vars)),
            "var_improvement_mean": float(np.mean(bl_imps)),
            "var_improvement_ci": bootstrap_ci(np.array(bl_imps)),
            "defer_rate_mean": float(np.mean(bl_defs)),
        }

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"RAC on SWE-agent ({len(args.seeds)} seeds)")
    print(f"{'=' * 60}")
    print("LSTM:")
    print(f"  VaR cost:     {np.mean(lstm_vars):.4f} +/- {np.std(lstm_vars):.4f}")
    print(f"  Binary VaR:   {np.mean(lstm_binary):.4f} +/- {np.std(lstm_binary):.4f}")
    print(f"  Improvement:  {np.mean(lstm_improvements):.4f} +/- {np.std(lstm_improvements):.4f}")
    print(f"  Defer rate:   {np.mean(lstm_defers):.3f}")

    for bn, bv in bl_agg.items():
        print(f"\n{bn}:")
        print(f"  VaR cost:     {bv['var_cost_mean']:.4f}")
        print(f"  Improvement:  {bv['var_improvement_mean']:.4f}")
        print(f"  Defer rate:   {bv['defer_rate_mean']:.3f}")

    print(f"\nElapsed: {elapsed:.0f}s")

    # --- Save ---
    output = {
        "config": {
            "dataset": "nebius/SWE-agent-trajectories",
            "max_traces": args.max_traces,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "alpha": args.alpha,
            "seeds": args.seeds,
            "cost_ratios": args.cost_ratios,
            "defer_budgets": args.defer_budgets,
        },
        "aggregate": {
            "lstm": {
                "var_cost": {
                    "mean": float(np.mean(lstm_vars)),
                    "std": float(np.std(lstm_vars)),
                    **bootstrap_ci(np.array(lstm_vars)),
                },
                "binary_var_cost": {
                    "mean": float(np.mean(lstm_binary)),
                    "std": float(np.std(lstm_binary)),
                },
                "var_improvement": {
                    "mean": float(np.mean(lstm_improvements)),
                    "std": float(np.std(lstm_improvements)),
                    **bootstrap_ci(np.array(lstm_improvements)),
                },
                "defer_rate_mean": float(np.mean(lstm_defers)),
            },
        },
        "baselines": bl_agg,
        "per_seed": [
            {
                "seed": r["seed"],
                "lstm_rac": r["lstm_rac"],
                "baseline_rac": r["baseline_rac"],
                "split_sizes": r["split_sizes"],
            }
            for r in results
        ],
        "elapsed_seconds": elapsed,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rac_swe_eval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
