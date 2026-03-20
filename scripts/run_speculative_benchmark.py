"""E15: Speculative vs conservative execution benchmark.

Compares speculative execution (sandbox + conformal rollback) against
conservative execution (block-first, human review) strategies.

Validates Theorem 6: E[Regret_spec] <= E[Regret_conservative]
when P(anomaly) < theta*, where theta* = C_block / (C_block + C_rollback).

Key metrics:
  - Total regret comparison
  - Rollback rate (speculative) vs false-block rate (conservative)
  - Latency simulation (p50, p95, p99)
  - Empirical validation of theta* crossover point

Usage:
    python scripts/run_speculative_benchmark.py
    python scripts/run_speculative_benchmark.py --seeds 42,123,456,789,1024
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from torch.utils.data import DataLoader

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from trustlens.conformal import split_conformal_calibrate
from trustlens.speculative import (
    DualThresholdConfig,
    SpeculativeConfig,
    compute_theta_star,
    compute_theta_star_dual,
    evaluate_speculative_vs_conservative,
    speculative_latency_simulation,
    sweep_anomaly_rate,
)
from trustlens.stats import bootstrap_ci

from run_cp_comparison import (
    compute_consequence_scores,
    train_consequence_encoder_for_seed,
)


def compute_per_step_scores(
    model: torch.nn.Module,
    dataset: TraceDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Compute per-step scores AND per-trace scores for all traces.

    Returns:
        per_step_scores: List of (T_i,) arrays, one per trace.
        per_step_labels: List of (T_i,) arrays, one per trace.
        trace_scores: (N,) per-trace mean NLL.
        trace_labels: (N,) per-trace binary labels.
    """
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_traces,
        num_workers=0,
    )

    all_per_step_scores = []
    all_per_step_labels = []
    all_trace_scores = []
    all_trace_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]
            lengths = batch["lengths"]

            outcome_hashes = None
            if model.use_outcome:
                outcome_hashes = batch["outcome_hashes"].to(device)

            # Per-step NLL
            if model.use_outcome and outcome_hashes is not None:
                nll = model.compute_joint_nll(
                    tool_ids, args_hashes, mask, outcome_hashes
                )
            else:
                nll = model.compute_nll(tool_ids, args_hashes, mask, outcome_hashes)

            nll_np = nll.cpu().numpy()
            labels_np = labels.numpy()
            lengths_np = lengths.numpy()

            # Extract per-trace data
            for i in range(len(lengths_np)):
                length = int(lengths_np[i])
                step_scores = nll_np[i, :length]
                step_labels = labels_np[i, :length]
                all_per_step_scores.append(step_scores)
                all_per_step_labels.append(step_labels)

                trace_score = float(step_scores.sum() / max(length, 1))
                trace_label = int(np.any(step_labels == 1))
                all_trace_scores.append(trace_score)
                all_trace_labels.append(trace_label)

    return (
        all_per_step_scores,
        all_per_step_labels,
        np.array(all_trace_scores),
        np.array(all_trace_labels),
    )


def run_speculative_for_seed(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    vocab: TraceVocab,
    device,
    alpha_cons: float,
    alpha_spec: float,
    encoder_epochs: int,
    outcome_weight: float,
    cost_configs: list[SpeculativeConfig],
    alpha_spec_sweep: list[float] | None = None,
) -> dict:
    """Run speculative vs conservative comparison for one seed.

    v12: Dual-threshold design. Conservative uses tau_cons (calibrated at
    alpha_cons, strict). Speculative uses tau_spec (calibrated at alpha_spec,
    permissive). This tests the real thesis: speculative can afford a more
    permissive threshold because rollback provides a safety net.
    """
    set_seeds(seed)

    model = train_consequence_encoder_for_seed(
        seed,
        train_traces,
        vocab,
        device,
        epochs=encoder_epochs,
        outcome_weight=outcome_weight,
    )

    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    cal_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_traces],
        dtype=np.int32,
    )

    # Calibrate thresholds on per-step max scores (not trace means).
    cal_per_step, _, _, _ = compute_per_step_scores(model, cal_ds, device)
    cal_max_step = np.array([float(s.max()) for s in cal_per_step])
    cal_normal_max = cal_max_step[cal_labels == 0]

    # Dual calibration: strict for conservative, permissive for speculative
    tau_cons = split_conformal_calibrate(cal_normal_max, alpha=alpha_cons)
    tau_spec = split_conformal_calibrate(cal_normal_max, alpha=alpha_spec)

    # Get per-step scores for test set
    per_step_scores, per_step_labels, trace_scores, trace_labels = (
        compute_per_step_scores(model, test_ds, device)
    )

    anomaly_rate = float(trace_labels.mean())

    # Run comparison for each cost configuration with dual thresholds
    comparisons = []
    for config in cost_configs:
        config_with_tau = SpeculativeConfig(
            tau=tau_spec,
            max_steps=config.max_steps,
            rollback_cost=config.rollback_cost,
            block_cost=config.block_cost,
            miss_cost=config.miss_cost,
            commit_reward=config.commit_reward,
        )
        result = evaluate_speculative_vs_conservative(
            per_step_scores,
            per_step_labels,
            trace_scores,
            trace_labels,
            tau_spec,
            config_with_tau,
            tau_cons=tau_cons,
        )
        result["config_label"] = (
            f"Rb={config.rollback_cost},Bl={config.block_cost},"
            f"Ms={config.miss_cost},Cr={config.commit_reward}"
        )
        comparisons.append(result)

    # Latency simulation (uses tau_spec for speculative)
    latency = speculative_latency_simulation(per_step_scores, tau_spec)

    # Anomaly rate sweep — test near and above theta*
    config_no_reward = SpeculativeConfig(
        tau=tau_spec, rollback_cost=1.0, block_cost=2.0,
        miss_cost=10.0, commit_reward=0.0,
    )
    theta_star = compute_theta_star(config_no_reward)
    sweep_rates = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    sweep_rates = sorted(set(r for r in sweep_rates if 0 < r < 1))
    rng = np.random.default_rng(seed)
    rate_sweep = sweep_anomaly_rate(
        per_step_scores, per_step_labels, trace_labels,
        tau_spec, config_no_reward, sweep_rates, rng,
        tau_cons=tau_cons,
    )

    # Economic model (commit_reward=1) — reported separately
    config_with_reward = SpeculativeConfig(
        tau=tau_spec, rollback_cost=1.0, block_cost=2.0,
        miss_cost=10.0, commit_reward=1.0,
    )
    rng2 = np.random.default_rng(seed)
    rate_sweep_economic = sweep_anomaly_rate(
        per_step_scores, per_step_labels, trace_labels,
        tau_spec, config_with_reward, sweep_rates, rng2,
        tau_cons=tau_cons,
    )

    # Alpha-spec sweep: vary speculative permissiveness
    alpha_sweep_results = []
    if alpha_spec_sweep:
        for a_spec in alpha_spec_sweep:
            tau_s = split_conformal_calibrate(cal_normal_max, alpha=a_spec)
            config_sweep = SpeculativeConfig(
                tau=tau_s, rollback_cost=1.0, block_cost=2.0,
                miss_cost=10.0, commit_reward=0.0,
            )
            rng_sw = np.random.default_rng(seed)
            sweep_res = sweep_anomaly_rate(
                per_step_scores, per_step_labels, trace_labels,
                tau_s, config_sweep, sweep_rates, rng_sw,
                tau_cons=tau_cons,
            )
            alpha_sweep_results.append({
                "alpha_spec": a_spec,
                "tau_spec": float(tau_s),
                "rate_sweep": sweep_res,
            })

    return {
        "seed": seed,
        "tau_spec": float(tau_spec),
        "tau_cons": float(tau_cons),
        "alpha_spec": alpha_spec,
        "alpha_cons": alpha_cons,
        "anomaly_rate": anomaly_rate,
        "comparisons": comparisons,
        "latency": latency,
        "rate_sweep": rate_sweep,
        "rate_sweep_economic": rate_sweep_economic,
        "alpha_sweep": alpha_sweep_results,
    }


def aggregate_speculative_results(all_results: list[dict]) -> dict:
    """Aggregate per-seed speculative results."""
    # Use first cost config as default for headline numbers
    default_comps = [r["comparisons"][0] for r in all_results]

    spec_regrets = np.array([c["speculative"]["total_regret"] for c in default_comps])
    cons_regrets = np.array([c["conservative"]["total_regret"] for c in default_comps])
    regret_diffs = np.array([c["regret_difference"] for c in default_comps])
    spec_wins = np.array([c["speculative_wins_empirical"] for c in default_comps])
    rollback_rates = np.array([c["speculative"]["rollback_rate"] for c in default_comps])
    false_block_rates = np.array(
        [c["conservative"]["false_block_rate"] for c in default_comps]
    )

    # Latency
    spec_means = np.array([r["latency"]["speculative_mean_ms"] for r in all_results])
    cons_means = np.array([r["latency"]["conservative_mean_ms"] for r in all_results])
    speedups = np.array([r["latency"]["speedup_mean"] for r in all_results])

    agg = {
        "default_config": {
            "speculative_regret": {
                "mean": float(spec_regrets.mean()),
                "std": float(spec_regrets.std()),
                **bootstrap_ci(spec_regrets),
            },
            "conservative_regret": {
                "mean": float(cons_regrets.mean()),
                "std": float(cons_regrets.std()),
                **bootstrap_ci(cons_regrets),
            },
            "regret_difference": {
                "mean": float(regret_diffs.mean()),
                "std": float(regret_diffs.std()),
                **bootstrap_ci(regret_diffs),
            },
            "speculative_wins_fraction": float(spec_wins.mean()),
            "rollback_rate": {
                "mean": float(rollback_rates.mean()),
                "std": float(rollback_rates.std()),
            },
            "false_block_rate": {
                "mean": float(false_block_rates.mean()),
                "std": float(false_block_rates.std()),
            },
        },
        "latency": {
            "speculative_mean_ms": {
                "mean": float(spec_means.mean()),
                "std": float(spec_means.std()),
            },
            "conservative_mean_ms": {
                "mean": float(cons_means.mean()),
                "std": float(cons_means.std()),
            },
            "speedup": {
                "mean": float(speedups.mean()),
                "std": float(speedups.std()),
            },
        },
        "n_seeds": len(all_results),
    }

    # Aggregate rate sweep across seeds
    if all_results and "rate_sweep" in all_results[0]:
        # Collect all target rates
        all_rates = set()
        for r in all_results:
            for pt in r.get("rate_sweep", []):
                all_rates.add(pt["target_rate"])

        rate_sweep_agg = []
        for target_rate in sorted(all_rates):
            spec_regs = []
            cons_regs = []
            wins = []
            for r in all_results:
                for pt in r.get("rate_sweep", []):
                    if abs(pt["target_rate"] - target_rate) < 1e-6:
                        spec_regs.append(pt["speculative_regret"])
                        cons_regs.append(pt["conservative_regret"])
                        wins.append(pt["speculative_wins"])
            if spec_regs:
                rate_sweep_agg.append({
                    "target_rate": target_rate,
                    "speculative_regret_mean": float(np.mean(spec_regs)),
                    "conservative_regret_mean": float(np.mean(cons_regs)),
                    "speculative_wins_fraction": float(np.mean(wins)),
                    "regret_diff_mean": float(
                        np.mean(np.array(spec_regs) - np.array(cons_regs))
                    ),
                    "n_seeds": len(spec_regs),
                })
        agg["rate_sweep"] = rate_sweep_agg

    # Aggregate economic rate sweep (commit_reward=1)
    if all_results and "rate_sweep_economic" in all_results[0]:
        all_rates_econ = set()
        for r in all_results:
            for pt in r.get("rate_sweep_economic", []):
                all_rates_econ.add(pt["target_rate"])

        rate_sweep_econ_agg = []
        for target_rate in sorted(all_rates_econ):
            spec_regs = []
            cons_regs = []
            wins = []
            for r in all_results:
                for pt in r.get("rate_sweep_economic", []):
                    if abs(pt["target_rate"] - target_rate) < 1e-6:
                        spec_regs.append(pt["speculative_regret"])
                        cons_regs.append(pt["conservative_regret"])
                        wins.append(pt["speculative_wins"])
            if spec_regs:
                rate_sweep_econ_agg.append({
                    "target_rate": target_rate,
                    "speculative_regret_mean": float(np.mean(spec_regs)),
                    "conservative_regret_mean": float(np.mean(cons_regs)),
                    "speculative_wins_fraction": float(np.mean(wins)),
                    "regret_diff_mean": float(
                        np.mean(np.array(spec_regs) - np.array(cons_regs))
                    ),
                    "n_seeds": len(spec_regs),
                })
        agg["rate_sweep_economic"] = rate_sweep_econ_agg

    # Aggregate alpha_spec sweep across seeds
    if all_results and all_results[0].get("alpha_sweep"):
        alpha_vals = [
            entry["alpha_spec"] for entry in all_results[0]["alpha_sweep"]
        ]
        alpha_sweep_agg = []
        for a_spec in alpha_vals:
            per_rate_data: dict[float, list[dict]] = {}
            for r in all_results:
                for entry in r.get("alpha_sweep", []):
                    if abs(entry["alpha_spec"] - a_spec) < 1e-6:
                        for pt in entry["rate_sweep"]:
                            rate = pt["target_rate"]
                            per_rate_data.setdefault(rate, []).append(pt)

            rate_agg = []
            for rate in sorted(per_rate_data):
                pts = per_rate_data[rate]
                rate_agg.append({
                    "target_rate": rate,
                    "speculative_regret_mean": float(
                        np.mean([p["speculative_regret"] for p in pts])
                    ),
                    "conservative_regret_mean": float(
                        np.mean([p["conservative_regret"] for p in pts])
                    ),
                    "speculative_wins_fraction": float(
                        np.mean([p["speculative_wins"] for p in pts])
                    ),
                })
            alpha_sweep_agg.append({
                "alpha_spec": a_spec,
                "rate_sweep": rate_agg,
            })
        agg["alpha_sweep"] = alpha_sweep_agg

    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E15: Speculative vs conservative execution (v12 dual-threshold)"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument(
        "--alpha-cons", type=float, default=0.05,
        help="Alpha for conservative threshold (strict, default=0.05)",
    )
    p.add_argument(
        "--alpha-spec", type=float, default=0.10,
        help="Alpha for speculative threshold (permissive, default=0.10)",
    )
    p.add_argument(
        "--alpha-spec-sweep", type=str, default="0.10,0.15,0.20,0.30",
        help="Comma-separated alpha_spec values for sweep",
    )
    p.add_argument("--encoder-epochs", type=int, default=30)
    p.add_argument("--outcome-weight", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    alpha_spec_sweep = [float(x) for x in args.alpha_spec_sweep.split(",")]
    device = get_device(args.device)

    print(f"E15: Speculative vs Conservative (v12 dual-threshold) on {device}")
    print(f"Seeds: {seeds}, alpha_cons={args.alpha_cons}, alpha_spec={args.alpha_spec}")
    print(f"Alpha-spec sweep: {alpha_spec_sweep}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    print(f"Train: {len(train_traces)}, Cal: {len(cal_traces)}, Test: {len(test_traces)}")

    cost_configs = [
        SpeculativeConfig(
            tau=0.0, rollback_cost=1.0, block_cost=2.0,
            miss_cost=10.0, commit_reward=0.0,
        ),
        SpeculativeConfig(
            tau=0.0, rollback_cost=1.0, block_cost=2.0,
            miss_cost=10.0, commit_reward=1.0,
        ),
        SpeculativeConfig(
            tau=0.0, rollback_cost=0.5, block_cost=2.0,
            miss_cost=10.0, commit_reward=0.0,
        ),
        SpeculativeConfig(
            tau=0.0, rollback_cost=2.0, block_cost=2.0,
            miss_cost=10.0, commit_reward=0.0,
        ),
        SpeculativeConfig(
            tau=0.0, rollback_cost=1.0, block_cost=5.0,
            miss_cost=10.0, commit_reward=0.0,
        ),
    ]

    print("\nTheoretical crossover points (theta*):")
    for cfg in cost_configs:
        ts = compute_theta_star(cfg)
        print(
            f"  Rb={cfg.rollback_cost}, Bl={cfg.block_cost}: "
            f"theta*={ts:.3f}"
        )

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        result = run_speculative_for_seed(
            seed,
            train_traces,
            cal_traces,
            test_traces,
            vocab,
            device,
            alpha_cons=args.alpha_cons,
            alpha_spec=args.alpha_spec,
            encoder_epochs=args.encoder_epochs,
            outcome_weight=args.outcome_weight,
            cost_configs=cost_configs,
            alpha_spec_sweep=alpha_spec_sweep,
        )

        c0 = result["comparisons"][0]
        print(
            f"  Anomaly rate: {result['anomaly_rate']:.3f}, "
            f"tau_spec: {result['tau_spec']:.4f}, "
            f"tau_cons: {result['tau_cons']:.4f}"
        )
        print(
            f"  Speculative regret: {c0['speculative']['total_regret']:.2f}, "
            f"Conservative regret: {c0['conservative']['total_regret']:.2f}"
        )
        print(
            f"  Spec wins: {c0['speculative_wins_empirical']}, "
            f"Rollback rate: {c0['speculative']['rollback_rate']:.3f}"
        )
        print(
            f"  Latency: spec={result['latency']['speculative_mean_ms']:.1f}ms, "
            f"cons={result['latency']['conservative_mean_ms']:.1f}ms, "
            f"speedup={result['latency']['speedup_mean']:.2f}x"
        )
        all_results.append(result)

    # Aggregate
    agg = aggregate_speculative_results(all_results)

    print("\n" + "=" * 80)
    print("E15: SPECULATIVE vs CONSERVATIVE (AGGREGATE)")
    print("=" * 80)

    dc = agg["default_config"]
    print(f"\nDefault config (Rb=1, Bl=2, Ms=10):")
    print(
        f"  Speculative regret: "
        f"{dc['speculative_regret']['mean']:.2f} +/- {dc['speculative_regret']['std']:.2f}"
    )
    print(
        f"  Conservative regret: "
        f"{dc['conservative_regret']['mean']:.2f} +/- {dc['conservative_regret']['std']:.2f}"
    )
    print(
        f"  Regret difference (spec - cons): "
        f"{dc['regret_difference']['mean']:.2f} +/- {dc['regret_difference']['std']:.2f}"
    )
    print(f"  Speculative wins: {dc['speculative_wins_fraction'] * 100:.0f}% of seeds")
    print(
        f"  Rollback rate: {dc['rollback_rate']['mean']:.3f} +/- {dc['rollback_rate']['std']:.3f}"
    )

    lt = agg["latency"]
    print(f"\nLatency simulation:")
    print(
        f"  Speculative mean: {lt['speculative_mean_ms']['mean']:.1f} ms"
    )
    print(
        f"  Conservative mean: {lt['conservative_mean_ms']['mean']:.1f} ms"
    )
    print(f"  Speedup: {lt['speedup']['mean']:.2f}x")

    # Print rate sweep (error-only, commit_reward=0)
    if "rate_sweep" in agg:
        theta_star = compute_theta_star(
            SpeculativeConfig(tau=0.0, rollback_cost=1.0, block_cost=2.0,
                              miss_cost=10.0, commit_reward=0.0)
        )
        print(f"\nAnomaly rate sweep — ERROR-ONLY (commit_reward=0, theta*={theta_star:.3f}):")
        print(f"  {'Rate':>6} | {'Spec Regret':>12} | {'Cons Regret':>12} | {'Spec Wins':>10}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for pt in agg["rate_sweep"]:
            marker = " <-- theta*" if abs(pt["target_rate"] - theta_star) < 0.01 else ""
            print(
                f"  {pt['target_rate']:6.3f} | "
                f"{pt['speculative_regret_mean']:12.2f} | "
                f"{pt['conservative_regret_mean']:12.2f} | "
                f"{pt['speculative_wins_fraction']*100:9.0f}%{marker}"
            )

    # Print rate sweep (economic, commit_reward=1)
    if "rate_sweep_economic" in agg:
        print(f"\nAnomaly rate sweep — ECONOMIC (commit_reward=1):")
        print(f"  {'Rate':>6} | {'Spec Regret':>12} | {'Cons Regret':>12} | {'Spec Wins':>10}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        for pt in agg["rate_sweep_economic"]:
            print(
                f"  {pt['target_rate']:6.3f} | "
                f"{pt['speculative_regret_mean']:12.2f} | "
                f"{pt['conservative_regret_mean']:12.2f} | "
                f"{pt['speculative_wins_fraction']*100:9.0f}%"
            )

    # Save results
    output = {
        "config": {
            "seeds": seeds,
            "alpha_cons": args.alpha_cons,
            "alpha_spec": args.alpha_spec,
            "alpha_spec_sweep": alpha_spec_sweep,
            "encoder_epochs": args.encoder_epochs,
            "outcome_weight": args.outcome_weight,
            "n_seeds": len(seeds),
            "cost_configs": [
                {
                    "rollback_cost": c.rollback_cost,
                    "block_cost": c.block_cost,
                    "miss_cost": c.miss_cost,
                    "commit_reward": c.commit_reward,
                }
                for c in cost_configs
            ],
        },
        "aggregate": agg,
        "per_seed": all_results,
    }
    out_path = tables_dir / "speculative_execution.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
