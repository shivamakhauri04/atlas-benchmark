"""v16 Reframing evaluation: simple baselines on synthetic + RAC with honest scorers.

Produces evidence for the paper pivot:
  "When Do Coverage Guarantees Hold? An Empirical Study of CP for Agent Monitoring"

Key experiments:
  1. Simple heuristic baselines (trace_length, low_diversity, repeat_rate) on SYNTHETIC data
     → Shows learned encoders dominate in-domain (complementing v15 where they fail cross-domain)
  2. RAC with honest scorers (LSTM seq-only, simple baselines) instead of CQ encoder
     → Evaluates RAC without leakage contamination
  3. Unified scorer comparison table across both in-domain and cross-domain

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_reframing_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_reframing_eval.py --encoder-epochs 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import f1_score, roc_auc_score

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.rac import evaluate_rac, rac_sensitivity_sweep
from trustlens.stats import bootstrap_ci

# Reuse training/scoring from run_cp_comparison
from run_cp_comparison import (
    compute_encoder_scores,
    evaluate_cp,
    train_encoder_for_seed,
)

OUTPUT_DIR = Path("output/tables")


# ---------------------------------------------------------------------------
# Simple heuristic scorers for synthetic Trace objects
# ---------------------------------------------------------------------------


def compute_simple_scores_synthetic(traces: list) -> dict[str, np.ndarray]:
    """Compute simple heuristic anomaly scores for synthetic Trace objects.

    These are the same heuristics that beat the LSTM on SWE-agent (v15),
    adapted for the canonical Trace schema.

    Returns dict mapping scorer name -> (n_traces,) score array.
    """
    n = len(traces)
    length_scores = np.zeros(n)
    diversity_scores = np.zeros(n)
    repeat_scores = np.zeros(n)

    for i, trace in enumerate(traces):
        tools = [a.tool for a in trace.actions]
        n_actions = len(tools)
        length_scores[i] = float(n_actions)
        unique_tools = len(set(tools))
        diversity_scores[i] = 1.0 - (unique_tools / max(n_actions, 1))
        if n_actions > 1:
            repeats = sum(
                1 for j in range(1, n_actions) if tools[j] == tools[j - 1]
            )
            repeat_scores[i] = repeats / (n_actions - 1)

    return {
        "trace_length": length_scores,
        "low_diversity": diversity_scores,
        "repeat_rate": repeat_scores,
    }


# ---------------------------------------------------------------------------
# Split CP evaluation helper
# ---------------------------------------------------------------------------


def evaluate_cp_simple(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
    normal_only_cal: bool = True,
) -> dict:
    """Run Split CP and compute metrics. Mirrors evaluate_cp from run_cp_comparison."""
    if normal_only_cal:
        cal_for_cp = cal_scores[cal_labels == 0]
    else:
        cal_for_cp = cal_scores

    tau = split_conformal_calibrate(cal_for_cp, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n = len(test_labels)
    n_pos = int(test_labels.sum())
    n_neg = n - n_pos

    coverage = float((preds[test_labels == 0] == 0).sum() / max(n_neg, 1))
    recall = float((preds[test_labels == 1] == 1).sum() / max(n_pos, 1))
    f1 = float(f1_score(test_labels, preds, zero_division=0.0))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = float("nan")

    n_cal = len(cal_for_cp)
    bound = alpha + 1.0 / (n_cal + 1)
    miscoverage = float(preds[test_labels == 0].mean())

    return {
        "alpha": alpha,
        "tau": float(tau),
        "auroc": auroc,
        "f1": f1,
        "recall": recall,
        "coverage": coverage,
        "miscoverage": miscoverage,
        "theoretical_bound": float(bound),
        "bound_holds": bool(miscoverage <= bound + 1e-9),
        "n_cal": n_cal,
        "n_test": n,
        "n_anomalous": n_pos,
        "calibration": "normal_only" if normal_only_cal else "mixed",
    }


# ---------------------------------------------------------------------------
# Per-seed evaluation
# ---------------------------------------------------------------------------


def run_single_seed(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    vocab: TraceVocab,
    device,
    alpha: float,
    encoder_epochs: int,
    cost_ratios: list[float],
    defer_budgets: list[float],
) -> dict:
    """Run all scorers + RAC for one seed on synthetic data."""
    set_seeds(seed)

    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    results: dict = {"seed": seed, "cp_results": {}, "rac_results": {}}

    # --- 1. Transformer Encoder (seq-only, no args) ---
    print(f"  Training Transformer encoder (seed={seed})...")
    model = train_encoder_for_seed(
        seed,
        train_traces,
        vocab,
        device,
        epochs=encoder_epochs,
        use_args=False,
        d_model=128,
    )
    enc_cal, _ = compute_encoder_scores(model, cal_ds, device)
    enc_test, _ = compute_encoder_scores(model, test_ds, device)

    r = evaluate_cp_simple(enc_cal, cal_labels, enc_test, test_labels, alpha)
    results["cp_results"]["transformer_seq_only"] = r
    print(f"    Transformer: AUROC={r['auroc']:.4f}, Cov={r['coverage']:.4f}")

    # RAC with Transformer
    rac_r = evaluate_rac(
        enc_cal, cal_labels, enc_test, test_labels,
        alpha=alpha, cost_ratio=5.0, defer_budget=0.15, normal_only_cal=True,
    )
    results["rac_results"]["transformer_seq_only"] = rac_r
    results["rac_results"]["transformer_seq_only_sweep"] = rac_sensitivity_sweep(
        enc_cal, cal_labels, enc_test, test_labels,
        alpha=alpha, cost_ratios=cost_ratios, defer_budgets=defer_budgets,
        normal_only_cal=True,
    )

    # --- 2. LSTM Next-Token ---
    print(f"  Training LSTM scorer (seed={seed})...")
    from baselines.all_baselines import LSTMNextTokenBaseline

    set_seeds(seed)
    lstm_bl = LSTMNextTokenBaseline()
    lstm_bl.fit_traces(train_traces, cal_traces, vocab)
    lstm_cal = lstm_bl.score_traces(cal_traces, vocab)
    lstm_test = lstm_bl.score_traces(test_traces, vocab)

    r = evaluate_cp_simple(lstm_cal, cal_labels, lstm_test, test_labels, alpha)
    results["cp_results"]["lstm_next_token"] = r
    print(f"    LSTM: AUROC={r['auroc']:.4f}, Cov={r['coverage']:.4f}")

    # RAC with LSTM
    rac_r = evaluate_rac(
        lstm_cal, cal_labels, lstm_test, test_labels,
        alpha=alpha, cost_ratio=5.0, defer_budget=0.15, normal_only_cal=True,
    )
    results["rac_results"]["lstm_next_token"] = rac_r
    results["rac_results"]["lstm_next_token_sweep"] = rac_sensitivity_sweep(
        lstm_cal, cal_labels, lstm_test, test_labels,
        alpha=alpha, cost_ratios=cost_ratios, defer_budgets=defer_budgets,
        normal_only_cal=True,
    )

    # --- 3. Simple heuristic baselines ---
    print(f"  Computing simple baselines (seed={seed})...")
    cal_simple = compute_simple_scores_synthetic(cal_traces)
    test_simple = compute_simple_scores_synthetic(test_traces)

    for name in ["trace_length", "low_diversity", "repeat_rate"]:
        r = evaluate_cp_simple(
            cal_simple[name], cal_labels,
            test_simple[name], test_labels,
            alpha,
        )
        results["cp_results"][name] = r
        print(f"    {name}: AUROC={r['auroc']:.4f}, Cov={r['coverage']:.4f}")

        # RAC with simple baseline
        rac_r = evaluate_rac(
            cal_simple[name], cal_labels,
            test_simple[name], test_labels,
            alpha=alpha, cost_ratio=5.0, defer_budget=0.15, normal_only_cal=True,
        )
        results["rac_results"][name] = rac_r
        results["rac_results"][f"{name}_sweep"] = rac_sensitivity_sweep(
            cal_simple[name], cal_labels,
            test_simple[name], test_labels,
            alpha=alpha, cost_ratios=cost_ratios, defer_budgets=defer_budgets,
            normal_only_cal=True,
        )

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(all_results: list[dict]) -> dict:
    """Aggregate per-seed results into mean +/- std with bootstrap CIs."""
    scorer_names = list(all_results[0]["cp_results"].keys())
    n_seeds = len(all_results)

    cp_agg = {}
    for sn in scorer_names:
        aurocs = np.array([r["cp_results"][sn]["auroc"] for r in all_results])
        coverages = np.array([r["cp_results"][sn]["coverage"] for r in all_results])
        f1s = np.array([r["cp_results"][sn]["f1"] for r in all_results])
        bounds = np.array([r["cp_results"][sn]["bound_holds"] for r in all_results])

        cp_agg[sn] = {
            "auroc": {
                "mean": float(aurocs.mean()),
                "std": float(aurocs.std()),
                **bootstrap_ci(aurocs),
            },
            "coverage": {
                "mean": float(coverages.mean()),
                "std": float(coverages.std()),
                **bootstrap_ci(coverages),
            },
            "f1": {
                "mean": float(f1s.mean()),
                "std": float(f1s.std()),
                **bootstrap_ci(f1s),
            },
            "bound_holds_pct": float(bounds.mean() * 100),
        }

    # RAC aggregation (default settings only)
    rac_agg = {}
    rac_scorer_names = [
        k for k in all_results[0]["rac_results"].keys()
        if not k.endswith("_sweep")
    ]
    for sn in rac_scorer_names:
        var_costs = np.array([r["rac_results"][sn]["var_cost"] for r in all_results])
        binary_vars = np.array(
            [r["rac_results"][sn]["binary_var_cost"] for r in all_results]
        )
        improvements = np.array(
            [r["rac_results"][sn]["var_improvement"] for r in all_results]
        )
        fnrs = np.array([r["rac_results"][sn]["fnr"] for r in all_results])
        defer_rates = np.array(
            [r["rac_results"][sn]["defer_rate"] for r in all_results]
        )
        covs = np.array(
            [r["rac_results"][sn]["coverage_approve_normal"] for r in all_results]
        )

        rac_agg[sn] = {
            "var_cost": {
                "mean": float(var_costs.mean()),
                "std": float(var_costs.std()),
                **bootstrap_ci(var_costs),
            },
            "binary_var_cost": {
                "mean": float(binary_vars.mean()),
                "std": float(binary_vars.std()),
            },
            "var_improvement": {
                "mean": float(improvements.mean()),
                "std": float(improvements.std()),
                **bootstrap_ci(improvements),
            },
            "fnr": {
                "mean": float(fnrs.mean()),
                "std": float(fnrs.std()),
            },
            "defer_rate": {
                "mean": float(defer_rates.mean()),
                "std": float(defer_rates.std()),
            },
            "coverage": {
                "mean": float(covs.mean()),
                "std": float(covs.std()),
            },
        }

    return {
        "cp_comparison": cp_agg,
        "rac_comparison": rac_agg,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v16: Reframing eval — simple baselines on synthetic + RAC honest"
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--encoder-epochs", type=int, default=30)
    p.add_argument(
        "--cost-ratios", type=str, default="1,2,5,10,20",
    )
    p.add_argument(
        "--defer-budgets", type=str, default="0.05,0.10,0.15,0.20,0.30",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)
    cost_ratios = [float(c) for c in args.cost_ratios.split(",")]
    defer_budgets = [float(d) for d in args.defer_budgets.split(",")]

    print("=" * 70)
    print("v16: Reframing Evaluation")
    print("  Simple baselines on synthetic + RAC with honest scorers")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Alpha: {args.alpha}")
    print(f"Encoder epochs: {args.encoder_epochs}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load synthetic data
    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    cal_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_traces],
        dtype=np.int32,
    )
    test_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in test_traces],
        dtype=np.int32,
    )

    n_cal_normal = int((cal_labels == 0).sum())
    n_cal_anom = int((cal_labels == 1).sum())
    n_test_normal = int((test_labels == 0).sum())
    n_test_anom = int((test_labels == 1).sum())

    print(f"\nData: Train={len(train_traces)}, "
          f"Cal={len(cal_traces)} ({n_cal_normal} normal, {n_cal_anom} anom), "
          f"Test={len(test_traces)} ({n_test_normal} normal, {n_test_anom} anom)")
    print(f"Vocab: {vocab.vocab_size} tokens")

    start_time = time.time()
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Seed {seed} ({i + 1}/{len(seeds)})")
        print(f"{'='*50}")
        result = run_single_seed(
            seed, train_traces, cal_traces, test_traces,
            cal_labels, test_labels, vocab, device,
            args.alpha, args.encoder_epochs,
            cost_ratios, defer_budgets,
        )
        all_results.append(result)

    elapsed = time.time() - start_time

    # Aggregate
    agg = aggregate_results(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS: Split CP Comparison (Synthetic Data)")
    print("=" * 70)
    print(f"{'Scorer':<25} {'AUROC':>12} {'Coverage':>12} {'Bound%':>8}")
    print("-" * 60)
    for sn, metrics in agg["cp_comparison"].items():
        auroc_str = f"{metrics['auroc']['mean']:.4f}+/-{metrics['auroc']['std']:.3f}"
        cov_str = f"{metrics['coverage']['mean']:.4f}+/-{metrics['coverage']['std']:.3f}"
        bound_str = f"{metrics['bound_holds_pct']:.0f}%"
        print(f"{sn:<25} {auroc_str:>12} {cov_str:>12} {bound_str:>8}")

    print(f"\n{'='*70}")
    print("RESULTS: RAC Comparison (Honest Scorers, cost_ratio=5, defer=15%)")
    print("=" * 70)
    print(f"{'Scorer':<25} {'VaR Cost':>12} {'Improvement':>12} {'Defer%':>8}")
    print("-" * 60)
    for sn, metrics in agg["rac_comparison"].items():
        var_str = f"{metrics['var_cost']['mean']:.4f}+/-{metrics['var_cost']['std']:.3f}"
        imp_str = f"{metrics['var_improvement']['mean']:.4f}+/-{metrics['var_improvement']['std']:.3f}"
        defer_str = f"{metrics['defer_rate']['mean']*100:.1f}%"
        print(f"{sn:<25} {var_str:>12} {imp_str:>12} {defer_str:>8}")

    # Save
    output = {
        "config": {
            "seeds": seeds,
            "alpha": args.alpha,
            "encoder_epochs": args.encoder_epochs,
            "cost_ratios": cost_ratios,
            "defer_budgets": defer_budgets,
            "n_seeds": len(seeds),
        },
        "aggregate": agg,
        "per_seed": all_results,
        "elapsed_seconds": elapsed,
    }
    out_path = tables_dir / "reframing_eval_v16.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
