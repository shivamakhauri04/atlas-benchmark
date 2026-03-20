"""v26.1: Adversarial domain selection experiment.

For each ordered pair (source, target) across all domains:
    1. Compute aggregate FSI
    2. Train LogisticRegression scorer on source, evaluate AUROC + evasion on target
    3. Test: Spearman(|FSI|, AUROC) should be negative (high reversal → low AUROC)
       and Spearman(|FSI|, evasion) should be positive (high reversal → high evasion)

Key SaTML contribution: an attacker who knows the training domain
can exploit reversed features to evade monitoring.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_adversarial_domain.py
    PYTHONPATH=../../lib:src uv run python scripts/run_adversarial_domain.py \
        --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.cross_domain_scorer import CrossDomainScorer
from trustlens.polarity import (
    aggregate_fsi,
    compute_cross_domain_fsi,
    compute_domain_correlations,
)
from trustlens.stats import bootstrap_ci

OUTPUT_DIR = Path("output/tables")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v26.1: Adversarial domain selection")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-swe-traces", type=int, default=5000)
    return p.parse_args()


def compute_evasion_rate(
    source_features: dict[str, np.ndarray],
    source_labels: np.ndarray,
    target_features: dict[str, np.ndarray],
    target_labels: np.ndarray,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    """Train LR scorer on source, compute evasion rate and AUROC on target.

    Evasion rate = fraction of target anomalies that are NOT flagged
    (false negative rate on the target anomalous subset).
    """
    scorer = CrossDomainScorer(seed=seed)
    try:
        cal_scores = scorer.fit(source_features, source_labels)
    except ValueError:
        return {
            "evasion_rate": float("nan"),
            "fnr": float("nan"),
            "n_anomalies": int((target_labels == 1).sum()),
            "auroc": float("nan"),
        }

    test_scores = scorer.score(target_features)

    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    anomaly_mask = target_labels == 1
    n_anomalies = anomaly_mask.sum()
    if n_anomalies == 0:
        return {"evasion_rate": 0.0, "fnr": 0.0, "n_anomalies": 0, "auroc": 0.5}

    false_negatives = (preds[anomaly_mask] == 0).sum()
    evasion_rate = float(false_negatives / n_anomalies)

    try:
        auroc = float(roc_auc_score(target_labels, test_scores))
    except ValueError:
        auroc = 0.5

    return {
        "evasion_rate": evasion_rate,
        "fnr": evasion_rate,
        "n_anomalies": int(n_anomalies),
        "auroc": auroc,
    }


def run_single_seed(
    seed: int,
    domain_data: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
    alpha: float,
) -> dict:
    """Run adversarial analysis for one seed."""
    set_seeds(seed)
    domains = sorted(domain_data.keys())

    pair_results = []
    for source in domains:
        corr_source = compute_domain_correlations(
            domain_data[source][0], domain_data[source][1],
        )
        for target in domains:
            if source == target:
                continue
            corr_target = compute_domain_correlations(
                domain_data[target][0], domain_data[target][1],
            )
            fsi = compute_cross_domain_fsi(corr_source, corr_target)
            agg = aggregate_fsi(fsi)

            evasion = compute_evasion_rate(
                domain_data[source][0], domain_data[source][1],
                domain_data[target][0], domain_data[target][1],
                alpha, seed,
            )

            pair_results.append({
                "source": source,
                "target": target,
                "aggregate_fsi": agg,
                "abs_aggregate_fsi": abs(agg),
                "per_feature_fsi": {k: float(v) for k, v in fsi.items()},
                **evasion,
            })

    valid = [r for r in pair_results if np.isfinite(r["auroc"])]
    abs_fsi_vals = np.array([r["abs_aggregate_fsi"] for r in valid])
    auroc_vals = np.array([r["auroc"] for r in valid])
    evasion_vals = np.array([r["evasion_rate"] for r in valid])

    if len(abs_fsi_vals) >= 3 and np.std(abs_fsi_vals) > 1e-10:
        rho_auroc, p_auroc = sp_stats.spearmanr(abs_fsi_vals, auroc_vals)
        rho_evasion, p_evasion = sp_stats.spearmanr(abs_fsi_vals, evasion_vals)
    else:
        rho_auroc, p_auroc = 0.0, 1.0
        rho_evasion, p_evasion = 0.0, 1.0

    def _safe(v: float, default: float = 0.0) -> float:
        return float(v) if np.isfinite(v) else default

    return {
        "seed": seed,
        "fsi_auroc_spearman_rho": _safe(rho_auroc),
        "fsi_auroc_p_value": _safe(p_auroc, 1.0),
        "fsi_evasion_spearman_rho": _safe(rho_evasion),
        "fsi_evasion_p_value": _safe(p_evasion, 1.0),
        "n_valid_pairs": len(valid),
        "n_total_pairs": len(pair_results),
        "pair_results": pair_results,
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("v26.1: Adversarial Domain Selection Experiment")
    print("=" * 60)

    from data.domain_loader import load_all_domains

    print("\nLoading domains...")
    domain_data = load_all_domains(
        seeds=args.seeds,
        max_swe_traces=args.max_swe_traces,
    )

    domains = sorted(domain_data.keys())
    print(f"\nLoaded {len(domains)} domains: {domains}")
    n_pairs = len(domains) * (len(domains) - 1)
    print(f"Testing {n_pairs} ordered domain pairs")

    all_seed_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        result = run_single_seed(seed, domain_data, args.alpha)
        rho_a = result["fsi_auroc_spearman_rho"]
        p_a = result["fsi_auroc_p_value"]
        rho_e = result["fsi_evasion_spearman_rho"]
        p_e = result["fsi_evasion_p_value"]
        n_v = result["n_valid_pairs"]
        print(f"  Spearman(|FSI|, AUROC)   = {rho_a:.3f}  (p={p_a:.4f})")
        print(f"  Spearman(|FSI|, evasion) = {rho_e:.3f}  (p={p_e:.4f})")
        print(f"  Valid pairs: {n_v}/{result['n_total_pairs']}")
        all_seed_results.append(result)

    rho_auroc_values = np.array([
        r["fsi_auroc_spearman_rho"] for r in all_seed_results
    ])
    rho_evasion_values = np.array([
        r["fsi_evasion_spearman_rho"] for r in all_seed_results
    ])
    rho_auroc_ci = bootstrap_ci(rho_auroc_values)
    rho_evasion_ci = bootstrap_ci(rho_evasion_values)

    all_pairs_data: dict[str, dict[str, list[float]]] = {}
    for r in all_seed_results:
        for pr in r["pair_results"]:
            key = f"{pr['source']}_to_{pr['target']}"
            if key not in all_pairs_data:
                all_pairs_data[key] = {"evasion_rate": [], "abs_fsi": [], "auroc": []}
            if np.isfinite(pr["auroc"]):
                all_pairs_data[key]["evasion_rate"].append(pr["evasion_rate"])
                all_pairs_data[key]["abs_fsi"].append(pr["abs_aggregate_fsi"])
                all_pairs_data[key]["auroc"].append(pr["auroc"])

    pair_aggregate = {}
    for key, vals in all_pairs_data.items():
        if not vals["auroc"]:
            continue
        pair_aggregate[key] = {
            "evasion_rate": bootstrap_ci(np.array(vals["evasion_rate"])),
            "abs_fsi": bootstrap_ci(np.array(vals["abs_fsi"])),
            "auroc": bootstrap_ci(np.array(vals["auroc"])),
        }

    output = {
        "version": "v26.1",
        "scorer": "logistic_regression_balanced",
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "alpha": args.alpha,
        "domains": domains,
        "spearman_rho_auroc_summary": rho_auroc_ci,
        "spearman_rho_evasion_summary": rho_evasion_ci,
        "pair_aggregate": pair_aggregate,
        "per_seed_results": all_seed_results,
        "elapsed_seconds": time.time() - t0,
    }

    out_path = OUTPUT_DIR / "adversarial_domain.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    print("\n--- Adversarial Domain Selection Summary ---")
    print(f"  Spearman(|FSI|, AUROC):   rho={rho_auroc_ci['mean']:.3f} "
          f"[{rho_auroc_ci['ci_lower']:.3f}, {rho_auroc_ci['ci_upper']:.3f}]")
    print(f"  Spearman(|FSI|, evasion): rho={rho_evasion_ci['mean']:.3f} "
          f"[{rho_evasion_ci['ci_lower']:.3f}, {rho_evasion_ci['ci_upper']:.3f}]")

    print("\n  Top evasion pairs:")
    sorted_pairs = sorted(pair_aggregate.items(), key=lambda x: -x[1]["evasion_rate"]["mean"])
    for key, agg in sorted_pairs[:5]:
        er = agg["evasion_rate"]["mean"]
        fsi_v = agg["abs_fsi"]["mean"]
        auroc_v = agg["auroc"]["mean"]
        print(f"    {key}: evasion={er:.3f}  |FSI|={fsi_v:.3f}  AUROC={auroc_v:.3f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
