"""v26.1: Polarity-Aware Conformal Prediction (PA-CP) evaluation.

For each domain pair (source, target):
    1. Standard CP transfer: train LR on ALL source features, evaluate on target
    2. PA-CP: train LR on STABLE features only (PUF removed via FSI)
    Measures: coverage, AUROC, F1

Compares two trained LR scorers — all-feature vs stable-feature — to show
that removing polarity-unstable features improves cross-domain AUROC.

Also runs PUF threshold sweep: {0.0, -0.05, -0.1, -0.15, -0.2}

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_pacp_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_pacp_eval.py --seeds 42 123 456 789 1024
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds
from sklearn.metrics import f1_score, roc_auc_score

from trustlens.conformal import (
    compute_polarity_coverage_bound,
    split_conformal_calibrate,
    split_conformal_predict,
)
from trustlens.cross_domain_scorer import CrossDomainScorer
from trustlens.polarity import (
    compute_cross_domain_fsi,
    compute_domain_correlations,
    identify_puf,
)
from trustlens.stats import bootstrap_ci

OUTPUT_DIR = Path("output/tables")
PUF_THRESHOLDS = [0.0, -0.05, -0.1, -0.15, -0.2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v26.1: PA-CP evaluation")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-swe-traces", type=int, default=5000)
    return p.parse_args()


def _score_and_evaluate(
    source_features: dict[str, np.ndarray],
    source_labels: np.ndarray,
    target_features: dict[str, np.ndarray],
    target_labels: np.ndarray,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    """Train LR on source features, calibrate+predict on target."""
    scorer = CrossDomainScorer(seed=seed)
    try:
        cal_scores = scorer.fit(source_features, source_labels)
    except ValueError:
        return {"coverage": float("nan"), "auroc": float("nan"), "f1": 0.0, "tau": 0.0}

    test_scores = scorer.score(target_features)
    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    normal_mask = target_labels == 0
    coverage = float((preds[normal_mask] == 0).sum() / max(normal_mask.sum(), 1))

    try:
        auroc = float(roc_auc_score(target_labels, test_scores))
    except ValueError:
        auroc = 0.5

    try:
        f1 = float(f1_score(target_labels, preds))
    except ValueError:
        f1 = 0.0

    return {"coverage": coverage, "auroc": auroc, "f1": f1, "tau": float(tau)}


def evaluate_domain_pair(
    source_name: str,
    target_name: str,
    source_features: dict[str, np.ndarray],
    source_labels: np.ndarray,
    target_features: dict[str, np.ndarray],
    target_labels: np.ndarray,
    alpha: float,
    seed: int,
) -> dict:
    """Evaluate standard CP vs PA-CP for one (source, target) pair."""
    set_seeds(seed)

    corr_source = compute_domain_correlations(source_features, source_labels)
    corr_target = compute_domain_correlations(target_features, target_labels)
    fsi = compute_cross_domain_fsi(corr_source, corr_target)

    standard_result = _score_and_evaluate(
        source_features, source_labels,
        target_features, target_labels,
        alpha, seed,
    )

    threshold_results = []
    for threshold in PUF_THRESHOLDS:
        puf = identify_puf(fsi, threshold=threshold)
        stable = {k for k in source_features if k not in puf}
        if not stable:
            stable = set(source_features.keys())

        src_stable = {k: v for k, v in source_features.items() if k in stable}
        tgt_stable = {k: v for k, v in target_features.items() if k in stable}

        pacp_result = _score_and_evaluate(
            src_stable, source_labels,
            tgt_stable, target_labels,
            alpha, seed,
        )

        n_features = len(source_features)
        n_puf = len(puf & set(source_features.keys()))
        delta_polarity = max(abs(v) for v in fsi.values()) if fsi else 0.0
        bound = compute_polarity_coverage_bound(n_features, n_puf, alpha, delta_polarity)

        pacp_cov = pacp_result["coverage"]
        threshold_results.append({
            "threshold": threshold,
            "puf": sorted(puf),
            "n_puf": n_puf,
            "n_stable": n_features - n_puf,
            "coverage": pacp_cov,
            "auroc": pacp_result["auroc"],
            "f1": pacp_result["f1"],
            "tau": pacp_result["tau"],
            "coverage_bound": bound,
            "bound_holds": pacp_cov >= bound - 0.01 if np.isfinite(pacp_cov) else False,
        })

    return {
        "source": source_name,
        "target": target_name,
        "seed": seed,
        "fsi": {k: float(v) for k, v in fsi.items()},
        "standard_cp": standard_result,
        "pacp_by_threshold": threshold_results,
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("v26.1: Polarity-Aware CP (PA-CP) Evaluation")
    print("=" * 60)

    from data.domain_loader import load_all_domains

    print("\nLoading domains...")
    domain_data = load_all_domains(
        seeds=args.seeds,
        max_swe_traces=args.max_swe_traces,
    )

    domains = sorted(domain_data.keys())
    print(f"\nLoaded {len(domains)} domains: {domains}")

    all_results = []
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                print(f"  {source} -> {target}...", end=" ")
                result = evaluate_domain_pair(
                    source, target,
                    domain_data[source][0], domain_data[source][1],
                    domain_data[target][0], domain_data[target][1],
                    args.alpha, seed,
                )
                std_cov = result["standard_cp"]["coverage"]
                pacp_cov = result["pacp_by_threshold"][0]["coverage"]
                std_auroc = result["standard_cp"]["auroc"]
                pacp_auroc = result["pacp_by_threshold"][0]["auroc"]
                if np.isfinite(std_cov) and np.isfinite(pacp_cov):
                    improvement = pacp_cov - std_cov
                    print(f"cov: std={std_cov:.3f} pacp={pacp_cov:.3f} "
                          f"(delta={improvement:+.3f})  "
                          f"auroc: std={std_auroc:.3f} pacp={pacp_auroc:.3f}")
                else:
                    print("fit failed (skipped)")
                all_results.append(result)

    aggregate = {}
    for source in domains:
        for target in domains:
            if source == target:
                continue
            key = f"{source}_to_{target}"
            pair_results = [
                r for r in all_results
                if r["source"] == source and r["target"] == target
            ]
            std_coverages = np.array([
                r["standard_cp"]["coverage"] for r in pair_results
                if np.isfinite(r["standard_cp"]["coverage"])
            ])
            pacp_coverages = np.array([
                r["pacp_by_threshold"][0]["coverage"] for r in pair_results
                if np.isfinite(r["pacp_by_threshold"][0]["coverage"])
            ])
            std_aurocs = np.array([
                r["standard_cp"]["auroc"] for r in pair_results
                if np.isfinite(r["standard_cp"]["auroc"])
            ])
            pacp_aurocs = np.array([
                r["pacp_by_threshold"][0]["auroc"] for r in pair_results
                if np.isfinite(r["pacp_by_threshold"][0]["auroc"])
            ])
            if len(std_coverages) == 0:
                continue
            min_len = min(len(std_coverages), len(pacp_coverages))
            aggregate[key] = {
                "standard_cp_coverage": bootstrap_ci(std_coverages),
                "pacp_coverage": bootstrap_ci(pacp_coverages),
                "improvement": bootstrap_ci(pacp_coverages[:min_len] - std_coverages[:min_len]),
                "standard_cp_auroc": bootstrap_ci(std_aurocs),
                "pacp_auroc": bootstrap_ci(pacp_aurocs),
            }

    output = {
        "version": "v26.1",
        "scorer": "logistic_regression_balanced",
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "alpha": args.alpha,
        "puf_thresholds": PUF_THRESHOLDS,
        "domains": domains,
        "aggregate": aggregate,
        "per_seed_results": all_results,
        "elapsed_seconds": time.time() - t0,
    }

    out_path = OUTPUT_DIR / "pacp_eval.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    print("\n--- PA-CP vs Standard CP (threshold=0.0) ---")
    for key, agg in sorted(aggregate.items()):
        std_c = agg["standard_cp_coverage"]["mean"]
        pacp_c = agg["pacp_coverage"]["mean"]
        imp = agg["improvement"]["mean"]
        std_a = agg["standard_cp_auroc"]["mean"]
        pacp_a = agg["pacp_auroc"]["mean"]
        print(f"  {key}: cov std={std_c:.3f} pacp={pacp_c:.3f} delta={imp:+.3f}"
              f"  auroc std={std_a:.3f} pacp={pacp_a:.3f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
