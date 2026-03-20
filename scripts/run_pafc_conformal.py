"""v29: PAFC-Conformal coverage validation (Theorem 2).

Validates that PAFC sign correction + conformal prediction maintains
coverage >= 1-alpha-delta on target domains, while uncorrected CP
suffers catastrophic coverage violation on reversed domains.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_pafc_conformal.py
    PYTHONPATH=../../lib:src uv run python scripts/run_pafc_conformal.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds

from data.domain_loader import (
    load_all_domains,
)
from trustlens.conformal import (
    pafc_conformal_calibrate,
    pafc_conformal_predict,
    split_conformal_calibrate,
    split_conformal_predict,
    theorem2_coverage_bound,
)
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]

ALPHA_VALUES = [0.05, 0.10, 0.20]


def compute_domain_d(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, float]:
    """Cohen's d per feature for a single domain."""
    result: dict[str, float] = {}
    for feat in FEATURE_NAMES:
        if feat not in features:
            result[feat] = 0.0
            continue
        vals = features[feat]
        result[feat] = cohens_d(vals[labels == 1], vals[labels == 0])
    return result


def heuristic_scorer(
    features: dict[str, np.ndarray],
    d_orient: dict[str, float] | None = None,
) -> np.ndarray:
    """Z-score + mean combination, optionally oriented by Cohen's d sign."""
    cols = []
    for name in sorted(features.keys()):
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        z = (vals - mu) / sigma
        if d_orient is not None and d_orient.get(name, 0.0) < 0:
            z = -z
        cols.append(z)
    return np.mean(np.column_stack(cols), axis=1)


def run_coverage_experiment(
    src_name: str,
    tgt_name: str,
    src_features: dict[str, np.ndarray],
    src_labels: np.ndarray,
    tgt_features: dict[str, np.ndarray],
    tgt_labels: np.ndarray,
    seed: int,
    alpha: float,
) -> dict:
    """Run one coverage experiment: cross-domain calibration.

    Uncorrected: calibrate on SOURCE with source-oriented scorer, test on TARGET.
    PAFC-corrected: use N=50 target seed to detect reversal, PAFC-correct target
    features, calibrate on seed, test on remaining target.
    """
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    from functools import partial

    from sklearn.metrics import roc_auc_score

    src_d = compute_domain_d(src_features, src_labels)
    scorer_src_orient = partial(heuristic_scorer, d_orient=src_d)

    n_src = len(src_labels)
    src_idx = rng.permutation(n_src)
    src_cal_idx = src_idx[: n_src // 2]

    src_cal_features = {k: v[src_cal_idx] for k, v in src_features.items()}
    src_cal_scores = scorer_src_orient(src_cal_features)
    uncorrected_tau = split_conformal_calibrate(src_cal_scores, alpha=alpha)

    n_tgt = len(tgt_labels)
    tgt_idx = rng.permutation(n_tgt)
    n_seed = min(50, n_tgt // 3)
    seed_idx = tgt_idx[:n_seed]
    test_idx = tgt_idx[n_seed:]

    test_features = {k: v[test_idx] for k, v in tgt_features.items()}
    test_labels = tgt_labels[test_idx]
    seed_features = {k: v[seed_idx] for k, v in tgt_features.items()}
    seed_labels = tgt_labels[seed_idx]

    uncorrected_test_scores = scorer_src_orient(test_features)
    uncorrected_preds = split_conformal_predict(uncorrected_test_scores, uncorrected_tau)

    normal_mask = test_labels == 0
    uncorrected_coverage = float(
        (uncorrected_preds[normal_mask] == 0).sum() / max(normal_mask.sum(), 1)
    )
    try:
        uncorrected_auroc = float(roc_auc_score(test_labels, uncorrected_test_scores))
    except ValueError:
        uncorrected_auroc = 0.5

    seed_d = compute_domain_d(seed_features, seed_labels)
    scorer_seed_orient = partial(heuristic_scorer, d_orient=seed_d)

    pafc_tau, corrected_names = pafc_conformal_calibrate(
        seed_features, seed_labels, src_d, seed_d,
        alpha=alpha, scorer_fn=scorer_seed_orient,
    )
    pafc_result = pafc_conformal_predict(
        test_features, test_labels, src_d, seed_d, pafc_tau,
        scorer_fn=scorer_seed_orient,
    )

    n_reversed = sum(
        1 for f in FEATURE_NAMES
        if abs(src_d.get(f, 0)) > 0.1
        and abs(seed_d.get(f, 0)) > 0.1
        and (src_d[f] > 0) != (seed_d[f] > 0)
    )

    min_d = min(
        abs(seed_d.get(f, 0)) for f in FEATURE_NAMES if abs(seed_d.get(f, 0)) > 0.1
    ) if any(abs(seed_d.get(f, 0)) > 0.1 for f in FEATURE_NAMES) else 0.0
    if min_d > 0:
        se_d = np.sqrt(2.0 / n_seed + min_d**2 / (2.0 * n_seed))
        z = min_d / se_d
        from scipy.stats import norm
        delta = 1.0 - norm.cdf(z)
    else:
        delta = 0.5

    theoretical_bound = theorem2_coverage_bound(alpha, delta)

    uncorrected_set_size = float(uncorrected_preds.mean())
    pafc_preds = np.array(pafc_result["predictions"])
    pafc_set_size = float(pafc_preds.mean())

    return {
        "src_domain": src_name,
        "tgt_domain": tgt_name,
        "alpha": alpha,
        "seed": seed,
        "n_seed": n_seed,
        "n_test": len(test_labels),
        "n_reversed_features": n_reversed,
        "uncorrected_coverage": uncorrected_coverage,
        "uncorrected_auroc": uncorrected_auroc,
        "uncorrected_set_size": uncorrected_set_size,
        "pafc_coverage": pafc_result["coverage"],
        "pafc_auroc": pafc_result["auroc"],
        "pafc_set_size": pafc_set_size,
        "pafc_n_corrected": pafc_result["n_corrected_features"],
        "delta_sign_detection": delta,
        "theoretical_bound": theoretical_bound,
        "pafc_satisfies_bound": pafc_result["coverage"] >= theoretical_bound - 0.01,
        "uncorrected_violates": uncorrected_coverage < (1.0 - alpha - 0.05),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v29: PAFC-Conformal coverage validation")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-swe-traces", type=int, default=5000)
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123, 456]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  v29: PAFC-Conformal Coverage Validation (Theorem 2)")
    print("=" * 70)

    print("\nLoading domains...")
    domain_data = load_all_domains(max_swe_traces=args.max_swe_traces)
    loaded = sorted(domain_data.keys())
    print(f"  Loaded: {loaded}")

    domain_d: dict[str, dict[str, float]] = {}
    for name, (features, labels) in domain_data.items():
        domain_d[name] = compute_domain_d(features, labels)

    cross_pairs: list[tuple[str, str]] = []
    for src in loaded:
        for tgt in loaded:
            if src == tgt:
                continue
            n_reversed = sum(
                1 for f in FEATURE_NAMES
                if abs(domain_d[src].get(f, 0)) > 0.1
                and abs(domain_d[tgt].get(f, 0)) > 0.1
                and (domain_d[src][f] > 0) != (domain_d[tgt][f] > 0)
            )
            if n_reversed > 0:
                cross_pairs.append((src, tgt))

    if not cross_pairs:
        print("  No reversed pairs found, using synthetic->* as default")
        for tgt in loaded:
            if tgt != "synthetic":
                cross_pairs.append(("synthetic", tgt))

    print(f"\n  Pairs with polarity reversal: {len(cross_pairs)}")

    all_results: list[dict] = []

    for src_name, tgt_name in cross_pairs:
        src_feat, src_lbl = domain_data[src_name]
        tgt_feat, tgt_lbl = domain_data[tgt_name]

        print(f"\n  {src_name} -> {tgt_name}:")
        for alpha in ALPHA_VALUES:
            seed_results = []
            for seed in args.seeds:
                res = run_coverage_experiment(
                    src_name, tgt_name,
                    src_feat, src_lbl, tgt_feat, tgt_lbl,
                    seed, alpha,
                )
                seed_results.append(res)
                all_results.append(res)

            uncorr_covs = [r["uncorrected_coverage"] for r in seed_results]
            pafc_covs = [r["pafc_coverage"] for r in seed_results]
            bounds = [r["theoretical_bound"] for r in seed_results]

            print(
                f"    alpha={alpha:.2f}: "
                f"uncorrected={np.mean(uncorr_covs):.3f}+/-{np.std(uncorr_covs):.3f}  "
                f"PAFC={np.mean(pafc_covs):.3f}+/-{np.std(pafc_covs):.3f}  "
                f"bound={np.mean(bounds):.3f}  "
                f"PAFC_ok={sum(r['pafc_satisfies_bound'] for r in seed_results)}"
                f"/{len(seed_results)}"
            )

    print("\n" + "=" * 70)
    print("  Summary Table")
    print("=" * 70)

    for alpha in ALPHA_VALUES:
        alpha_results = [r for r in all_results if r["alpha"] == alpha]
        if not alpha_results:
            continue
        uncorr_mean = np.mean([r["uncorrected_coverage"] for r in alpha_results])
        pafc_mean = np.mean([r["pafc_coverage"] for r in alpha_results])
        n_pafc_ok = sum(1 for r in alpha_results if r["pafc_satisfies_bound"])
        n_uncorr_violates = sum(1 for r in alpha_results if r["uncorrected_violates"])
        print(
            f"  alpha={alpha:.2f}: "
            f"uncorrected_cov={uncorr_mean:.3f} (violates={n_uncorr_violates})  "
            f"pafc_cov={pafc_mean:.3f} (satisfies_bound={n_pafc_ok}/{len(alpha_results)})"
        )

    output = {
        "experiment": "v29_pafc_conformal",
        "seeds": args.seeds,
        "alpha_values": ALPHA_VALUES,
        "cross_pairs": [f"{s}->{t}" for s, t in cross_pairs],
        "results": all_results,
    }
    out_path = OUTPUT_DIR / "v29_pafc_conformal.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
