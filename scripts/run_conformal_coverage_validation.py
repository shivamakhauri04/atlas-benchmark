"""Conformal coverage validation under polarity reversal.

Tests whether split CP coverage guarantees hold when features undergo
polarity reversal. For each of the 26 cross-domain pairs:

1. Calibrate CP threshold on source domain (split CP at alpha=0.05)
2. Apply threshold to target domain (raw features)
3. Apply threshold to target domain (PAFC-corrected features)
4. Report empirical coverage vs theoretical bound

Key question: Does polarity reversal break CP coverage guarantees?

Usage:
    cd experiments/trustlens-auditable-agent-trust
    PYTHONPATH=../../lib:src uv run python scripts/run_conformal_coverage_validation.py
    PYTHONPATH=../../lib:src uv run python scripts/run_conformal_coverage_validation.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds
from sklearn.metrics import roc_auc_score

from data.domain_loader import load_all_domains
from trustlens.conformal import split_conformal_calibrate
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")
FIG_DIR = Path("output/figures")
FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]


def compute_domain_d(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for feat in FEATURE_NAMES:
        vals = features.get(feat, np.zeros(len(labels)))
        normal = vals[labels == 0]
        anomalous = vals[labels == 1]
        result[feat] = cohens_d(anomalous, normal)
    return result


def heuristic_score(features: dict[str, np.ndarray]) -> np.ndarray:
    cols = []
    for name in sorted(features.keys()):
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        cols.append((vals - mu) / sigma)
    return np.mean(np.column_stack(cols), axis=1)


def apply_pafc_correction(
    features: dict[str, np.ndarray],
    source_d: dict[str, float],
    target_d: dict[str, float],
) -> dict[str, np.ndarray]:
    corrected = {}
    for name, values in features.items():
        d_src = source_d.get(name, 0.0)
        d_tgt = target_d.get(name, 0.0)
        if abs(d_src) > 0.1 and abs(d_tgt) > 0.1 and (d_src > 0) != (d_tgt > 0):
            corrected[name] = -values
        else:
            corrected[name] = values
    return corrected


def count_reversed_features(
    src_d: dict[str, float],
    tgt_d: dict[str, float],
) -> int:
    n = 0
    for f in FEATURE_NAMES:
        d_s = src_d.get(f, 0.0)
        d_t = tgt_d.get(f, 0.0)
        if abs(d_s) > 0.1 and abs(d_t) > 0.1 and (d_s > 0) != (d_t > 0):
            n += 1
    return n


def evaluate_coverage_pair(
    src_features: dict[str, np.ndarray],
    src_labels: np.ndarray,
    tgt_features: dict[str, np.ndarray],
    tgt_labels: np.ndarray,
    src_d: dict[str, float],
    tgt_d: dict[str, float],
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Evaluate CP coverage for one pair and one seed."""
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    results: dict[str, float] = {}

    n_src = len(src_labels)
    perm = rng.permutation(n_src)
    cal_idx = perm[: n_src // 2]
    src_cal_features = {k: v[cal_idx] for k, v in src_features.items()}
    src_cal_labels = src_labels[cal_idx]

    normal_mask_cal = src_cal_labels == 0
    if normal_mask_cal.sum() < 5:
        return {"error": "too few normal calibration samples"}

    cal_normal_features = {k: v[normal_mask_cal] for k, v in src_cal_features.items()}

    cal_scores = heuristic_score(cal_normal_features)
    tau_raw = split_conformal_calibrate(cal_scores, alpha=alpha)

    tgt_scores_raw = heuristic_score(tgt_features)
    tgt_normal = tgt_labels == 0
    n_normal = tgt_normal.sum()

    if n_normal == 0:
        return {"error": "no normal target samples"}

    preds_raw = (tgt_scores_raw > tau_raw).astype(int)
    coverage_raw = float((preds_raw[tgt_normal] == 0).sum() / n_normal)

    try:
        auroc_raw = float(roc_auc_score(tgt_labels, tgt_scores_raw))
    except ValueError:
        auroc_raw = 0.5

    results["coverage_raw"] = coverage_raw
    results["auroc_raw"] = auroc_raw
    results["tau_raw"] = float(tau_raw)

    corrected_tgt = apply_pafc_correction(tgt_features, src_d, tgt_d)
    tgt_scores_pafc = heuristic_score(corrected_tgt)
    preds_pafc = (tgt_scores_pafc > tau_raw).astype(int)
    coverage_pafc = float((preds_pafc[tgt_normal] == 0).sum() / n_normal)

    corrected_cal = apply_pafc_correction(cal_normal_features, src_d, tgt_d)
    cal_scores_pafc = heuristic_score(corrected_cal)
    tau_pafc = split_conformal_calibrate(cal_scores_pafc, alpha=alpha)
    preds_pafc_recal = (tgt_scores_pafc > tau_pafc).astype(int)
    coverage_pafc_recal = float((preds_pafc_recal[tgt_normal] == 0).sum() / n_normal)

    try:
        auroc_pafc = float(roc_auc_score(tgt_labels, tgt_scores_pafc))
    except ValueError:
        auroc_pafc = 0.5

    results["coverage_pafc"] = coverage_pafc
    results["coverage_pafc_recalibrated"] = coverage_pafc_recal
    results["auroc_pafc"] = auroc_pafc
    results["tau_pafc"] = float(tau_pafc)

    n_reversed = count_reversed_features(src_d, tgt_d)
    results["n_reversed_features"] = n_reversed

    theoretical_bound = 1.0 - alpha
    results["theoretical_coverage_bound"] = theoretical_bound
    results["raw_within_bound"] = bool(coverage_raw >= theoretical_bound - 0.05)
    results["pafc_within_bound"] = bool(coverage_pafc_recal >= theoretical_bound - 0.05)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conformal coverage validation under polarity reversal",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  Conformal Coverage Validation Under Polarity Reversal")
    print("=" * 70)

    print("\nLoading domains...")
    domain_data = load_all_domains()
    loaded = sorted(domain_data.keys())
    print(f"  Loaded: {loaded}")

    domain_d: dict[str, dict[str, float]] = {}
    for name, (features, labels) in domain_data.items():
        domain_d[name] = compute_domain_d(features, labels)

    all_pairs: list[tuple[str, str]] = []
    for da in loaded:
        for db in loaded:
            if da == db:
                continue
            all_pairs.append((da, db))

    print(f"\n  Evaluating {len(all_pairs)} pairs x {len(args.seeds)} seeds")

    results: dict[str, list[dict]] = {}

    for pair_idx, (src, tgt) in enumerate(all_pairs):
        pair_key = f"{src}->{tgt}"
        print(f"\n  [{pair_idx + 1}/{len(all_pairs)}] {pair_key}")
        results[pair_key] = []

        src_feat, src_lab = domain_data[src]
        tgt_feat, tgt_lab = domain_data[tgt]
        n_rev = count_reversed_features(domain_d[src], domain_d[tgt])

        for seed in args.seeds:
            r = evaluate_coverage_pair(
                src_feat, src_lab, tgt_feat, tgt_lab,
                domain_d[src], domain_d[tgt],
                seed=seed, alpha=args.alpha,
            )
            results[pair_key].append(r)

        if results[pair_key] and "error" not in results[pair_key][0]:
            cov_raw = np.mean([r["coverage_raw"] for r in results[pair_key]])
            cov_pafc = np.mean([r["coverage_pafc_recalibrated"] for r in results[pair_key]])
            print(
                f"    reversed={n_rev}, "
                f"cov_raw={cov_raw:.3f}, "
                f"cov_pafc_recal={cov_pafc:.3f}, "
                f"bound={1 - args.alpha:.2f}"
            )

    summary: dict[str, dict[str, float]] = {}
    for pair_key, seed_results in results.items():
        valid = [r for r in seed_results if "error" not in r]
        if not valid:
            continue
        summary[pair_key] = {
            "coverage_raw_mean": float(np.mean([r["coverage_raw"] for r in valid])),
            "coverage_raw_std": float(np.std([r["coverage_raw"] for r in valid])),
            "coverage_pafc_mean": float(np.mean([r["coverage_pafc_recalibrated"] for r in valid])),
            "coverage_pafc_std": float(np.std([r["coverage_pafc_recalibrated"] for r in valid])),
            "auroc_raw_mean": float(np.mean([r["auroc_raw"] for r in valid])),
            "auroc_pafc_mean": float(np.mean([r["auroc_pafc"] for r in valid])),
            "n_reversed_features": int(valid[0]["n_reversed_features"]),
            "raw_within_bound_pct": float(np.mean([r["raw_within_bound"] for r in valid])) * 100,
            "pafc_within_bound_pct": float(np.mean([r["pafc_within_bound"] for r in valid])) * 100,
        }

    reversed_pairs = {k: v for k, v in summary.items() if v["n_reversed_features"] > 0}
    non_reversed = {k: v for k, v in summary.items() if v["n_reversed_features"] == 0}

    if reversed_pairs:
        rev_cov_raw = np.mean([v["coverage_raw_mean"] for v in reversed_pairs.values()])
        rev_cov_pafc = np.mean([v["coverage_pafc_mean"] for v in reversed_pairs.values()])
        rev_bound_raw = np.mean([v["raw_within_bound_pct"] for v in reversed_pairs.values()])
        rev_bound_pafc = np.mean([v["pafc_within_bound_pct"] for v in reversed_pairs.values()])
    else:
        rev_cov_raw = rev_cov_pafc = rev_bound_raw = rev_bound_pafc = float("nan")

    if non_reversed:
        non_cov_raw = np.mean([v["coverage_raw_mean"] for v in non_reversed.values()])
        non_bound_raw = np.mean([v["raw_within_bound_pct"] for v in non_reversed.values()])
    else:
        non_cov_raw = non_bound_raw = float("nan")

    aggregate = {
        "reversed_pairs": {
            "n_pairs": len(reversed_pairs),
            "coverage_raw_mean": float(rev_cov_raw),
            "coverage_pafc_mean": float(rev_cov_pafc),
            "within_bound_raw_pct": float(rev_bound_raw),
            "within_bound_pafc_pct": float(rev_bound_pafc),
        },
        "non_reversed_pairs": {
            "n_pairs": len(non_reversed),
            "coverage_raw_mean": float(non_cov_raw),
            "within_bound_raw_pct": float(non_bound_raw),
        },
    }

    elapsed = time.time() - t0
    output = {
        "experiment": "conformal_coverage_validation",
        "alpha": args.alpha,
        "seeds": args.seeds,
        "n_pairs": len(all_pairs),
        "results": results,
        "summary": summary,
        "aggregate": aggregate,
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "conformal_coverage_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("  Aggregate Results")
    print(f"{'=' * 70}")
    print(f"\n  Reversed pairs ({len(reversed_pairs)}):")
    print(f"    Raw coverage:  {rev_cov_raw:.3f} (within bound: {rev_bound_raw:.1f}%)")
    print(f"    PAFC coverage: {rev_cov_pafc:.3f} (within bound: {rev_bound_pafc:.1f}%)")
    print(f"\n  Non-reversed pairs ({len(non_reversed)}):")
    print(f"    Raw coverage:  {non_cov_raw:.3f} (within bound: {non_bound_raw:.1f}%)")
    print(f"\n  Results saved to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
