"""Transfer prediction: when does cross-domain detection work?

Computes domain divergence metrics (KL, MMD, Wasserstein) between all
domain pairs and correlates with transfer AUROC from v30 results.
Tests whether semantic group membership alone predicts transfer success.

Key questions:
  1. Does feature divergence predict AUROC drop?
  2. Does number of reversed features predict PAFC delta?
  3. Can semantic group membership serve as a simple decision rule?

Usage:
    cd experiments/trustlens-auditable-agent-trust
    PYTHONPATH=../../lib:src uv run python scripts/run_transfer_prediction.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from data.domain_loader import load_all_domains
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")
FIG_DIR = Path("output/figures")
FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]

SEMANTIC_GROUPS = {
    "agentharm": "safety",
    "atbench": "safety",
    "injecagent": "safety",
    "rjudge": "safety",
    "swe_agent": "failure",
    "trail": "failure",
    "synthetic": "injected",
    "toolbench": "api_misuse",
}


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


def wasserstein_per_feature(
    feat_a: dict[str, np.ndarray],
    feat_b: dict[str, np.ndarray],
) -> dict[str, float]:
    result = {}
    for name in FEATURE_NAMES:
        if name in feat_a and name in feat_b:
            result[name] = float(stats.wasserstein_distance(feat_a[name], feat_b[name]))
        else:
            result[name] = 0.0
    return result


def kl_divergence_per_feature(
    feat_a: dict[str, np.ndarray],
    feat_b: dict[str, np.ndarray],
    n_bins: int = 50,
) -> dict[str, float]:
    """Binned KL divergence per feature (symmetrized)."""
    result = {}
    for name in FEATURE_NAMES:
        if name not in feat_a or name not in feat_b:
            result[name] = 0.0
            continue
        all_vals = np.concatenate([feat_a[name], feat_b[name]])
        bins = np.linspace(all_vals.min() - 1e-8, all_vals.max() + 1e-8, n_bins + 1)
        hist_a, _ = np.histogram(feat_a[name], bins=bins, density=True)
        hist_b, _ = np.histogram(feat_b[name], bins=bins, density=True)
        hist_a = hist_a + 1e-10
        hist_b = hist_b + 1e-10
        hist_a = hist_a / hist_a.sum()
        hist_b = hist_b / hist_b.sum()
        kl_ab = float(np.sum(hist_a * np.log(hist_a / hist_b)))
        kl_ba = float(np.sum(hist_b * np.log(hist_b / hist_a)))
        result[name] = (kl_ab + kl_ba) / 2.0
    return result


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        combined = np.vstack([X, Y])
        dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
        gamma = 1.0 / max(np.median(dists[dists > 0]), 1e-8)

    def _rbf(A, B):
        dists = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * dists)

    k_xx = _rbf(X, X).mean()
    k_yy = _rbf(Y, Y).mean()
    k_xy = _rbf(X, Y).mean()
    return float(k_xx + k_yy - 2 * k_xy)


def features_to_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    cols = [features[name] for name in FEATURE_NAMES if name in features]
    return np.column_stack(cols)


def load_v30_cross_domain() -> dict[str, dict[str, float]]:
    """Load v30 cross-domain AUROC results."""
    path = OUTPUT_DIR / "v30_expanded_cross_domain.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    pair_aurocs: dict[str, dict[str, float]] = {}
    for pair_key, methods in data["results"].items():
        pair_aurocs[pair_key] = {}
        for method, values in methods.items():
            pair_aurocs[pair_key][method] = float(np.mean(values))
    return pair_aurocs


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Transfer Prediction: Domain Divergence Analysis")
    print("=" * 70)

    print("\nLoading domains...")
    domain_data = load_all_domains()
    loaded = sorted(domain_data.keys())
    print(f"  Loaded: {loaded}")

    domain_d: dict[str, dict[str, float]] = {}
    for name, (features, labels) in domain_data.items():
        domain_d[name] = compute_domain_d(features, labels)

    print("\nLoading v30 cross-domain AUROC results...")
    v30_aurocs = load_v30_cross_domain()
    print(f"  Found {len(v30_aurocs)} pair results")

    print("\nComputing domain divergences...")
    pair_divergences: dict[str, dict] = {}

    for da in loaded:
        for db in loaded:
            if da == db:
                continue
            pair_key = f"{da}->{db}"
            feat_a, lab_a = domain_data[da]
            feat_b, lab_b = domain_data[db]

            w_dist = wasserstein_per_feature(feat_a, feat_b)
            kl_dist = kl_divergence_per_feature(feat_a, feat_b)

            X_a = features_to_matrix(feat_a)
            X_b = features_to_matrix(feat_b)
            mmd_val = mmd_rbf(X_a[:500], X_b[:500])

            n_rev = count_reversed_features(domain_d[da], domain_d[db])
            same_group = SEMANTIC_GROUPS.get(da) == SEMANTIC_GROUPS.get(db)

            d_diff = {}
            for f in FEATURE_NAMES:
                d_diff[f] = abs(domain_d[da].get(f, 0.0) - domain_d[db].get(f, 0.0))

            pair_divergences[pair_key] = {
                "source": da,
                "target": db,
                "same_semantic_group": same_group,
                "source_group": SEMANTIC_GROUPS.get(da, "unknown"),
                "target_group": SEMANTIC_GROUPS.get(db, "unknown"),
                "n_reversed_features": n_rev,
                "wasserstein": w_dist,
                "wasserstein_mean": float(np.mean(list(w_dist.values()))),
                "kl_divergence": kl_dist,
                "kl_mean": float(np.mean(list(kl_dist.values()))),
                "mmd_rbf": mmd_val,
                "cohens_d_diff": d_diff,
                "cohens_d_diff_mean": float(np.mean(list(d_diff.values()))),
            }

            if pair_key in v30_aurocs:
                pair_divergences[pair_key]["auroc_raw"] = v30_aurocs[pair_key].get(
                    "heuristic_raw", None,
                )
                pair_divergences[pair_key]["auroc_pafc"] = v30_aurocs[pair_key].get(
                    "heuristic_pafc", None,
                )
                raw = v30_aurocs[pair_key].get("heuristic_raw", 0.5)
                pafc = v30_aurocs[pair_key].get("heuristic_pafc", 0.5)
                pair_divergences[pair_key]["pafc_delta"] = pafc - raw

    correlation_data: list[dict] = []
    for pair_key, div in pair_divergences.items():
        if "auroc_raw" not in div or div["auroc_raw"] is None:
            continue
        correlation_data.append({
            "pair": pair_key,
            "auroc_raw": div["auroc_raw"],
            "auroc_pafc": div.get("auroc_pafc"),
            "pafc_delta": div.get("pafc_delta", 0.0),
            "wasserstein_mean": div["wasserstein_mean"],
            "kl_mean": div["kl_mean"],
            "mmd_rbf": div["mmd_rbf"],
            "d_diff_mean": div["cohens_d_diff_mean"],
            "n_reversed": div["n_reversed_features"],
            "same_group": div["same_semantic_group"],
        })

    correlations = {}
    if len(correlation_data) >= 5:
        aurocs = np.array([d["auroc_raw"] for d in correlation_data])
        deltas = np.array([d["pafc_delta"] for d in correlation_data])

        for metric in ["wasserstein_mean", "kl_mean", "mmd_rbf", "d_diff_mean", "n_reversed"]:
            vals = np.array([d[metric] for d in correlation_data])
            if np.std(vals) > 1e-10:
                r_auroc, p_auroc = stats.spearmanr(vals, aurocs)
                r_delta, p_delta = stats.spearmanr(vals, deltas)
                correlations[metric] = {
                    "vs_auroc_raw": {"rho": float(r_auroc), "p": float(p_auroc)},
                    "vs_pafc_delta": {"rho": float(r_delta), "p": float(p_delta)},
                }

        same_mask = np.array([d["same_group"] for d in correlation_data])
        if same_mask.sum() > 0 and (~same_mask).sum() > 0:
            within_auroc = aurocs[same_mask]
            between_auroc = aurocs[~same_mask]
            u_stat, p_val = stats.mannwhitneyu(
                within_auroc, between_auroc, alternative="two-sided",
            )
            correlations["semantic_group_test"] = {
                "within_group_auroc_mean": float(within_auroc.mean()),
                "between_group_auroc_mean": float(between_auroc.mean()),
                "mann_whitney_u": float(u_stat),
                "p_value": float(p_val),
                "n_within": int(same_mask.sum()),
                "n_between": int((~same_mask).sum()),
            }

            within_delta = deltas[same_mask]
            between_delta = deltas[~same_mask]
            u2, p2 = stats.mannwhitneyu(
                within_delta, between_delta, alternative="two-sided",
            )
            correlations["semantic_group_pafc_delta"] = {
                "within_group_delta_mean": float(within_delta.mean()),
                "between_group_delta_mean": float(between_delta.mean()),
                "mann_whitney_u": float(u2),
                "p_value": float(p2),
            }

    meta_predictor = {}
    if len(correlation_data) >= 10:
        X_meta = np.array([
            [d["wasserstein_mean"], d["kl_mean"], d["mmd_rbf"],
             d["d_diff_mean"], d["n_reversed"]]
            for d in correlation_data
        ])
        y_meta = np.array([d["auroc_raw"] for d in correlation_data])

        lr = LinearRegression()
        lr.fit(X_meta, y_meta)
        y_pred = lr.predict(X_meta)
        residuals = y_meta - y_pred
        ss_res = (residuals ** 2).sum()
        ss_tot = ((y_meta - y_meta.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

        meta_predictor = {
            "features": ["wasserstein", "kl", "mmd", "d_diff", "n_reversed"],
            "coefficients": lr.coef_.tolist(),
            "intercept": float(lr.intercept_),
            "r_squared": float(r2),
            "n_samples": len(correlation_data),
        }

        group_only = np.array([
            [1.0 if d["same_group"] else 0.0] for d in correlation_data
        ])
        lr_group = LinearRegression()
        lr_group.fit(group_only, y_meta)
        y_pred_group = lr_group.predict(group_only)
        residuals_g = y_meta - y_pred_group
        ss_res_g = (residuals_g ** 2).sum()
        r2_group = 1.0 - ss_res_g / max(ss_tot, 1e-10)
        meta_predictor["group_only_r_squared"] = float(r2_group)

    when_to_correct = {}
    if correlation_data:
        helps = [d for d in correlation_data if d["pafc_delta"] > 0.05]
        hurts = [d for d in correlation_data if d["pafc_delta"] < -0.05]
        neutral = [d for d in correlation_data if abs(d["pafc_delta"]) <= 0.05]

        when_to_correct = {
            "pafc_helps": {
                "n_pairs": len(helps),
                "pairs": [d["pair"] for d in helps],
                "mean_n_reversed": float(np.mean([d["n_reversed"] for d in helps])) if helps else 0,
                "pct_same_group": float(np.mean([d["same_group"] for d in helps]) * 100) if helps else 0,
            },
            "pafc_hurts": {
                "n_pairs": len(hurts),
                "pairs": [d["pair"] for d in hurts],
                "mean_n_reversed": float(np.mean([d["n_reversed"] for d in hurts])) if hurts else 0,
                "pct_same_group": float(np.mean([d["same_group"] for d in hurts]) * 100) if hurts else 0,
            },
            "neutral": {
                "n_pairs": len(neutral),
            },
        }

        rev_vals = np.array([d["n_reversed"] for d in correlation_data])
        delta_vals = np.array([d["pafc_delta"] for d in correlation_data])
        if np.std(rev_vals) > 1e-10:
            r, p = stats.spearmanr(rev_vals, delta_vals)
            when_to_correct["n_reversed_vs_delta"] = {
                "spearman_rho": float(r),
                "p_value": float(p),
            }

    elapsed = time.time() - t0
    output = {
        "experiment": "transfer_prediction",
        "n_domains": len(loaded),
        "domains": loaded,
        "semantic_groups": SEMANTIC_GROUPS,
        "pair_divergences": pair_divergences,
        "correlations": correlations,
        "meta_predictor": meta_predictor,
        "when_to_correct": when_to_correct,
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "transfer_prediction.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("  Results")
    print(f"{'=' * 70}")

    if correlations:
        print("\n  Divergence vs AUROC correlations:")
        for metric, vals in correlations.items():
            if metric.startswith("semantic"):
                continue
            print(
                f"    {metric}: rho={vals['vs_auroc_raw']['rho']:.3f} "
                f"(p={vals['vs_auroc_raw']['p']:.4f})"
            )

        sg = correlations.get("semantic_group_test", {})
        if sg:
            print(f"\n  Semantic group test:")
            print(f"    Within-group AUROC:  {sg['within_group_auroc_mean']:.3f} (n={sg['n_within']})")
            print(f"    Between-group AUROC: {sg['between_group_auroc_mean']:.3f} (n={sg['n_between']})")
            print(f"    Mann-Whitney p:      {sg['p_value']:.4f}")

    if meta_predictor:
        print(f"\n  Meta-predictor R²: {meta_predictor['r_squared']:.3f}")
        print(f"  Group-only R²:    {meta_predictor.get('group_only_r_squared', 0):.3f}")

    if when_to_correct:
        h = when_to_correct["pafc_helps"]
        d = when_to_correct["pafc_hurts"]
        print(f"\n  When PAFC helps ({h['n_pairs']} pairs):")
        print(f"    Mean reversed features: {h['mean_n_reversed']:.1f}")
        print(f"    Same semantic group:    {h['pct_same_group']:.0f}%")
        print(f"  When PAFC hurts ({d['n_pairs']} pairs):")
        print(f"    Mean reversed features: {d['mean_n_reversed']:.1f}")
        print(f"    Same semantic group:    {d['pct_same_group']:.0f}%")

    print(f"\n  Results saved to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
