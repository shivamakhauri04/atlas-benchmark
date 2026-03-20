"""v29: Few-shot baselines on N=50 calibration budget.

Compares PAFC (unsupervised correction) against supervised baselines
(LR, kNN, RF) given the same N labeled traces. Sweeps N in {10,20,50,100,200}.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_fewshot_baselines.py
    PYTHONPATH=../../lib:src uv run python scripts/run_fewshot_baselines.py --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from data.domain_loader import (
    load_all_domains,
)
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]

N_VALUES = [10, 20, 50, 100, 200]


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


def build_feature_matrix(features: dict[str, np.ndarray]) -> np.ndarray:
    """Stack features into (n, 3) matrix."""
    return np.column_stack([features[f] for f in FEATURE_NAMES if f in features])


def apply_pafc(
    features: dict[str, np.ndarray],
    source_d: dict[str, float],
    target_d: dict[str, float],
) -> dict[str, np.ndarray]:
    """PAFC sign correction."""
    corrected = {}
    for name, values in features.items():
        d_src = source_d.get(name, 0.0)
        d_tgt = target_d.get(name, 0.0)
        if abs(d_src) > 0.1 and abs(d_tgt) > 0.1 and (d_src > 0) != (d_tgt > 0):
            corrected[name] = -values
        else:
            corrected[name] = values
    return corrected


def heuristic_scorer(features: dict[str, np.ndarray]) -> np.ndarray:
    """Z-score + mean."""
    cols = []
    for name in sorted(features.keys()):
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        cols.append((vals - mu) / sigma)
    return np.mean(np.column_stack(cols), axis=1)


def evaluate_fewshot(
    src_features: dict[str, np.ndarray],
    src_labels: np.ndarray,
    tgt_features: dict[str, np.ndarray],
    tgt_labels: np.ndarray,
    n_labeled: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Evaluate all methods given N labeled target traces."""
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    n_tgt = len(tgt_labels)
    if n_labeled >= n_tgt:
        n_labeled = n_tgt // 2

    idx = rng.permutation(n_tgt)
    labeled_idx = idx[:n_labeled]
    test_idx = idx[n_labeled:]

    labeled_features = {k: v[labeled_idx] for k, v in tgt_features.items()}
    labeled_labels = tgt_labels[labeled_idx]
    test_features_split = {k: v[test_idx] for k, v in tgt_features.items()}
    test_labels = tgt_labels[test_idx]

    if len(test_labels) < 5 or test_labels.sum() < 2 or (1 - test_labels).sum() < 2:
        return {}

    results: dict[str, dict[str, float]] = {}

    X_train = build_feature_matrix(labeled_features)
    X_test = build_feature_matrix(test_features_split)
    y_train = labeled_labels

    if len(np.unique(y_train)) < 2:
        return {}

    for name, clf in [
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("knn", KNeighborsClassifier(n_neighbors=min(5, n_labeled - 1))),
        ("rf", RandomForestClassifier(n_estimators=50, max_depth=3, random_state=seed)),
    ]:
        try:
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            auroc = float(roc_auc_score(test_labels, probs))
        except (ValueError, IndexError):
            auroc = 0.5
        results[name] = {"auroc": auroc}

    src_d = compute_domain_d(src_features, src_labels)
    tgt_d_labeled = compute_domain_d(labeled_features, labeled_labels)

    def oriented_scorer(features: dict[str, np.ndarray], d_ref: dict[str, float]) -> np.ndarray:
        """Score using z-scores oriented by Cohen's d sign (positive d = anomalous)."""
        cols = []
        for name in sorted(features.keys()):
            vals = features[name]
            mu, sigma = vals.mean(), max(vals.std(), 1e-8)
            z = (vals - mu) / sigma
            if d_ref.get(name, 0.0) < 0:
                z = -z
            cols.append(z)
        return np.mean(np.column_stack(cols), axis=1)

    source_scores = oriented_scorer(test_features_split, src_d)
    try:
        source_auroc = float(roc_auc_score(test_labels, source_scores))
    except ValueError:
        source_auroc = 0.5
    results["heuristic_source"] = {"auroc": source_auroc}

    raw_scores = oriented_scorer(test_features_split, tgt_d_labeled)
    try:
        raw_auroc = float(roc_auc_score(test_labels, raw_scores))
    except ValueError:
        raw_auroc = 0.5
    results["heuristic_target"] = {"auroc": raw_auroc}

    corrected_train = apply_pafc(labeled_features, src_d, tgt_d_labeled)
    corrected_test = apply_pafc(test_features_split, src_d, tgt_d_labeled)

    pafc_scores = oriented_scorer(corrected_test, src_d)
    try:
        pafc_auroc = float(roc_auc_score(test_labels, pafc_scores))
    except ValueError:
        pafc_auroc = 0.5
    results["pafc_heuristic"] = {"auroc": pafc_auroc}
    X_train_pafc = build_feature_matrix(corrected_train)
    X_test_pafc = build_feature_matrix(corrected_test)
    try:
        lr_pafc = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr_pafc.fit(X_train_pafc, y_train)
        pafc_lr_scores = lr_pafc.predict_proba(X_test_pafc)[:, 1]
        pafc_lr_auroc = float(roc_auc_score(test_labels, pafc_lr_scores))
    except (ValueError, IndexError):
        pafc_lr_auroc = 0.5
    results["lr_pafc"] = {"auroc": pafc_lr_auroc}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="v29: Few-shot baselines")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-swe-traces", type=int, default=5000)
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123, 456]
        N_VALUES[:] = [10, 50]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  v29: Few-Shot Baselines (N=50 Calibration Budget)")
    print("=" * 70)

    print("\nLoading domains...")
    domain_data = load_all_domains(max_swe_traces=args.max_swe_traces)
    loaded = sorted(domain_data.keys())
    print(f"  Loaded: {loaded}")

    cross_pairs = [
        (src, tgt)
        for src in loaded
        for tgt in loaded
        if src != tgt and src == "synthetic"
    ]
    print(f"  Cross-domain pairs: {len(cross_pairs)}")

    all_results: list[dict] = []

    for src_name, tgt_name in cross_pairs:
        src_feat, src_lbl = domain_data[src_name]
        tgt_feat, tgt_lbl = domain_data[tgt_name]

        print(f"\n{'='*60}")
        print(f"  {src_name} -> {tgt_name} "
              f"(n_tgt={len(tgt_lbl)}, anom={int(tgt_lbl.sum())})")
        print(f"{'='*60}")

        for n_labeled in N_VALUES:
            if n_labeled > len(tgt_lbl) // 2:
                print(f"\n  N={n_labeled}: skipping (exceeds available data)")
                continue

            print(f"\n  N={n_labeled}:")
            method_aurocs: dict[str, list[float]] = {}

            for seed in args.seeds:
                res = evaluate_fewshot(
                    src_feat, src_lbl, tgt_feat, tgt_lbl,
                    n_labeled, seed,
                )
                if not res:
                    continue

                for method, metrics in res.items():
                    method_aurocs.setdefault(method, []).append(metrics["auroc"])
                    all_results.append({
                        "src": src_name,
                        "tgt": tgt_name,
                        "n_labeled": n_labeled,
                        "seed": seed,
                        "method": method,
                        "auroc": metrics["auroc"],
                    })

            for method, aurocs in sorted(method_aurocs.items()):
                arr = np.array(aurocs)
                print(f"    {method:20s}: {arr.mean():.4f} +/- {arr.std():.4f}")

    print("\n" + "=" * 70)
    print("  Summary: Sample Efficiency Curves")
    print("=" * 70)

    for src_name, tgt_name in cross_pairs:
        pair_results = [
            r for r in all_results
            if r["src"] == src_name and r["tgt"] == tgt_name
        ]
        if not pair_results:
            continue

        print(f"\n  {src_name} -> {tgt_name}:")
        methods = sorted(set(r["method"] for r in pair_results))
        header = f"  {'N':>5s}"
        for m in methods:
            header += f"  {m:>14s}"
        print(header)

        for n in N_VALUES:
            row = f"  {n:>5d}"
            for m in methods:
                vals = [
                    r["auroc"] for r in pair_results
                    if r["n_labeled"] == n and r["method"] == m
                ]
                if vals:
                    row += f"  {np.mean(vals):>14.4f}"
                else:
                    row += f"  {'---':>14s}"
            print(row)

    output = {
        "experiment": "v29_fewshot_baselines",
        "seeds": args.seeds,
        "n_values": N_VALUES,
        "domain_pairs": [f"{s}->{t}" for s, t in cross_pairs],
        "results": all_results,
    }
    out_path = OUTPUT_DIR / "v29_fewshot_baselines.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
