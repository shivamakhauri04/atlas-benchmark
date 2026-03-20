"""E4: Conformal coverage sweep across alpha values and domain pairs.

Evaluates split conformal prediction coverage on all 9-dataset pairs,
stratified by polarity reversal status. Tests multiple alpha values.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_conformal_coverage_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.domain_loader import DomainData, load_all_domains
from trustlens.conformal import split_conformal_calibrate
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")
FIG_DIR = Path("output/figures")
PAPER_FIG_DIR = Path(
    "../../papers/trustlens-auditable-agent-trust/submission/figures"
)

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]
ALPHA_VALUES = [0.01, 0.05, 0.10, 0.20]

SEMANTIC_GROUPS_4 = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "injecagent", "agentdojo", "agentharm", "rjudge"],
    "api_misuse": ["toolbench"],
}


def heuristic_score(features: dict[str, np.ndarray]) -> np.ndarray:
    cols = []
    for name in sorted(features.keys()):
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        cols.append((vals - mu) / sigma)
    return np.mean(np.column_stack(cols), axis=1)


def apply_pafc(
    features: dict[str, np.ndarray],
    src_d: dict[str, float],
    tgt_d: dict[str, float],
) -> dict[str, np.ndarray]:
    corrected = {}
    for name, values in features.items():
        d_s = src_d.get(name, 0.0)
        d_t = tgt_d.get(name, 0.0)
        if abs(d_s) > 0.1 and abs(d_t) > 0.1 and (d_s > 0) != (d_t > 0):
            corrected[name] = -values
        else:
            corrected[name] = values
    return corrected


def compute_d(
    features: dict[str, np.ndarray], labels: np.ndarray,
) -> dict[str, float]:
    result = {}
    for feat in FEATURE_NAMES:
        if feat not in features:
            result[feat] = 0.0
            continue
        result[feat] = cohens_d(features[feat][labels == 1], features[feat][labels == 0])
    return result


def check_reversal(d_a: dict[str, float], d_b: dict[str, float]) -> int:
    n = 0
    for f in FEATURE_NAMES:
        da, db = d_a.get(f, 0.0), d_b.get(f, 0.0)
        if abs(da) > 0.1 and abs(db) > 0.1 and (da > 0) != (db > 0):
            n += 1
    return n


def same_group(a: str, b: str) -> bool:
    for members in SEMANTIC_GROUPS_4.values():
        if a in members and b in members:
            return True
    return False


def run_coverage_experiment(
    domain_data: dict[str, DomainData],
    seeds: list[int],
) -> list[dict]:
    domains = sorted(domain_data.keys())
    domain_d = {
        name: compute_d(feat, lab) for name, (feat, lab) in domain_data.items()
    }

    results = []
    for src in domains:
        for tgt in domains:
            if src == tgt:
                continue
            src_feat, src_lab = domain_data[src]
            tgt_feat, tgt_lab = domain_data[tgt]
            n_reversed = check_reversal(domain_d[src], domain_d[tgt])
            within = same_group(src, tgt)

            for alpha in ALPHA_VALUES:
                raw_coverages = []
                pafc_coverages = []

                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    n_tgt = len(tgt_lab)
                    idx = rng.permutation(n_tgt)
                    mid = n_tgt // 2
                    cal_idx, test_idx = idx[:mid], idx[mid:]

                    cal_feat = {k: v[cal_idx] for k, v in tgt_feat.items()}
                    test_feat = {k: v[test_idx] for k, v in tgt_feat.items()}
                    test_labels = tgt_lab[test_idx]

                    # Raw scorer — conformal coverage = P(score <= tau | normal)
                    cal_scores = heuristic_score(cal_feat)
                    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
                    test_scores = heuristic_score(test_feat)
                    normal_mask = test_labels == 0
                    if normal_mask.sum() > 0:
                        coverage = float(
                            (test_scores[normal_mask] <= tau).mean()
                        )
                    else:
                        coverage = float("nan")
                    raw_coverages.append(coverage)

                    # PAFC scorer
                    pafc_cal = apply_pafc(cal_feat, domain_d[src], domain_d[tgt])
                    pafc_test = apply_pafc(test_feat, domain_d[src], domain_d[tgt])
                    cal_scores_p = heuristic_score(pafc_cal)
                    tau_p = split_conformal_calibrate(cal_scores_p, alpha=alpha)
                    test_scores_p = heuristic_score(pafc_test)
                    if normal_mask.sum() > 0:
                        coverage_p = float(
                            (test_scores_p[normal_mask] <= tau_p).mean()
                        )
                    else:
                        coverage_p = float("nan")
                    pafc_coverages.append(coverage_p)

                bound = 1 - alpha - 0.05
                raw_mean = float(np.mean(raw_coverages))
                pafc_mean = float(np.mean(pafc_coverages))

                results.append({
                    "source": src,
                    "target": tgt,
                    "alpha": alpha,
                    "n_reversed": n_reversed,
                    "within_group": within,
                    "raw_coverage_mean": round(raw_mean, 4),
                    "raw_meets_bound": raw_mean >= bound,
                    "pafc_coverage_mean": round(pafc_mean, 4),
                    "pafc_meets_bound": pafc_mean >= bound,
                    "n_target": len(tgt_lab),
                    "cal_size": len(tgt_lab) // 2,
                })

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456, 789, 1024]

    print("=" * 60)
    print("  E4: Conformal Coverage Sweep")
    print("=" * 60)

    print("\nLoading domains...")
    domain_data = load_all_domains(max_swe_traces=5000)
    domains = sorted(domain_data.keys())
    print(f"  Loaded {len(domains)} domains")

    print("\nRunning coverage experiments...")
    results = run_coverage_experiment(domain_data, seeds)

    # Summary at alpha=0.05
    alpha05 = [r for r in results if r["alpha"] == 0.05]
    reversed_pairs = [r for r in alpha05 if r["n_reversed"] > 0]
    nonrev_pairs = [r for r in alpha05 if r["n_reversed"] == 0]

    print(f"\n  At alpha=0.05:")
    print(f"    Reversed pairs ({len(reversed_pairs)}):")
    raw_meet = sum(1 for r in reversed_pairs if r["raw_meets_bound"])
    pafc_meet = sum(1 for r in reversed_pairs if r["pafc_meets_bound"])
    raw_cov = np.mean([r["raw_coverage_mean"] for r in reversed_pairs])
    pafc_cov = np.mean([r["pafc_coverage_mean"] for r in reversed_pairs])
    print(f"      Raw:  mean cov={raw_cov:.3f}, meets bound={raw_meet}/{len(reversed_pairs)} "
          f"({raw_meet/max(len(reversed_pairs),1):.1%})")
    print(f"      PAFC: mean cov={pafc_cov:.3f}, meets bound={pafc_meet}/{len(reversed_pairs)} "
          f"({pafc_meet/max(len(reversed_pairs),1):.1%})")

    print(f"    Non-reversed pairs ({len(nonrev_pairs)}):")
    raw_meet_nr = sum(1 for r in nonrev_pairs if r["raw_meets_bound"])
    pafc_meet_nr = sum(1 for r in nonrev_pairs if r["pafc_meets_bound"])
    raw_cov_nr = np.mean([r["raw_coverage_mean"] for r in nonrev_pairs])
    pafc_cov_nr = np.mean([r["pafc_coverage_mean"] for r in nonrev_pairs])
    print(f"      Raw:  mean cov={raw_cov_nr:.3f}, meets bound={raw_meet_nr}/{len(nonrev_pairs)} "
          f"({raw_meet_nr/max(len(nonrev_pairs),1):.1%})")
    print(f"      PAFC: mean cov={pafc_cov_nr:.3f}, meets bound={pafc_meet_nr}/{len(nonrev_pairs)} "
          f"({pafc_meet_nr/max(len(nonrev_pairs),1):.1%})")

    # Generate figure: coverage vs alpha, stratified by reversal
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, (label, subset) in zip(axes, [
        ("Reversed pairs", [r for r in results if r["n_reversed"] > 0]),
        ("Non-reversed pairs", [r for r in results if r["n_reversed"] == 0]),
    ]):
        for method, key, color, marker in [
            ("Raw", "raw_coverage_mean", "#F44336", "o"),
            ("PAFC", "pafc_coverage_mean", "#4CAF50", "s"),
        ]:
            means = []
            stds = []
            for alpha in ALPHA_VALUES:
                vals = [r[key] for r in subset if r["alpha"] == alpha]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.errorbar(
                ALPHA_VALUES, means, yerr=stds,
                label=method, color=color, marker=marker,
                markersize=6, capsize=3, linewidth=1.5,
            )

        # Plot the 1-alpha line
        ax.plot(ALPHA_VALUES, [1 - a for a in ALPHA_VALUES],
                "k--", alpha=0.5, linewidth=1, label="1 − α")
        ax.plot(ALPHA_VALUES, [1 - a - 0.05 for a in ALPHA_VALUES],
                "k:", alpha=0.3, linewidth=1, label="1 − α − 0.05")

        ax.set_xlabel("α (target miscoverage)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_xlim(-0.01, 0.22)
        ax.set_ylim(0.4, 1.05)

    axes[0].set_ylabel("Mean Coverage")
    fig.suptitle("Conformal Coverage Under Polarity Reversal", fontsize=12, y=1.02)
    plt.tight_layout()

    for dest in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(dest / "coverage_alpha_sweep.pdf")
    plt.close()
    print("\n  Saved coverage_alpha_sweep.pdf")

    output = {
        "experiment": "e4_conformal_coverage_sweep",
        "n_domains": len(domains),
        "alpha_values": ALPHA_VALUES,
        "seeds": seeds,
        "summary_alpha05": {
            "reversed": {
                "n_pairs": len(reversed_pairs),
                "raw_mean_coverage": round(float(raw_cov), 4),
                "raw_meets_bound_pct": round(raw_meet / max(len(reversed_pairs), 1), 4),
                "pafc_mean_coverage": round(float(pafc_cov), 4),
                "pafc_meets_bound_pct": round(pafc_meet / max(len(reversed_pairs), 1), 4),
            },
            "non_reversed": {
                "n_pairs": len(nonrev_pairs),
                "raw_mean_coverage": round(float(raw_cov_nr), 4),
                "raw_meets_bound_pct": round(raw_meet_nr / max(len(nonrev_pairs), 1), 4),
                "pafc_mean_coverage": round(float(pafc_cov_nr), 4),
                "pafc_meets_bound_pct": round(pafc_meet_nr / max(len(nonrev_pairs), 1), 4),
            },
        },
        "all_results": results,
    }
    out_path = OUTPUT_DIR / "e4_conformal_coverage_sweep.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
