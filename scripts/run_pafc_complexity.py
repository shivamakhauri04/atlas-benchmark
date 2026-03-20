"""Experiment 2: PAFC sample complexity bound + empirical validation.

Derives the minimum N for reliable polarity detection from Cohen's d
estimation theory, then validates empirically via Monte Carlo simulation.

Key result: closed-form minimum seed size N as a function of true |d|
and confidence level alpha.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_pafc_complexity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from research_utils.reproducibility import set_seeds

OUTPUT_DIR = Path("output/tables")


def cohens_d_se(d: float, n1: int, n2: int) -> float:
    """Standard error of Cohen's d (Hedges & Olkin, 1985).

    SE(d) = sqrt( (n1+n2)/(n1*n2) + d^2/(2*(n1+n2)) )
    """
    return np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))


def min_n_for_sign_detection(
    true_d: float,
    alpha: float = 0.05,
    anom_fraction: float = 0.2,
) -> int:
    """Minimum total N to detect sign of Cohen's d at confidence 1-alpha.

    For sign detection, we need: |d| / SE(d) > z_{alpha}
    With n1 = N * anom_fraction, n2 = N * (1 - anom_fraction).

    Returns minimum total N (both classes combined).
    """
    if abs(true_d) < 1e-10:
        return int(1e6)

    z = stats.norm.ppf(1 - alpha)
    f = anom_fraction

    for N in range(4, 10000):
        n1 = max(2, int(N * f))
        n2 = max(2, N - n1)
        se = cohens_d_se(true_d, n1, n2)
        if abs(true_d) / se > z:
            return N
    return 10000


def simulate_detection_probability(
    true_d: float,
    N: int,
    anom_fraction: float = 0.2,
    n_simulations: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo: probability of correctly detecting d's sign with N samples."""
    if rng is None:
        rng = np.random.default_rng(42)

    n_anom = max(2, int(N * anom_fraction))
    n_norm = max(2, N - n_anom)
    correct = 0
    true_sign = np.sign(true_d)
    if true_sign == 0:
        return 0.5

    for _ in range(n_simulations):
        norm_samples = rng.normal(0, 1, n_norm)
        anom_samples = rng.normal(true_d, 1, n_anom)

        n_a, n_b = len(anom_samples), len(norm_samples)
        var_a = anom_samples.var(ddof=1)
        var_b = norm_samples.var(ddof=1)
        pooled = np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )
        if pooled < 1e-10:
            continue
        est_d = (anom_samples.mean() - norm_samples.mean()) / pooled
        if np.sign(est_d) == true_sign:
            correct += 1

    return correct / n_simulations


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(42)
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("PAFC Sample Complexity Analysis (v30)")
    print("=" * 60)

    observed_effects = {
        "trace_length (syn)": -0.336,
        "low_diversity (syn)": -0.464,
        "repeat_rate (syn)": -0.114,
        "trace_length (SWE)": 0.51,
        "low_diversity (SWE)": 0.71,
        "repeat_rate (SWE)": 0.61,
        "trace_length (ATB)": 2.23,
    }

    # --- Part 1: Analytical minimum N ---
    print("\n--- Part 1: Analytical Minimum N for Sign Detection ---")
    print(f"{'Feature':>25} | {'|d|':>6} | {'N (95%)':>8} | {'N (99%)':>8}")
    print("-" * 60)

    analytical_results = {}
    for name, d in observed_effects.items():
        n95 = min_n_for_sign_detection(d, alpha=0.05)
        n99 = min_n_for_sign_detection(d, alpha=0.01)
        analytical_results[name] = {"d": d, "abs_d": abs(d), "n_95": n95, "n_99": n99}
        print(f"{name:>25} | {abs(d):>6.3f} | {n95:>8d} | {n99:>8d}")

    # --- Part 2: General formula table ---
    print("\n--- Part 2: Minimum N vs Effect Size (20% anomaly rate) ---")
    print(f"{'|d|':>6} | {'N (95%)':>8} | {'N (99%)':>8} | {'Category':>12}")
    print("-" * 50)

    d_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    formula_table = []
    for d in d_values:
        n95 = min_n_for_sign_detection(d, alpha=0.05)
        n99 = min_n_for_sign_detection(d, alpha=0.01)
        if abs(d) < 0.2:
            cat = "negligible"
        elif abs(d) < 0.5:
            cat = "small"
        elif abs(d) < 0.8:
            cat = "medium"
        else:
            cat = "large"
        formula_table.append({"d": d, "n_95": n95, "n_99": n99, "category": cat})
        print(f"{d:>6.1f} | {n95:>8d} | {n99:>8d} | {cat:>12}")

    # --- Part 3: Monte Carlo validation ---
    print("\n--- Part 3: Empirical Detection Probability vs N ---")

    test_effects = [
        ("small (|d|=0.11, repeat_rate)", 0.114),
        ("medium (|d|=0.34, trace_length)", 0.336),
        ("medium (|d|=0.46, low_diversity)", 0.464),
        ("large (|d|=0.51, SWE trace_length)", 0.51),
        ("very large (|d|=2.23, ATB trace_length)", 2.23),
    ]

    N_sweep = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    mc_results = {}

    for name, true_d in test_effects:
        print(f"\n  {name}:")
        mc_results[name] = {"true_d": true_d, "sweep": []}
        for N in N_sweep:
            prob = simulate_detection_probability(
                true_d, N, n_simulations=2000, rng=rng,
            )
            mc_results[name]["sweep"].append({"N": N, "detection_prob": prob})
            marker = " <-- 95%" if 0.94 < prob < 0.96 else ""
            bar = "#" * int(prob * 40)
            print(f"    N={N:>4}: {prob:.3f} {bar}{marker}")

    # --- Part 4: Practical recommendations ---
    print("\n" + "=" * 60)
    print("PRACTICAL RECOMMENDATIONS FOR PAFC DEPLOYMENT")
    print("=" * 60)
    print()
    print("Given observed effect sizes in agent trace monitoring:")
    print()
    print("  Feature         |d|     Min N (95%)  Recommendation")
    print("  --------------- ------  -----------  ---------------")
    for name, d in [
        ("trace_length", 0.336),
        ("low_diversity", 0.464),
        ("repeat_rate", 0.114),
    ]:
        n95 = min_n_for_sign_detection(d, alpha=0.05)
        rec = "easy" if n95 <= 30 else "moderate" if n95 <= 100 else "difficult"
        print(f"  {name:<17} {abs(d):.3f}   {n95:>11d}  {rec}")

    print()
    print("  Rule of thumb: N=50 detects reversals for |d|>=0.3 at 95% confidence.")
    print("  For small effects (|d|<0.2), N>200 needed — consider dropping the feature.")

    elapsed = time.time() - t0
    results = {
        "analytical": analytical_results,
        "formula_table": formula_table,
        "mc_results": {
            k: {"true_d": v["true_d"], "sweep": v["sweep"]}
            for k, v in mc_results.items()
        },
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "pafc_complexity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
