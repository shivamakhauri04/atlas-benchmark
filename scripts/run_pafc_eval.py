"""v29: Polarity-Aware Feature Correction (PAFC) evaluation.

Validates the PAFC mitigation for feature polarity reversal:
1. Compute source-domain Cohen's d for each heuristic feature
2. Sample a small labeled seed set from the target domain
3. Compute seed Cohen's d, detect reversed polarities
4. Negate reversed features, re-score with corrected features
5. Compare naive transfer vs PAFC-corrected AUROC

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_pafc_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_pafc_eval.py --seed-sizes 20 50 100
    PYTHONPATH=../../lib:src uv run python scripts/run_pafc_eval.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from research_utils.reproducibility import set_seeds

OUTPUT_DIR = Path("output/tables")
MIN_TRACE_LEN = 3
MAX_TRACE_LEN = 200

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d: (mean_a - mean_b) / pooled_std."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def compute_features(traces: list[dict], label_key: str) -> dict:
    """Compute heuristic features and labels for a list of trace dicts."""
    n = len(traces)
    features = {
        "trace_length": np.zeros(n),
        "low_diversity": np.zeros(n),
        "repeat_rate": np.zeros(n),
    }
    labels = np.zeros(n, dtype=np.int32)

    for i, t in enumerate(traces):
        actions = t["actions"]
        n_actions = len(actions)
        features["trace_length"][i] = float(n_actions)
        unique = len(set(actions))
        features["low_diversity"][i] = 1.0 - (unique / max(n_actions, 1))
        if n_actions > 1:
            repeats = sum(
                1 for j in range(1, n_actions) if actions[j] == actions[j - 1]
            )
            features["repeat_rate"][i] = repeats / (n_actions - 1)
        labels[i] = t[label_key]

    return {"features": features, "labels": labels}


def compute_domain_cohens_d(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute Cohen's d (anomalous - normal) for each feature."""
    normal_mask = labels == 0
    anom_mask = labels == 1
    result = {}
    for fname in FEATURE_NAMES:
        vals = features[fname]
        result[fname] = cohens_d(vals[anom_mask], vals[normal_mask])
    return result


def score_transfer(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    source_d_signs: dict[str, float],
    corrections: dict[str, float] | None = None,
) -> float:
    """Score using source-domain feature directions transferred to target.

    This simulates cross-domain transfer: z-normalize features globally
    (scale normalization only, no class-aware adaptation), then weight by
    source domain's Cohen's d sign. Corrections negate reversed features.

    Returns AUROC.
    """
    n = len(labels)
    if n < 5:
        return float("nan")

    combined_scores = np.zeros(n)
    active = 0

    for fname in FEATURE_NAMES:
        vals = features[fname].copy()
        mu = vals.mean()
        std = vals.std()
        if std < 1e-10:
            continue
        z = (vals - mu) / std
        direction = source_d_signs.get(fname, 1.0)
        if corrections and fname in corrections:
            direction *= corrections[fname]
        combined_scores += z * direction
        active += 1

    if active == 0:
        return float("nan")
    combined_scores /= active

    n_pos = labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    return float(roc_auc_score(labels, combined_scores))


def run_pafc_single_seed(
    source_data: dict,
    target_traces: list[dict],
    target_label_key: str,
    seed_size: int,
    rng: np.random.Generator,
) -> dict:
    """Run PAFC for one random seed.

    1. Compute source-domain Cohen's d
    2. Sample seed_size traces from target
    3. Compute seed Cohen's d
    4. Detect reversed features (sign disagrees with source)
    5. Score full target with corrected features
    """
    source_d = compute_domain_cohens_d(
        source_data["features"], source_data["labels"],
    )

    target_data = compute_features(target_traces, target_label_key)
    target_features = target_data["features"]
    target_labels = target_data["labels"]
    n_target = len(target_labels)

    indices = rng.permutation(n_target)[:seed_size]
    seed_features = {
        fname: target_features[fname][indices] for fname in FEATURE_NAMES
    }
    seed_labels = target_labels[indices]

    n_seed_normal = int((seed_labels == 0).sum())
    n_seed_anom = int((seed_labels == 1).sum())

    if n_seed_normal < 2 or n_seed_anom < 2:
        return {"error": "insufficient_seed_labels"}

    seed_d = compute_domain_cohens_d(seed_features, seed_labels)

    corrections = {}
    reversed_features = []
    for fname in FEATURE_NAMES:
        src_sign = np.sign(source_d[fname])
        seed_sign = np.sign(seed_d[fname])
        if np.isnan(source_d[fname]) or np.isnan(seed_d[fname]):
            corrections[fname] = 1.0
        elif src_sign != seed_sign and src_sign != 0 and seed_sign != 0:
            corrections[fname] = -1.0
            reversed_features.append(fname)
        else:
            corrections[fname] = 1.0

    source_d_signs = {
        fname: float(np.sign(source_d[fname])) if not np.isnan(source_d[fname]) else 1.0
        for fname in FEATURE_NAMES
    }

    naive_auroc = score_transfer(target_features, target_labels, source_d_signs)
    pafc_auroc = score_transfer(
        target_features, target_labels, source_d_signs, corrections,
    )

    full_target_d = compute_domain_cohens_d(target_features, target_labels)
    target_d_signs = {
        fname: float(np.sign(full_target_d[fname]))
        if not np.isnan(full_target_d[fname]) else 1.0
        for fname in FEATURE_NAMES
    }
    oracle_auroc = score_transfer(
        target_features, target_labels, target_d_signs,
    )

    return {
        "source_d": source_d,
        "seed_d": seed_d,
        "full_target_d": full_target_d,
        "corrections": corrections,
        "reversed_features": reversed_features,
        "n_reversed": len(reversed_features),
        "seed_size": seed_size,
        "n_seed_normal": n_seed_normal,
        "n_seed_anom": n_seed_anom,
        "naive_auroc": naive_auroc,
        "pafc_auroc": pafc_auroc,
        "oracle_auroc": oracle_auroc,
        "delta_pafc": pafc_auroc - naive_auroc,
        "delta_oracle": oracle_auroc - naive_auroc,
    }


# ---------------------------------------------------------------------------
# Data loading (reuse patterns from existing scripts)
# ---------------------------------------------------------------------------


def load_swe_agent_data(max_traces: int = 5000) -> list[dict]:
    """Load SWE-agent trajectories."""
    import re
    from datasets import load_dataset

    print(f"Loading up to {max_traces} SWE-agent trajectories...")
    ds = load_dataset(
        "nebius/SWE-agent-trajectories", split="train", streaming=True,
    )
    traces = []
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions = []
        for turn in example["trajectory"]:
            if turn.get("role") == "ai" and turn.get("text"):
                code_blocks = re.findall(
                    r"```\s*\n?(.*?)\n?```", turn["text"], re.DOTALL,
                )
                for block in code_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    first_line = block.split("\n")[0].strip()
                    cmd = first_line.split()[0] if first_line.split() else ""
                    cmd = cmd.lstrip("./!$")
                    if cmd:
                        actions.append(cmd)
        if len(actions) < MIN_TRACE_LEN:
            continue
        if len(actions) > MAX_TRACE_LEN:
            actions = actions[:MAX_TRACE_LEN]
        traces.append({
            "actions": actions,
            "label": 0 if bool(example["target"]) else 1,
            "n_actions": len(actions),
        })
    solved = sum(1 for t in traces if t["label"] == 0)
    print(f"Loaded {len(traces)} traces ({solved} solved)")
    return traces


def load_atbench_traces() -> list[dict]:
    """Load ATBench traces."""
    from data.trace_loader import load_atbench

    print("Loading ATBench...")
    traces = load_atbench(max_traces=None)
    result = []
    for t in traces:
        actions = [a.tool for a in t.actions]
        if len(actions) < 2:
            continue
        result.append({
            "actions": actions[:50],
            "label": 1 if t.has_anomaly else 0,
            "n_actions": min(len(actions), 50),
        })
    safe = sum(1 for r in result if r["label"] == 0)
    print(f"Loaded {len(result)} traces ({safe} safe)")
    return result


def load_synthetic_traces() -> list[dict]:
    """Load synthetic traces from local JSONL."""
    import orjson

    result = []
    for split_name, path in [("train", "data/train.jsonl"), ("test", "data/test.jsonl")]:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for line in f:
                obj = orjson.loads(line)
                raw_actions = obj.get("tools", obj.get("actions", []))
                actions = [
                    a["tool"] if isinstance(a, dict) else str(a)
                    for a in raw_actions
                ]
                if len(actions) >= MIN_TRACE_LEN:
                    label = obj.get("label", obj.get("anomaly", None))
                    if label is None:
                        label = int(any(
                            a.get("label", 0) == 1
                            for a in raw_actions
                            if isinstance(a, dict)
                        ))
                    result.append({
                        "actions": actions[:MAX_TRACE_LEN],
                        "label": int(label),
                        "n_actions": min(len(actions), MAX_TRACE_LEN),
                    })
    normal = sum(1 for r in result if r["label"] == 0)
    print(f"Loaded {len(result)} synthetic ({normal} normal)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="PAFC mitigation evaluation")
    parser.add_argument(
        "--seed-sizes", nargs="+", type=int, default=[20, 50, 100],
        help="Seed set sizes to test",
    )
    parser.add_argument(
        "--seeds", type=int, default=10,
        help="Number of random seeds for variance estimation",
    )
    parser.add_argument(
        "--max-swe-traces", type=int, default=5000,
        help="Max SWE-agent traces to load",
    )
    args = parser.parse_args()

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random_seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
    random_seeds = random_seeds[: args.seeds]

    print("=" * 60)
    print("PAFC Mitigation Evaluation (v29)")
    print(f"Seed sizes: {args.seed_sizes}")
    print(f"Random seeds: {len(random_seeds)}")
    print("=" * 60)

    synthetic = load_synthetic_traces()
    swe_traces = load_swe_agent_data(max_traces=args.max_swe_traces)
    atbench_traces = load_atbench_traces()

    source_data = compute_features(synthetic, "label")
    print(f"\nSource (Synthetic): {len(synthetic)} traces")
    source_d = compute_domain_cohens_d(source_data["features"], source_data["labels"])
    for f, d in source_d.items():
        print(f"  {f}: d={d:+.3f}")

    targets = [
        ("SWE-agent", swe_traces, "label"),
        ("ATBench", atbench_traces, "label"),
    ]

    all_results = {"source_domain": "synthetic", "source_d": source_d, "targets": {}}

    for target_name, target_traces, label_key in targets:
        print(f"\n{'='*60}")
        print(f"Target: {target_name} ({len(target_traces)} traces)")
        print("=" * 60)

        target_results = {"n_traces": len(target_traces), "seed_sizes": {}}

        for seed_size in args.seed_sizes:
            print(f"\n  Seed size: {seed_size}")
            seed_results = []

            for rs in random_seeds:
                set_seeds(rs)
                rng = np.random.default_rng(rs)
                result = run_pafc_single_seed(
                    source_data, target_traces, label_key, seed_size, rng,
                )
                if "error" not in result:
                    seed_results.append(result)
                    print(
                        f"    seed={rs}: naive={result['naive_auroc']:.3f} "
                        f"pafc={result['pafc_auroc']:.3f} "
                        f"oracle={result['oracle_auroc']:.3f} "
                        f"delta={result['delta_pafc']:+.3f} "
                        f"reversed={result['reversed_features']}"
                    )
                else:
                    print(f"    seed={rs}: {result['error']}")

            if seed_results:
                naive_arr = np.array([r["naive_auroc"] for r in seed_results])
                pafc_arr = np.array([r["pafc_auroc"] for r in seed_results])
                oracle_arr = np.array([r["oracle_auroc"] for r in seed_results])
                delta_arr = pafc_arr - naive_arr

                n_improved = int((delta_arr > 0).sum())
                n_hurt = int((delta_arr < 0).sum())
                n_same = int((delta_arr == 0).sum())

                if len(delta_arr) >= 5:
                    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                        pafc_arr, naive_arr, alternative="greater",
                    )
                else:
                    wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

                reversal_counts: dict[str, int] = {}
                for r in seed_results:
                    for rf in r["reversed_features"]:
                        reversal_counts[rf] = reversal_counts.get(rf, 0) + 1

                summary = {
                    "seed_size": seed_size,
                    "n_seeds": len(seed_results),
                    "naive_auroc_mean": float(naive_arr.mean()),
                    "naive_auroc_std": float(naive_arr.std()),
                    "pafc_auroc_mean": float(pafc_arr.mean()),
                    "pafc_auroc_std": float(pafc_arr.std()),
                    "oracle_auroc_mean": float(oracle_arr.mean()),
                    "oracle_auroc_std": float(oracle_arr.std()),
                    "delta_mean": float(delta_arr.mean()),
                    "delta_std": float(delta_arr.std()),
                    "n_improved": n_improved,
                    "n_hurt": n_hurt,
                    "n_same": n_same,
                    "wilcoxon_p": float(wilcoxon_p),
                    "reversal_detection_rate": {
                        fname: count / len(seed_results)
                        for fname, count in reversal_counts.items()
                    },
                    "per_seed": seed_results,
                }

                target_results["seed_sizes"][str(seed_size)] = summary

                print(f"\n  --- Seed size {seed_size} summary ---")
                print(f"  Naive:  {naive_arr.mean():.3f} ± {naive_arr.std():.3f}")
                print(f"  PAFC:   {pafc_arr.mean():.3f} ± {pafc_arr.std():.3f}")
                print(f"  Oracle: {oracle_arr.mean():.3f} ± {oracle_arr.std():.3f}")
                print(f"  Delta:  {delta_arr.mean():+.3f} ± {delta_arr.std():.3f}")
                print(f"  Improved/Hurt/Same: {n_improved}/{n_hurt}/{n_same}")
                print(f"  Wilcoxon p (PAFC > naive): {wilcoxon_p:.4f}")
                print(f"  Reversal detection: {reversal_counts}")

        all_results["targets"][target_name] = target_results

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    out_path = OUTPUT_DIR / "pafc_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("PAFC EVALUATION SUMMARY")
    print("=" * 60)
    for tname, tres in all_results["targets"].items():
        print(f"\n{tname}:")
        for ss, sdata in tres["seed_sizes"].items():
            delta = sdata["delta_mean"]
            p = sdata["wilcoxon_p"]
            marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(
                f"  N={ss:>3}: naive={sdata['naive_auroc_mean']:.3f} "
                f"→ pafc={sdata['pafc_auroc_mean']:.3f} "
                f"(Δ={delta:+.3f}{marker}) "
                f"oracle={sdata['oracle_auroc_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
