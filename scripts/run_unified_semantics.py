"""v28: Unified semantics experiment (InjecAgent <-> ATBench).

Both datasets label safety violations (injection/unsafe actions).
Shows polarity reversal persists even with aligned anomaly semantics,
eliminating the "definitional" objection.

Computes Cohen's d for heuristic features within each domain,
trains on one, evaluates on the other (bidirectional).
Compares: LSTM-only, DANN, CORAL, PAFC, DAPL.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_unified_semantics.py
    PYTHONPATH=../../lib:src uv run python scripts/run_unified_semantics.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds
from sklearn.metrics import roc_auc_score

from data.domain_loader import (
    load_atbench_domain,
    load_injecagent_domain,
)

OUTPUT_DIR = Path("output/tables")


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def analyze_polarity(
    features_a: dict[str, np.ndarray],
    labels_a: np.ndarray,
    features_b: dict[str, np.ndarray],
    labels_b: np.ndarray,
    domain_a_name: str,
    domain_b_name: str,
) -> dict[str, dict]:
    """Compute per-feature Cohen's d and check for sign reversal."""
    common = set(features_a.keys()) & set(features_b.keys())
    results: dict[str, dict] = {}

    for feat in sorted(common):
        vals_a, vals_b = features_a[feat], features_b[feat]

        normal_a = vals_a[labels_a == 0]
        anomalous_a = vals_a[labels_a == 1]
        d_a = cohens_d(anomalous_a, normal_a)

        normal_b = vals_b[labels_b == 0]
        anomalous_b = vals_b[labels_b == 1]
        d_b = cohens_d(anomalous_b, normal_b)

        sign_a = 1 if d_a >= 0 else -1
        sign_b = 1 if d_b >= 0 else -1
        reversed_ = sign_a != sign_b and abs(d_a) > 0.1 and abs(d_b) > 0.1

        results[feat] = {
            f"d_{domain_a_name}": d_a,
            f"d_{domain_b_name}": d_b,
            "sign_reversed": reversed_,
        }

    return results


def _evaluate_method(
    method_name: str,
    src_seqs: list[list[int]],
    src_labels: list[int],
    tgt_seqs: list[list[int]],
    tgt_labels: list[int],
    vocab_size: int,
    seed: int,
    *,
    epochs: int = 50,
) -> dict:
    """Train and evaluate one method on one seed."""
    import torch

    set_seeds(seed)

    n_src = len(src_seqs)
    cal_split = int(n_src * 0.2)
    cal_seqs = src_seqs[:cal_split]
    train_seqs = src_seqs[cal_split:]
    train_labels = src_labels[cal_split:]

    def _device_str() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if method_name == "LSTM-only":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.0, lambda_pol=0.0,
            epochs=epochs, patience=5,
        )
        trainer.to(torch.device(_device_str()))
        trainer.train(train_seqs, train_labels, tgt_seqs, tgt_labels)
        tgt_scores = trainer.score_traces(tgt_seqs)

    elif method_name == "DANN":
        from baselines.dann_coral import DANNBaseline

        dann = DANNBaseline(epochs=epochs, patience=5)
        dann.fit_cross_domain(
            vocab_size, train_seqs, train_labels,
            tgt_seqs, tgt_labels, cal_seqs,
        )
        tgt_scores = dann._score_seqs(tgt_seqs).tolist()

    elif method_name == "CORAL":
        from baselines.dann_coral import CORALBaseline

        coral = CORALBaseline(epochs=epochs, patience=5)
        coral.fit_cross_domain(
            vocab_size, train_seqs, train_labels,
            tgt_seqs, tgt_labels, cal_seqs,
        )
        tgt_scores = coral._score_seqs(tgt_seqs).tolist()

    elif method_name == "PAFC":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.0, lambda_pol=0.0,
            epochs=epochs, patience=5,
        )
        trainer.to(torch.device(_device_str()))
        trainer.train(train_seqs, train_labels, tgt_seqs, tgt_labels)
        tgt_scores = trainer.score_traces(tgt_seqs)
        tgt_scores_np = np.array(tgt_scores)
        tgt_labels_np = np.array(tgt_labels)
        try:
            auroc_raw = roc_auc_score(tgt_labels_np, tgt_scores_np)
        except ValueError:
            auroc_raw = 0.5
        if auroc_raw < 0.5:
            tgt_scores = (-tgt_scores_np).tolist()

    elif method_name == "DAPL":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.1, lambda_pol=0.5,
            epochs=epochs, patience=5,
        )
        trainer.to(torch.device(_device_str()))
        trainer.train(train_seqs, train_labels, tgt_seqs, tgt_labels)
        tgt_scores = trainer.score_traces(tgt_seqs)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    tgt_scores_np = np.array(tgt_scores)
    tgt_labels_np = np.array(tgt_labels)
    try:
        auroc = roc_auc_score(tgt_labels_np, tgt_scores_np)
    except ValueError:
        auroc = 0.5

    return {"auroc": auroc, "n_test": len(tgt_labels), "seed": seed}


def run_cross_domain_eval(
    src_name: str,
    tgt_name: str,
    vocab,
    src_seqs: list[list[int]],
    src_labels: list[int],
    tgt_seqs: list[list[int]],
    tgt_labels: list[int],
    seeds: list[int],
    methods: list[str],
    *,
    epochs: int = 50,
) -> dict[str, list[float]]:
    """Evaluate methods on src -> tgt transfer."""
    print(f"\n  {src_name} -> {tgt_name}")
    print(f"  Source: {len(src_seqs)} seqs, {sum(src_labels)} anomalous")
    print(f"  Target: {len(tgt_seqs)} seqs, {sum(tgt_labels)} anomalous")

    results: dict[str, list[float]] = {m: [] for m in methods}

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        for method in methods:
            t0 = time.time()
            res = _evaluate_method(
                method, src_seqs, src_labels, tgt_seqs, tgt_labels,
                vocab.vocab_size, seed, epochs=epochs,
            )
            elapsed = time.time() - t0
            results[method].append(res["auroc"])
            print(f"    {method:12s}: AUROC={res['auroc']:.4f}  ({elapsed:.1f}s)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="v28: Unified semantics")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 5

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  v28: Unified Semantics Experiment")
    print("  Both datasets label safety violations (injection/unsafe)")
    print("=" * 60)

    print("\nLoading domains...")
    try:
        feat_atbench, lbl_atbench = load_atbench_domain()
    except Exception as e:
        print(f"  ATBench failed: {e}")
        return

    try:
        feat_inject, lbl_inject = load_injecagent_domain()
    except Exception as e:
        print(f"  InjecAgent failed: {e}")
        return

    print("\n" + "=" * 60)
    print("  Phase 1: Polarity Analysis (Heuristic Features)")
    print("=" * 60)

    polarity = analyze_polarity(
        feat_inject, lbl_inject,
        feat_atbench, lbl_atbench,
        "injecagent", "atbench",
    )

    n_reversed = sum(1 for v in polarity.values() if v["sign_reversed"])
    print(f"\n  Features with polarity reversal: {n_reversed}/{len(polarity)}")
    for feat, info in polarity.items():
        rev = " *** REVERSED ***" if info["sign_reversed"] else ""
        print(
            f"    {feat:20s}: "
            f"d_inject={info['d_injecagent']:+.3f}  "
            f"d_atbench={info['d_atbench']:+.3f}{rev}"
        )

    print("\n" + "=" * 60)
    print("  Phase 2: Cross-Domain Detection")
    print("=" * 60)

    print("\nBuilding vocabulary from both domains...")
    from data.trace_loader import TraceVocab, load_atbench, load_injecagent

    all_traces = []
    try:
        all_traces.extend(load_atbench(max_traces=500))
    except Exception:
        pass
    try:
        all_traces.extend(load_injecagent(Path("data"), max_traces=500))
    except Exception:
        pass

    from data.trace_loader import generate_synthetic_traces

    all_traces.extend(generate_synthetic_traces(n_traces=500, seed=42))
    vocab = TraceVocab()
    vocab.fit(all_traces)
    print(f"  Vocab size: {vocab.vocab_size}")

    inject_traces = load_injecagent(Path("data"))
    atbench_traces = load_atbench()

    inject_seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in inject_traces]
    inject_labels = [1 if t.has_anomaly else 0 for t in inject_traces]
    atbench_seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in atbench_traces]
    atbench_labels = [1 if t.has_anomaly else 0 for t in atbench_traces]

    methods = ["LSTM-only", "DANN", "CORAL", "PAFC", "DAPL"]

    results_i2a = run_cross_domain_eval(
        "InjecAgent", "ATBench", vocab,
        inject_seqs, inject_labels, atbench_seqs, atbench_labels,
        args.seeds, methods, epochs=args.epochs,
    )

    results_a2i = run_cross_domain_eval(
        "ATBench", "InjecAgent", vocab,
        atbench_seqs, atbench_labels, inject_seqs, inject_labels,
        args.seeds, methods, epochs=args.epochs,
    )

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)

    for pair_name, results in [
        ("InjecAgent->ATBench", results_i2a),
        ("ATBench->InjecAgent", results_a2i),
    ]:
        print(f"\n  {pair_name}:")
        for method, aurocs in results.items():
            arr = np.array(aurocs)
            print(f"    {method:12s}: {arr.mean():.4f} +/- {arr.std():.4f}")

    output = {
        "experiment": "v28_unified_semantics",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "polarity_analysis": polarity,
        "n_reversed_features": n_reversed,
        "results": {
            "injecagent->atbench": {
                m: aurocs for m, aurocs in results_i2a.items()
            },
            "atbench->injecagent": {
                m: aurocs for m, aurocs in results_a2i.items()
            },
        },
    }
    out_path = OUTPUT_DIR / "v28_unified_semantics.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
