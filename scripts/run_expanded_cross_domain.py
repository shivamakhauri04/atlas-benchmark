"""v30: Expanded cross-domain scoring on all dataset pairs with polarity reversal.

Runs LSTM, Heuristic(raw), Heuristic(PAFC), Hybrid on cross-domain pairs
where polarity reversal exists. Includes ToolBench and TRAIL datasets.
Builds expanded scorer x dataset AUROC matrix.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_expanded_cross_domain.py
    PYTHONPATH=../../lib:src uv run python scripts/run_expanded_cross_domain.py --quick
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
    load_all_domains,
)
from data.trace_loader import (
    TraceVocab,
    generate_synthetic_traces,
    load_agentharm,
    load_atbench,
)
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]


def _get_device_str() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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
        normal = vals[labels == 0]
        anomalous = vals[labels == 1]
        result[feat] = cohens_d(anomalous, normal)
    return result


def apply_pafc_correction(
    features: dict[str, np.ndarray],
    source_d: dict[str, float],
    target_d: dict[str, float],
) -> dict[str, np.ndarray]:
    """Negate features where source and target Cohen's d signs differ."""
    corrected = {}
    for name, values in features.items():
        d_src = source_d.get(name, 0.0)
        d_tgt = target_d.get(name, 0.0)
        if abs(d_src) > 0.1 and abs(d_tgt) > 0.1 and (d_src > 0) != (d_tgt > 0):
            corrected[name] = -values
        else:
            corrected[name] = values
    return corrected


def heuristic_score(features: dict[str, np.ndarray]) -> np.ndarray:
    """Z-score + mean of heuristic features."""
    cols = []
    for name in sorted(features.keys()):
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        cols.append((vals - mu) / sigma)
    return np.mean(np.column_stack(cols), axis=1)


def evaluate_scorers_cross_domain(
    src_features: dict[str, np.ndarray],
    src_labels: np.ndarray,
    tgt_features: dict[str, np.ndarray],
    tgt_labels: np.ndarray,
    src_seqs: list[list[int]],
    src_labels_seq: list[int],
    tgt_seqs: list[list[int]],
    tgt_labels_seq: list[int],
    vocab_size: int,
    seed: int,
    *,
    epochs: int = 30,
) -> dict[str, float]:
    """Evaluate all scorers for one seed on one domain pair."""
    import torch

    set_seeds(seed)
    results: dict[str, float] = {}

    src_d = compute_domain_d(src_features, src_labels)
    tgt_d = compute_domain_d(tgt_features, tgt_labels)

    raw_scores = heuristic_score(tgt_features)
    try:
        results["heuristic_raw"] = float(roc_auc_score(tgt_labels, raw_scores))
    except ValueError:
        results["heuristic_raw"] = 0.5

    corrected_tgt = apply_pafc_correction(tgt_features, src_d, tgt_d)
    pafc_scores = heuristic_score(corrected_tgt)
    try:
        results["heuristic_pafc"] = float(roc_auc_score(tgt_labels, pafc_scores))
    except ValueError:
        results["heuristic_pafc"] = 0.5

    try:
        from trustlens.dapl import DAPLTrainer

        n_src = len(src_seqs)
        train_seqs = src_seqs[int(n_src * 0.2):]
        train_labels = src_labels_seq[int(n_src * 0.2):]

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.0, lambda_pol=0.0,
            epochs=epochs, patience=5,
        )
        trainer.to(torch.device(_get_device_str()))
        trainer.train(train_seqs, train_labels, tgt_seqs, tgt_labels_seq)
        lstm_scores = np.array(trainer.score_traces(tgt_seqs))

        try:
            results["lstm"] = float(roc_auc_score(
                np.array(tgt_labels_seq), lstm_scores,
            ))
        except ValueError:
            results["lstm"] = 0.5

        lstm_z = (lstm_scores - lstm_scores.mean()) / max(lstm_scores.std(), 1e-8)
        n_tgt = len(tgt_labels)
        if len(lstm_z) == n_tgt:
            hybrid_scores = (lstm_z + pafc_scores) / 2.0
            try:
                results["hybrid_pafc"] = float(roc_auc_score(tgt_labels, hybrid_scores))
            except ValueError:
                results["hybrid_pafc"] = 0.5
        else:
            results["hybrid_pafc"] = float("nan")

    except Exception as e:
        print(f"      LSTM/hybrid failed: {e}")
        results["lstm"] = float("nan")
        results["hybrid_pafc"] = float("nan")

    return results


def load_domain_traces_and_seqs(
    domain_name: str,
    vocab: TraceVocab,
    *,
    max_traces: int = 5000,
) -> tuple[list, list[list[int]], list[int]]:
    """Load traces, encode to sequences, return (traces, seqs, labels)."""
    import random

    from data.trace_loader import (
        _inject_anomaly,
        download_trail,
        generate_synthetic_traces,
        load_agentdojo,
        load_agentharm,
        load_atbench,
        load_injecagent,
        load_rjudge,
        load_swe_agent,
        load_toolbench,
        load_trail,
    )

    if domain_name == "synthetic":
        traces = generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42)
    elif domain_name == "atbench":
        traces = load_atbench(max_traces=max_traces)
    elif domain_name == "agentharm":
        traces = load_agentharm(max_traces=max_traces)
    elif domain_name == "injecagent":
        traces = load_injecagent(Path("data"), max_traces=max_traces)
    elif domain_name == "rjudge":
        traces = load_rjudge(Path("data"), max_traces=max_traces)
    elif domain_name == "swe_agent":
        traces = load_swe_agent(max_traces=max_traces)
    elif domain_name == "toolbench":
        raw_traces = load_toolbench(Path("data"), max_traces=max_traces)
        rng = random.Random(42)
        traces = []
        for trace in raw_traces:
            if rng.random() < 0.10:
                from data.schema import Trace

                new_actions = _inject_anomaly(list(trace.actions), rng)
                traces.append(Trace(
                    trace_id=trace.trace_id,
                    actions=new_actions,
                    source=trace.source,
                    task=trace.task,
                ))
            else:
                traces.append(trace)
    elif domain_name == "trail":
        trail_dir = Path("data/trail/TRAIL-Benchmark")
        if not trail_dir.exists():
            trail_dir = download_trail(trail_dir.parent)
        traces = load_trail(trail_dir)
    elif domain_name == "agentdojo":
        traces = load_agentdojo(
            Path("/tmp/agentdojo_data/runs/gpt-4o-2024-05-13"),
            max_traces=max_traces,
        )
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in traces]
    labels = [1 if t.has_anomaly else 0 for t in traces]
    return traces, seqs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="v30: Expanded cross-domain scoring")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 5
        args.max_traces = 200

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  v30: Expanded Cross-Domain Scoring (8 datasets)")
    print("=" * 70)

    print("\nLoading domains for feature analysis...")
    domain_data = load_all_domains(max_swe_traces=args.max_traces)
    loaded = sorted(domain_data.keys())
    print(f"  Loaded: {loaded}")

    domain_d: dict[str, dict[str, float]] = {}
    for name, (features, labels) in domain_data.items():
        domain_d[name] = compute_domain_d(features, labels)

    reversed_pairs: list[tuple[str, str]] = []
    for da in loaded:
        for db in loaded:
            if da >= db:
                continue
            n_reversed = sum(
                1 for f in FEATURE_NAMES
                if abs(domain_d[da].get(f, 0)) > 0.1
                and abs(domain_d[db].get(f, 0)) > 0.1
                and (domain_d[da][f] > 0) != (domain_d[db][f] > 0)
            )
            if n_reversed > 0:
                reversed_pairs.append((da, db))
                reversed_pairs.append((db, da))

    if not reversed_pairs:
        print("\n  No polarity reversal detected between any domain pair!")
        print("  Adding default cross-domain pairs for comparison...")
        for da in loaded:
            for db in loaded:
                if da != db and da == "synthetic":
                    reversed_pairs.append((da, db))

    unique_pairs = list(dict.fromkeys(reversed_pairs))
    print(f"\n  Cross-domain pairs to evaluate: {len(unique_pairs)}")
    for src, tgt in unique_pairs:
        print(f"    {src} -> {tgt}")

    print("\nBuilding shared vocabulary from all loaded domains...")
    from data.schema import Action, Trace

    vocab_traces: list[Trace] = []
    vocab_traces.extend(
        generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42),
    )
    for name in loaded:
        if name == "synthetic":
            continue
        features, labels = domain_data[name]
        n = min(500, len(labels))
        dummy_traces = [
            Trace(
                trace_id=f"vocab_{name}_{i}",
                actions=[Action(tool=f"tool_{name}_{i}", args="", context="", timestamp=0.0)],
                source=name,
                task="vocab",
            )
            for i in range(n)
        ]
        vocab_traces.extend(dummy_traces)
    try:
        if "atbench" in loaded:
            vocab_traces.extend(load_atbench(max_traces=500))
        if "agentharm" in loaded:
            vocab_traces.extend(load_agentharm(max_traces=500))
        if "toolbench" in loaded:
            from data.trace_loader import load_toolbench
            vocab_traces.extend(load_toolbench(Path("data"), max_traces=500))
        if "trail" in loaded:
            from data.trace_loader import download_trail, load_trail
            trail_dir = Path("data/trail/TRAIL-Benchmark")
            if not trail_dir.exists():
                trail_dir = download_trail(trail_dir.parent)
            vocab_traces.extend(load_trail(trail_dir))
        if "agentdojo" in loaded:
            from data.trace_loader import load_agentdojo
            vocab_traces.extend(load_agentdojo(
                Path("/tmp/agentdojo_data/runs/gpt-4o-2024-05-13"),
                max_traces=500,
            ))
    except Exception as e:
        print(f"  Vocab extension failed for some domains: {e}")
    vocab = TraceVocab()
    vocab.fit(vocab_traces)
    print(f"  Vocab size: {vocab.vocab_size}")

    all_results: dict[str, dict[str, list[float]]] = {}

    for src_name, tgt_name in unique_pairs:
        pair_key = f"{src_name}->{tgt_name}"
        print(f"\n{'='*60}")
        print(f"  {pair_key}")
        print(f"{'='*60}")

        src_feat, src_lbl = domain_data[src_name]
        tgt_feat, tgt_lbl = domain_data[tgt_name]

        try:
            _, src_seqs, src_labels_seq = load_domain_traces_and_seqs(
                src_name, vocab, max_traces=args.max_traces,
            )
            _, tgt_seqs, tgt_labels_seq = load_domain_traces_and_seqs(
                tgt_name, vocab, max_traces=args.max_traces,
            )
        except Exception as e:
            print(f"  Skipping (sequence load failed): {e}")
            continue

        scorer_results: dict[str, list[float]] = {}
        for seed in args.seeds:
            t0 = time.time()
            seed_res = evaluate_scorers_cross_domain(
                src_feat, src_lbl, tgt_feat, tgt_lbl,
                src_seqs, src_labels_seq,
                tgt_seqs, tgt_labels_seq,
                vocab.vocab_size, seed,
                epochs=args.epochs,
            )
            elapsed = time.time() - t0
            print(f"  Seed {seed} ({elapsed:.1f}s):", end="")
            for scorer, auroc in seed_res.items():
                scorer_results.setdefault(scorer, []).append(auroc)
                print(f"  {scorer}={auroc:.3f}", end="")
            print()

        all_results[pair_key] = scorer_results

        print(f"\n  Summary for {pair_key}:")
        for scorer, aurocs in scorer_results.items():
            arr = np.array([v for v in aurocs if np.isfinite(v)])
            if len(arr) > 0:
                print(f"    {scorer:20s}: {arr.mean():.4f} +/- {arr.std():.4f}")

    output = {
        "experiment": "v30_expanded_cross_domain",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "domain_pairs": [f"{s}->{t}" for s, t in unique_pairs],
        "domain_d_values": domain_d,
        "results": {
            pair: {s: aurocs for s, aurocs in res.items()}
            for pair, res in all_results.items()
        },
    }
    out_path = OUTPUT_DIR / "v30_expanded_cross_domain.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
