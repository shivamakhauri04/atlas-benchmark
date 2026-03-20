"""LSTM hidden-state polarity analysis on raw tool sequences.

Unlike the autoencoder analysis (which transforms heuristic features),
this trains a SINGLE shared LSTMEncoder on synthetic traces (the
reference domain) using raw tool-call sequences, then extracts 128-dim
mean-pooled hidden states for ALL domains through that shared encoder.
This ensures dimensions are comparable across domains.

These are genuinely independent learned representations — no heuristic
feature input at any point.

Computes Cohen's d per hidden dimension, then analyzes within-group
vs between-group polarity reversal rates across all 8 datasets.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_lstm_polarity.py
    PYTHONPATH=../../lib:src uv run python scripts/run_lstm_polarity.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research_utils.reproducibility import set_seeds
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

SEMANTIC_GROUPS: dict[str, list[str]] = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "rjudge", "agentharm", "injecagent"],
    "api_misuse": ["toolbench"],
}

DOMAIN_TO_GROUP: dict[str, str] = {
    d: g for g, ds in SEMANTIC_GROUPS.items() for d in ds
}

ALL_DOMAINS = [
    "synthetic", "swe_agent", "trail", "atbench",
    "rjudge", "agentharm", "injecagent", "toolbench",
]


def _get_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_domain_sequences(
    domain_name: str,
    vocab,
    *,
    max_traces: int = 5000,
) -> tuple[list[list[int]], list[int]]:
    """Load a domain's traces, encode to tool-ID sequences, return (seqs, labels)."""
    import random

    from data.trace_loader import (
        _inject_anomaly,
        download_trail,
        generate_synthetic_traces,
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
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in traces]
    labels = [1 if t.has_anomaly else 0 for t in traces]
    return seqs, labels


def build_vocab(max_traces: int = 5000):
    """Build a shared TraceVocab across all domains."""
    from data.trace_loader import (
        TraceVocab,
        generate_synthetic_traces,
        load_agentharm,
        load_atbench,
        load_injecagent,
        load_rjudge,
        load_swe_agent,
        load_toolbench,
    )

    all_traces = []
    all_traces.extend(generate_synthetic_traces(
        n_traces=2000, anomaly_rate=0.15, seed=42,
    ))

    loaders = {
        "atbench": lambda: load_atbench(max_traces=max_traces),
        "agentharm": lambda: load_agentharm(max_traces=max_traces),
        "rjudge": lambda: load_rjudge(Path("data"), max_traces=max_traces),
        "injecagent": lambda: load_injecagent(Path("data"), max_traces=max_traces),
        "swe_agent": lambda: load_swe_agent(max_traces=max_traces),
        "toolbench": lambda: load_toolbench(Path("data"), max_traces=max_traces),
    }
    for name, loader in loaders.items():
        try:
            all_traces.extend(loader())
        except Exception as e:
            print(f"  Warning: could not load {name} for vocab: {e}")

    try:
        from data.trace_loader import download_trail, load_trail
        trail_dir = Path("data/trail/TRAIL-Benchmark")
        if not trail_dir.exists():
            trail_dir = download_trail(trail_dir.parent)
        all_traces.extend(load_trail(trail_dir))
    except Exception as e:
        print(f"  Warning: could not load trail for vocab: {e}")

    vocab = TraceVocab()
    vocab.fit(all_traces)
    return vocab


def train_shared_lstm_encoder(
    ref_seqs: list[list[int]],
    ref_labels: list[int],
    vocab_size: int,
    *,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
) -> "torch.nn.Module":
    """Train a shared LSTM encoder on the reference domain (synthetic).

    Uses self-supervised NLL (next-token prediction) so the encoder
    learns general tool-sequence structure, not domain-specific patterns.
    """
    from trustlens.dapl import DAPLTrainer

    set_seeds(seed)
    device = torch.device(_get_device_str())

    trainer = DAPLTrainer(
        vocab_size,
        lambda_adv=0.0,
        lambda_pol=0.0,
        epochs=epochs,
        patience=patience,
    )
    trainer.to(device)

    # Train on synthetic as both source and target (self-supervised)
    trainer.train(ref_seqs, ref_labels, ref_seqs, ref_labels)

    return trainer.encoder


def extract_features(
    encoder: "torch.nn.Module",
    seqs: list[list[int]],
    batch_size: int = 64,
) -> np.ndarray:
    """Extract 128-dim mean-pooled hidden states for all sequences."""
    device = next(encoder.parameters()).device
    encoder.eval()
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            max_len = max(len(s) for s in batch_seqs)
            ids = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
            mask = torch.zeros(len(batch_seqs), max_len, dtype=torch.bool)
            for j, s in enumerate(batch_seqs):
                ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[j, : len(s)] = True
            ids = ids.to(device)
            mask = mask.to(device)
            feats = encoder.trace_features(ids, mask)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def compute_domain_d_vector(
    features: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Cohen's d per dimension. Returns (D,) array."""
    d_hidden = features.shape[1]
    d_vec = np.zeros(d_hidden)
    normal = labels == 0
    anomalous = labels == 1
    if normal.sum() < 2 or anomalous.sum() < 2:
        return d_vec * float("nan")
    for dim in range(d_hidden):
        d_vec[dim] = cohens_d(
            features[anomalous, dim],
            features[normal, dim],
        )
    return d_vec


def count_reversals(
    d_a: np.ndarray,
    d_b: np.ndarray,
    threshold: float = 0.1,
) -> tuple[int, int]:
    """Count dimensions with polarity reversal between two d-vectors.

    Returns (n_reversed, n_comparable) where comparable means
    both |d| > threshold.
    """
    n_reversed = 0
    n_comparable = 0
    for i in range(len(d_a)):
        if np.isnan(d_a[i]) or np.isnan(d_b[i]):
            continue
        if abs(d_a[i]) > threshold and abs(d_b[i]) > threshold:
            n_comparable += 1
            if np.sign(d_a[i]) != np.sign(d_b[i]):
                n_reversed += 1
    return n_reversed, n_comparable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSTM hidden-state polarity analysis",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42, 123]
        args.epochs = 10
        args.max_traces = 500

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  LSTM Hidden-State Polarity Analysis (128-dim, raw tool sequences)")
    print(f"  Seeds: {len(args.seeds)}, Epochs: {args.epochs}")
    print("=" * 70)

    print("\nBuilding shared vocabulary...")
    vocab = build_vocab(max_traces=args.max_traces)
    print(f"  Vocab size: {vocab.vocab_size}")

    # Load all domain sequences
    print("\nLoading domain sequences...")
    domain_seqs: dict[str, tuple[list[list[int]], list[int]]] = {}
    for domain in ALL_DOMAINS:
        try:
            seqs, labels = load_domain_sequences(
                domain, vocab, max_traces=args.max_traces,
            )
            n_anom = sum(labels)
            n_norm = len(labels) - n_anom
            print(f"  {domain:15s}: {len(seqs)} traces ({n_norm} normal, {n_anom} anomalous)")
            if n_anom < 2 or n_norm < 2:
                print(f"    Skipping {domain}: insufficient class balance")
                continue
            domain_seqs[domain] = (seqs, labels)
        except Exception as e:
            print(f"  {domain:15s}: FAILED ({e})")

    loaded_domains = sorted(domain_seqs.keys())
    print(f"\nLoaded {len(loaded_domains)} domains: {loaded_domains}")

    if "synthetic" not in domain_seqs:
        print("ERROR: synthetic domain required as reference. Aborting.")
        return

    # Per-seed: train ONE shared encoder on synthetic, extract features for ALL domains
    all_seed_d_vectors: list[dict[str, np.ndarray]] = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        ref_seqs, ref_labels = domain_seqs["synthetic"]
        print(f"\n  Training shared LSTM on synthetic ({len(ref_seqs)} seqs)...")
        st = time.time()
        encoder = train_shared_lstm_encoder(
            ref_seqs, ref_labels, vocab.vocab_size,
            epochs=args.epochs, seed=seed,
        )
        print(f"    Encoder trained ({time.time() - st:.1f}s)")

        seed_d: dict[str, np.ndarray] = {}

        for domain in loaded_domains:
            seqs, labels = domain_seqs[domain]
            features = extract_features(encoder, seqs)
            d_vec = compute_domain_d_vector(features, np.array(labels))
            seed_d[domain] = d_vec

            abs_d = np.abs(d_vec)
            top5 = np.argsort(abs_d)[::-1][:5]
            top_str = ", ".join(f"h{i}={d_vec[i]:+.2f}" for i in top5)
            print(f"    {domain:15s}: top dims: {top_str}")

        # Cleanup GPU memory
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_seed_d_vectors.append(seed_d)

    # --- Polarity analysis: cosine similarity of d-vectors ---
    # Cosine similarity measures whether two domains' anomaly structure
    # points in the same direction. Positive = similar polarity, negative = reversed.
    # This is more robust than per-dimension reversal counting for high-dim features.
    print(f"\n{'='*70}")
    print("  POLARITY ANALYSIS: COSINE SIMILARITY OF d-VECTORS")
    print(f"{'='*70}")

    pairs = []
    for i, da in enumerate(loaded_domains):
        for db in loaded_domains[i + 1 :]:
            pairs.append((da, db))

    within_cosines: list[float] = []
    between_cosines: list[float] = []
    pair_results: dict[str, dict] = {}

    for da, db in pairs:
        group_a = DOMAIN_TO_GROUP.get(da, "unknown")
        group_b = DOMAIN_TO_GROUP.get(db, "unknown")
        is_within = group_a == group_b

        seed_cosines = []
        seed_reversals = []
        seed_comparables = []

        for seed_d in all_seed_d_vectors:
            if da not in seed_d or db not in seed_d:
                continue
            d_a = seed_d[da]
            d_b = seed_d[db]

            # Filter out NaN dimensions
            valid = ~(np.isnan(d_a) | np.isnan(d_b))
            da_v = d_a[valid]
            db_v = d_b[valid]

            # Cosine similarity
            norm_a = np.linalg.norm(da_v)
            norm_b = np.linalg.norm(db_v)
            if norm_a > 1e-10 and norm_b > 1e-10:
                cos = float(np.dot(da_v, db_v) / (norm_a * norm_b))
                seed_cosines.append(cos)

            # Per-dim reversal (secondary metric)
            n_rev, n_comp = count_reversals(d_a, d_b)
            seed_reversals.append(n_rev)
            seed_comparables.append(n_comp)

        if not seed_cosines:
            continue

        avg_cos = float(np.mean(seed_cosines))
        std_cos = float(np.std(seed_cosines))
        avg_rev = float(np.mean(seed_reversals))
        avg_comp = float(np.mean(seed_comparables))
        rev_rate = avg_rev / max(avg_comp, 1)

        pair_results[f"{da}-{db}"] = {
            "group_a": group_a,
            "group_b": group_b,
            "is_within_group": is_within,
            "cosine_mean": avg_cos,
            "cosine_std": std_cos,
            "per_seed_cosines": seed_cosines,
            "reversal_rate": float(rev_rate),
        }

        if is_within:
            within_cosines.append(avg_cos)
        else:
            between_cosines.append(avg_cos)

    # Print pair-level results
    print(f"\n  {'Pair':<30s} {'Group':<20s} {'Type':<10s} {'Cosine':>8s} {'RevRate':>8s}")
    print("  " + "-" * 80)
    for pair_key, pr in sorted(pair_results.items()):
        ptype = "WITHIN" if pr["is_within_group"] else "between"
        gstr = f"{pr['group_a']}/{pr['group_b']}"
        print(
            f"  {pair_key:<30s} {gstr:<20s} {ptype:<10s} "
            f"{pr['cosine_mean']:>+8.3f} {pr['reversal_rate']:>8.1%}"
        )

    within_cos_mean = float(np.mean(within_cosines)) if within_cosines else float("nan")
    between_cos_mean = float(np.mean(between_cosines)) if between_cosines else float("nan")

    print(f"\n  COSINE SIMILARITY SUMMARY:")
    print(f"    Within-group ({len(within_cosines)} pairs):  mean cos = {within_cos_mean:+.3f}")
    print(f"    Between-group ({len(between_cosines)} pairs): mean cos = {between_cos_mean:+.3f}")

    # Interpretation
    if within_cos_mean > 0 and between_cos_mean < within_cos_mean:
        print(f"    Gap: within - between = {within_cos_mean - between_cos_mean:.3f}")
        print("    -> Within-group anomalies are more structurally similar (as expected)")

    # Count negative-cosine pairs (polarity reversal) in each group
    within_neg = sum(1 for c in within_cosines if c < 0)
    between_neg = sum(1 for c in between_cosines if c < 0)
    within_neg_rate = within_neg / max(len(within_cosines), 1)
    between_neg_rate = between_neg / max(len(between_cosines), 1)

    print(f"\n  NEGATIVE COSINE (POLARITY REVERSAL):")
    print(f"    Within-group:  {within_neg}/{len(within_cosines)} = {within_neg_rate:.1%}")
    print(f"    Between-group: {between_neg}/{len(between_cosines)} = {between_neg_rate:.1%}")

    # Statistical test: Mann-Whitney on cosines (within > between)
    mw_p_cosine = float("nan")
    if len(within_cosines) >= 3 and len(between_cosines) >= 3:
        try:
            from scipy.stats import mannwhitneyu
            _, mw_p_cosine = mannwhitneyu(
                within_cosines, between_cosines, alternative="greater",
            )
            print(f"    Mann-Whitney U (within cos > between cos): p = {mw_p_cosine:.4f}")
        except Exception:
            pass

    elapsed = time.time() - t0

    output = {
        "experiment": "v30_lstm_polarity",
        "d_hidden": 128,
        "input_type": "raw_tool_sequences",
        "encoder_training": "shared_on_synthetic",
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "loaded_domains": loaded_domains,
        "n_pairs_total": len(pairs),
        "n_within_pairs": len(within_cosines),
        "n_between_pairs": len(between_cosines),
        "cosine_summary": {
            "within_group_mean": within_cos_mean,
            "between_group_mean": between_cos_mean,
            "gap": float(within_cos_mean - between_cos_mean) if not np.isnan(within_cos_mean) else None,
            "within_negative_rate": float(within_neg_rate),
            "between_negative_rate": float(between_neg_rate),
            "mannwhitney_p": float(mw_p_cosine) if not np.isnan(mw_p_cosine) else None,
        },
        "pair_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_seed_cosines"}
            for k, v in pair_results.items()
        },
        "pair_cosines_per_seed": {
            k: v["per_seed_cosines"] for k, v in pair_results.items()
        },
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "v30_lstm_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
