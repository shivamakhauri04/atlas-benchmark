"""Multi-source LSTM polarity analysis.

Addresses reviewer question: "What happens when the LSTM is trained on
each source domain separately (not just synthetic)?"

The original run_lstm_polarity.py trains a shared LSTM on synthetic only.
This script trains a separate LSTM on EACH source domain, then extracts
128-dim features for ALL domains through that encoder. If within-group
0% reversal holds regardless of source domain, the finding is robust.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_multisource_lstm_polarity.py
    PYTHONPATH=../../lib:src uv run python scripts/run_multisource_lstm_polarity.py --quick
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


def load_domain_sequences(domain_name, vocab, *, max_traces=5000):
    """Load a domain's traces, encode to tool-ID sequences."""
    import random
    from data.trace_loader import (
        _inject_anomaly, download_trail, generate_synthetic_traces,
        load_agentharm, load_atbench, load_injecagent,
        load_rjudge, load_swe_agent, load_toolbench, load_trail,
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
                    trace_id=trace.trace_id, actions=new_actions,
                    source=trace.source, task=trace.task,
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


def build_vocab(max_traces=5000):
    """Build shared TraceVocab across all domains."""
    from data.trace_loader import (
        TraceVocab, generate_synthetic_traces, load_agentharm,
        load_atbench, load_injecagent, load_rjudge,
        load_swe_agent, load_toolbench,
    )

    all_traces = []
    all_traces.extend(generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42))

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


def train_lstm_encoder(
    train_seqs: list[list[int]],
    train_labels: list[int],
    vocab_size: int,
    *,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
) -> torch.nn.Module:
    """Train LSTM encoder on a given source domain."""
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
    trainer.train(train_seqs, train_labels, train_seqs, train_labels)
    return trainer.encoder


def extract_features(encoder, seqs, batch_size=64) -> np.ndarray:
    """Extract 128-dim mean-pooled hidden states."""
    device = next(encoder.parameters()).device
    encoder.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i + batch_size]
            max_len = max(len(s) for s in batch_seqs)
            ids = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
            mask = torch.zeros(len(batch_seqs), max_len, dtype=torch.bool)
            for j, s in enumerate(batch_seqs):
                ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
                mask[j, :len(s)] = True
            ids = ids.to(device)
            mask = mask.to(device)
            feats = encoder.trace_features(ids, mask)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_domain_d_vector(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    d_hidden = features.shape[1]
    d_vec = np.zeros(d_hidden)
    normal = labels == 0
    anomalous = labels == 1
    if normal.sum() < 2 or anomalous.sum() < 2:
        return d_vec * float("nan")
    for dim in range(d_hidden):
        d_vec[dim] = cohens_d(features[anomalous, dim], features[normal, dim])
    return d_vec


def analyze_polarity(
    domain_d: dict[str, np.ndarray],
    loaded_domains: list[str],
) -> dict:
    """Compute cosine similarity of d-vectors for all pairs."""
    pairs = [(da, db) for i, da in enumerate(loaded_domains) for db in loaded_domains[i+1:]]
    within_cosines = []
    between_cosines = []
    pair_results = {}

    for da, db in pairs:
        ga = DOMAIN_TO_GROUP.get(da, "?")
        gb = DOMAIN_TO_GROUP.get(db, "?")
        is_within = ga == gb

        d_a, d_b = domain_d[da], domain_d[db]
        valid = ~(np.isnan(d_a) | np.isnan(d_b))
        da_v, db_v = d_a[valid], d_b[valid]
        na, nb = np.linalg.norm(da_v), np.linalg.norm(db_v)
        if na < 1e-10 or nb < 1e-10:
            continue
        cos = float(np.dot(da_v, db_v) / (na * nb))

        pair_results[f"{da}-{db}"] = {
            "is_within_group": is_within, "cosine": cos,
        }
        (within_cosines if is_within else between_cosines).append(cos)

    w_mean = float(np.mean(within_cosines)) if within_cosines else float("nan")
    b_mean = float(np.mean(between_cosines)) if between_cosines else float("nan")
    w_neg = sum(1 for c in within_cosines if c < 0)
    b_neg = sum(1 for c in between_cosines if c < 0)

    return {
        "within_cos_mean": w_mean,
        "between_cos_mean": b_mean,
        "within_neg": w_neg,
        "within_total": len(within_cosines),
        "between_neg": b_neg,
        "between_total": len(between_cosines),
        "within_neg_rate": w_neg / max(len(within_cosines), 1),
        "between_neg_rate": b_neg / max(len(between_cosines), 1),
        "pair_results": pair_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-source LSTM polarity analysis")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 10
        args.max_traces = 500

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  MULTI-SOURCE LSTM POLARITY ANALYSIS")
    print(f"  Seeds: {args.seeds}, Epochs: {args.epochs}")
    print("=" * 70)

    print("\nBuilding shared vocabulary...")
    vocab = build_vocab(max_traces=args.max_traces)
    print(f"  Vocab size: {vocab.vocab_size}")

    # Load all domain sequences
    domain_seqs: dict[str, tuple[list[list[int]], list[int]]] = {}
    for domain in ALL_DOMAINS:
        try:
            seqs, labels = load_domain_sequences(domain, vocab, max_traces=args.max_traces)
            n_anom = sum(labels)
            n_norm = len(labels) - n_anom
            print(f"  {domain:15s}: {len(seqs)} traces ({n_norm} normal, {n_anom} anomalous)")
            if n_anom < 2 or n_norm < 2:
                print(f"    Skipping: insufficient class balance")
                continue
            domain_seqs[domain] = (seqs, labels)
        except Exception as e:
            print(f"  {domain:15s}: FAILED ({e})")

    loaded_domains = sorted(domain_seqs.keys())
    print(f"\nLoaded {len(loaded_domains)} domains: {loaded_domains}")

    # For each source domain, for each seed: train encoder, extract features, compute polarity
    source_domains = [d for d in loaded_domains if sum(domain_seqs[d][1]) >= 10]
    print(f"\nSource domains (>=10 anomalies): {source_domains}")

    per_source_results: dict[str, dict] = {}

    for src_domain in source_domains:
        print(f"\n{'='*60}")
        print(f"  SOURCE DOMAIN: {src_domain}")
        print(f"{'='*60}")

        seed_polarity_results = []

        for seed in args.seeds:
            print(f"\n  Seed {seed}: training on {src_domain}...")
            src_seqs, src_labels = domain_seqs[src_domain]

            st = time.time()
            encoder = train_lstm_encoder(
                src_seqs, src_labels, vocab.vocab_size,
                epochs=args.epochs, seed=seed,
            )
            print(f"    Trained in {time.time() - st:.1f}s")

            # Extract features for all domains
            seed_d: dict[str, np.ndarray] = {}
            for domain in loaded_domains:
                seqs, labels = domain_seqs[domain]
                features = extract_features(encoder, seqs)
                seed_d[domain] = compute_domain_d_vector(features, np.array(labels))

            polarity = analyze_polarity(seed_d, loaded_domains)
            seed_polarity_results.append(polarity)

            print(f"    Within cos: {polarity['within_cos_mean']:+.3f}, "
                  f"Between cos: {polarity['between_cos_mean']:+.3f}, "
                  f"Within neg: {polarity['within_neg']}/{polarity['within_total']}, "
                  f"Between neg: {polarity['between_neg']}/{polarity['between_total']}")

            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate across seeds
        w_cos = [r["within_cos_mean"] for r in seed_polarity_results if not np.isnan(r["within_cos_mean"])]
        b_cos = [r["between_cos_mean"] for r in seed_polarity_results if not np.isnan(r["between_cos_mean"])]
        w_neg_total = sum(r["within_neg"] for r in seed_polarity_results)
        w_total = sum(r["within_total"] for r in seed_polarity_results)
        b_neg_total = sum(r["between_neg"] for r in seed_polarity_results)
        b_total = sum(r["between_total"] for r in seed_polarity_results)

        per_source_results[src_domain] = {
            "n_seeds": len(args.seeds),
            "within_cos_mean": float(np.mean(w_cos)) if w_cos else float("nan"),
            "within_cos_std": float(np.std(w_cos)) if w_cos else float("nan"),
            "between_cos_mean": float(np.mean(b_cos)) if b_cos else float("nan"),
            "between_cos_std": float(np.std(b_cos)) if b_cos else float("nan"),
            "within_neg_total": w_neg_total,
            "within_pairs_total": w_total,
            "between_neg_total": b_neg_total,
            "between_pairs_total": b_total,
            "within_neg_rate": w_neg_total / max(w_total, 1),
            "between_neg_rate": b_neg_total / max(b_total, 1),
        }

    # Final summary table
    print(f"\n{'='*70}")
    print("  SUMMARY: MULTI-SOURCE LSTM POLARITY")
    print(f"{'='*70}")

    print(f"\n  {'Source':<15s} {'W cos':>8s} {'B cos':>8s} {'W neg%':>8s} {'B neg%':>8s} {'Gap?':>6s}")
    print("  " + "-" * 55)
    for src, res in per_source_results.items():
        gap = "YES" if res["within_neg_rate"] < res["between_neg_rate"] - 0.05 else "no"
        print(f"  {src:<15s} {res['within_cos_mean']:>+8.3f} {res['between_cos_mean']:>+8.3f} "
              f"{res['within_neg_rate']:>8.1%} {res['between_neg_rate']:>8.1%} {gap:>6s}")

    elapsed = time.time() - t0

    output = {
        "experiment": "v31_multisource_lstm_polarity",
        "d_hidden": 128,
        "input_type": "raw_tool_sequences",
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "source_domains": source_domains,
        "loaded_domains": loaded_domains,
        "per_source_results": per_source_results,
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "v31_multisource_lstm_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
