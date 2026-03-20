"""LLM embedding polarity analysis using sentence-transformers.

Embeds each trace's actions using a pre-trained sentence-transformer model
(all-MiniLM-L6-v2, 384-dim), mean-pools to trace-level representations,
then analyzes polarity reversal via cosine similarity of Cohen's d vectors.

These embeddings capture semantic meaning of tool names and arguments —
a richer representation than heuristic features or LSTM hidden states.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_llm_polarity.py
    PYTHONPATH=../../lib:src uv run python scripts/run_llm_polarity.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

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

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_domain_traces(
    domain_name: str,
    *,
    max_traces: int = 5000,
) -> list:
    """Load traces as Trace objects."""
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
        return generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42)
    elif domain_name == "atbench":
        return load_atbench(max_traces=max_traces)
    elif domain_name == "agentharm":
        return load_agentharm(max_traces=max_traces)
    elif domain_name == "injecagent":
        return load_injecagent(Path("data"), max_traces=max_traces)
    elif domain_name == "rjudge":
        return load_rjudge(Path("data"), max_traces=max_traces)
    elif domain_name == "swe_agent":
        return load_swe_agent(max_traces=max_traces)
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
        return traces
    elif domain_name == "trail":
        trail_dir = Path("data/trail/TRAIL-Benchmark")
        if not trail_dir.exists():
            trail_dir = download_trail(trail_dir.parent)
        return load_trail(trail_dir)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")


def trace_to_texts(trace) -> list[str]:
    """Convert a trace's actions to embeddable text strings."""
    texts = []
    for action in trace.actions:
        parts = [action.tool]
        if hasattr(action, "args") and action.args:
            args_str = str(action.args)[:200]
            parts.append(args_str)
        texts.append(" ".join(parts))
    return texts


def embed_traces(
    traces: list,
    model,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed traces by mean-pooling action-level sentence embeddings.

    Returns (N, embedding_dim) array.
    """
    all_embeddings = []

    for trace in traces:
        action_texts = trace_to_texts(trace)
        if not action_texts:
            all_embeddings.append(np.zeros(EMBEDDING_DIM))
            continue

        # Encode all actions in this trace
        action_embs = model.encode(
            action_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Mean-pool to trace-level
        trace_emb = action_embs.mean(axis=0)
        all_embeddings.append(trace_emb)

    return np.array(all_embeddings)


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
    """Count dimensions with polarity reversal."""
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
        description="LLM embedding polarity analysis",
    )
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--model", type=str, default=MODEL_NAME,
        help="Sentence-transformer model name",
    )
    args = parser.parse_args()

    if args.quick:
        args.max_traces = 500

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    set_seeds(42)

    print("=" * 70)
    print(f"  LLM Embedding Polarity Analysis ({args.model})")
    print("=" * 70)

    print(f"\nLoading sentence-transformer model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {emb_dim}")

    # Load and embed all domains
    domain_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for domain in ALL_DOMAINS:
        print(f"\n  Loading {domain}...")
        try:
            traces = load_domain_traces(domain, max_traces=args.max_traces)
            labels = np.array([1 if t.has_anomaly else 0 for t in traces])
            n_norm = int((labels == 0).sum())
            n_anom = int((labels == 1).sum())
            print(f"    {len(traces)} traces ({n_norm} normal, {n_anom} anomalous)")

            if n_anom < 2 or n_norm < 2:
                print(f"    Skipping: insufficient class balance")
                continue

            print(f"    Embedding {len(traces)} traces...")
            st = time.time()
            embeddings = embed_traces(traces, model)
            print(f"    Done ({time.time() - st:.1f}s), shape: {embeddings.shape}")

            domain_embeddings[domain] = (embeddings, labels)
        except Exception as e:
            print(f"    FAILED: {e}")

    loaded_domains = sorted(domain_embeddings.keys())
    print(f"\nLoaded {len(loaded_domains)} domains: {loaded_domains}")

    # Compute d-vectors
    print(f"\n{'='*70}")
    print("  COHEN'S D ANALYSIS")
    print(f"{'='*70}")

    domain_d: dict[str, np.ndarray] = {}
    for domain in loaded_domains:
        embeddings, labels = domain_embeddings[domain]
        d_vec = compute_domain_d_vector(embeddings, labels)
        domain_d[domain] = d_vec

        abs_d = np.abs(d_vec)
        top5 = np.argsort(abs_d)[::-1][:5]
        top_str = ", ".join(f"d{i}={d_vec[i]:+.3f}" for i in top5)
        n_strong = int((abs_d > 0.2).sum())
        print(f"  {domain:15s}: {n_strong} dims with |d|>0.2, top: {top_str}")

    # Polarity analysis: cosine similarity of d-vectors
    print(f"\n{'='*70}")
    print("  POLARITY ANALYSIS: COSINE SIMILARITY OF d-VECTORS")
    print(f"{'='*70}")

    pairs = []
    for i, da in enumerate(loaded_domains):
        for db in loaded_domains[i + 1:]:
            pairs.append((da, db))

    within_cosines: list[float] = []
    between_cosines: list[float] = []
    pair_results: dict[str, dict] = {}

    for da, db in pairs:
        group_a = DOMAIN_TO_GROUP.get(da, "unknown")
        group_b = DOMAIN_TO_GROUP.get(db, "unknown")
        is_within = group_a == group_b

        d_a = domain_d[da]
        d_b = domain_d[db]

        # Filter NaN
        valid = ~(np.isnan(d_a) | np.isnan(d_b))
        da_v = d_a[valid]
        db_v = d_b[valid]

        # Cosine similarity
        norm_a = np.linalg.norm(da_v)
        norm_b = np.linalg.norm(db_v)
        if norm_a < 1e-10 or norm_b < 1e-10:
            continue
        cos = float(np.dot(da_v, db_v) / (norm_a * norm_b))

        # Per-dim reversal
        n_rev, n_comp = count_reversals(d_a, d_b)
        rev_rate = n_rev / max(n_comp, 1)

        pair_results[f"{da}-{db}"] = {
            "group_a": group_a,
            "group_b": group_b,
            "is_within_group": is_within,
            "cosine": cos,
            "reversal_rate": float(rev_rate),
            "n_reversed": n_rev,
            "n_comparable": n_comp,
        }

        if is_within:
            within_cosines.append(cos)
        else:
            between_cosines.append(cos)

    # Print results
    print(f"\n  {'Pair':<30s} {'Group':<20s} {'Type':<10s} {'Cosine':>8s} {'RevRate':>8s}")
    print("  " + "-" * 80)
    for pair_key, pr in sorted(pair_results.items()):
        ptype = "WITHIN" if pr["is_within_group"] else "between"
        gstr = f"{pr['group_a']}/{pr['group_b']}"
        print(
            f"  {pair_key:<30s} {gstr:<20s} {ptype:<10s} "
            f"{pr['cosine']:>+8.3f} {pr['reversal_rate']:>8.1%}"
        )

    within_cos_mean = float(np.mean(within_cosines)) if within_cosines else float("nan")
    between_cos_mean = float(np.mean(between_cosines)) if between_cosines else float("nan")

    print(f"\n  COSINE SIMILARITY SUMMARY:")
    print(f"    Within-group ({len(within_cosines)} pairs):  mean cos = {within_cos_mean:+.3f}")
    print(f"    Between-group ({len(between_cosines)} pairs): mean cos = {between_cos_mean:+.3f}")

    if not np.isnan(within_cos_mean) and not np.isnan(between_cos_mean):
        gap = within_cos_mean - between_cos_mean
        print(f"    Gap (within - between): {gap:+.3f}")

    # Negative cosine analysis
    within_neg = sum(1 for c in within_cosines if c < 0)
    between_neg = sum(1 for c in between_cosines if c < 0)
    within_neg_rate = within_neg / max(len(within_cosines), 1)
    between_neg_rate = between_neg / max(len(between_cosines), 1)

    print(f"\n  NEGATIVE COSINE (POLARITY REVERSAL):")
    print(f"    Within-group:  {within_neg}/{len(within_cosines)} = {within_neg_rate:.1%}")
    print(f"    Between-group: {between_neg}/{len(between_cosines)} = {between_neg_rate:.1%}")

    # Statistical test
    mw_p = float("nan")
    if len(within_cosines) >= 3 and len(between_cosines) >= 3:
        try:
            from scipy.stats import mannwhitneyu
            _, mw_p = mannwhitneyu(
                within_cosines, between_cosines, alternative="greater",
            )
            print(f"    Mann-Whitney U (within > between): p = {mw_p:.4f}")
        except Exception:
            pass

    # Per-dimension reversal summary
    within_revs = [pr["reversal_rate"] for pr in pair_results.values() if pr["is_within_group"]]
    between_revs = [pr["reversal_rate"] for pr in pair_results.values() if not pr["is_within_group"]]
    within_rev_mean = float(np.mean(within_revs)) if within_revs else float("nan")
    between_rev_mean = float(np.mean(between_revs)) if between_revs else float("nan")

    print(f"\n  PER-DIMENSION REVERSAL RATE (threshold |d|>0.1):")
    print(f"    Within-group:  {within_rev_mean:.1%}")
    print(f"    Between-group: {between_rev_mean:.1%}")

    elapsed = time.time() - t0

    output = {
        "experiment": "v30_llm_polarity",
        "model": args.model,
        "embedding_dim": emb_dim,
        "input_type": "tool_name_plus_args_text",
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
            "mannwhitney_p": float(mw_p) if not np.isnan(mw_p) else None,
        },
        "reversal_summary": {
            "within_group_mean": float(within_rev_mean),
            "between_group_mean": float(between_rev_mean),
        },
        "pair_results": pair_results,
        "elapsed_seconds": elapsed,
    }

    model_slug = args.model.replace("/", "_").replace("-", "_")
    out_path = OUTPUT_DIR / f"v31_{model_slug}_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
