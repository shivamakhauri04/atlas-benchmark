"""Prototype-based anomaly scorer using sentence-transformer embeddings.

Implements the simplest possible "LLM-based" anomaly detector: embed traces
via sentence-transformer, compute a prototype for normal traces from the source
domain, and score target traces by distance to the normal prototype.

This addresses DA reviewer Major #3: "LLM-based scorers are excluded — the most
practically relevant detection approach is not tested."

While this is not a full LLM-as-judge approach (which requires API calls and
is not reproducible), prototype-based scoring captures the core mechanism:
using semantic representations for anomaly detection. If even these semantic
representations fail to transfer, prompt-based LLM scoring (which relies on
similar internal representations) cannot be expected to do better without
domain-specific prompting.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_prototype_scorer.py
    PYTHONPATH=../../lib:src uv run python scripts/run_prototype_scorer.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from research_utils.reproducibility import set_seeds
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

SEMANTIC_GROUPS: dict[str, list[str]] = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "rjudge", "agentharm", "injecagent", "agentdojo"],
    "api_misuse": ["toolbench"],
}

DOMAIN_TO_GROUP: dict[str, str] = {
    d: g for g, ds in SEMANTIC_GROUPS.items() for d in ds
}

ALL_DOMAINS = [
    "synthetic", "swe_agent", "trail", "atbench",
    "rjudge", "agentharm", "injecagent", "toolbench", "agentdojo",
]

MODEL_NAME = "all-MiniLM-L6-v2"


def load_domain_traces(domain_name: str, *, max_traces: int = 5000) -> list:
    """Load traces as Trace objects (reuses existing loaders)."""
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
        return generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42)
    elif domain_name == "atbench":
        return load_atbench(max_traces=max_traces)
    elif domain_name == "agentharm":
        return load_agentharm(max_traces=max_traces)
    elif domain_name == "injecagent":
        return load_injecagent(Path("data"), max_traces=max_traces)
    elif domain_name == "rjudge":
        return load_rjudge(Path("data"), max_traces=max_traces)
    elif domain_name == "agentdojo":
        return load_agentdojo(Path("data"), max_traces=max_traces)
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


def trace_to_text(trace) -> str:
    """Convert trace to a single text string for embedding."""
    parts = []
    for action in trace.actions:
        text = action.tool
        if hasattr(action, "args") and action.args:
            text += " " + str(action.args)[:200]
        parts.append(text)
    return " | ".join(parts)[:2000]


def embed_traces(traces: list, model, batch_size: int = 256) -> np.ndarray:
    """Embed traces by mean-pooling action-level embeddings."""
    all_embeddings = []
    for trace in traces:
        action_texts = []
        for action in trace.actions:
            text = action.tool
            if hasattr(action, "args") and action.args:
                text += " " + str(action.args)[:200]
            action_texts.append(text)
        if not action_texts:
            all_embeddings.append(np.zeros(384))
            continue
        embs = model.encode(
            action_texts, batch_size=batch_size,
            show_progress_bar=False, normalize_embeddings=True,
        )
        all_embeddings.append(embs.mean(axis=0))
    return np.array(all_embeddings)


def prototype_scorer(
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_embeddings: np.ndarray,
) -> np.ndarray:
    """Score target traces by distance to source normal prototype.

    Higher score = more anomalous (farther from normal prototype).
    """
    normal_mask = source_labels == 0
    normal_embs = source_embeddings[normal_mask]
    if len(normal_embs) == 0:
        return np.zeros(len(target_embeddings))

    prototype = normal_embs.mean(axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-10)

    target_norms = target_embeddings / (
        np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-10
    )
    similarities = target_norms @ prototype
    scores = 1.0 - similarities  # distance = anomaly score
    return scores


def mahalanobis_scorer(
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_embeddings: np.ndarray,
) -> np.ndarray:
    """Score by Mahalanobis distance to source normal distribution.

    More robust than prototype: accounts for covariance structure.
    Uses regularized covariance to avoid singularity.
    """
    normal_mask = source_labels == 0
    normal_embs = source_embeddings[normal_mask]
    if len(normal_embs) < 10:
        return prototype_scorer(source_embeddings, source_labels, target_embeddings)

    mean = normal_embs.mean(axis=0)
    cov = np.cov(normal_embs.T)
    cov += np.eye(cov.shape[0]) * 0.01  # regularize

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return prototype_scorer(source_embeddings, source_labels, target_embeddings)

    diff = target_embeddings - mean
    scores = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype anomaly scorer evaluation")
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    if args.quick:
        args.max_traces = 500

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seeds(42)
    t0 = time.time()

    print("=" * 70)
    print("  Prototype-Based Anomaly Scorer (Sentence-Transformer)")
    print("=" * 70)

    print(f"\nLoading model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # Load and embed all domains
    domain_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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

            st = time.time()
            embeddings = embed_traces(traces, model)
            print(f"    Embedded in {time.time() - st:.1f}s, shape: {embeddings.shape}")
            domain_data[domain] = (embeddings, labels)
        except Exception as e:
            print(f"    FAILED: {e}")

    loaded = sorted(domain_data.keys())
    print(f"\nLoaded {len(loaded)} domains: {loaded}")

    # Evaluate cross-domain transfer
    print(f"\n{'='*70}")
    print("  Cross-Domain AUROC (Prototype Scorer)")
    print(f"{'='*70}")

    cross_domain_results: dict[str, dict] = {}

    for src in loaded:
        for tgt in loaded:
            if src == tgt:
                continue
            src_emb, src_lab = domain_data[src]
            tgt_emb, tgt_lab = domain_data[tgt]

            # Prototype scorer
            proto_scores = prototype_scorer(src_emb, src_lab, tgt_emb)
            try:
                proto_auroc = float(roc_auc_score(tgt_lab, proto_scores))
            except ValueError:
                proto_auroc = 0.5

            # Mahalanobis scorer
            mah_scores = mahalanobis_scorer(src_emb, src_lab, tgt_emb)
            try:
                mah_auroc = float(roc_auc_score(tgt_lab, mah_scores))
            except ValueError:
                mah_auroc = 0.5

            pair_key = f"{src}->{tgt}"
            ga = DOMAIN_TO_GROUP.get(src, "unknown")
            gb = DOMAIN_TO_GROUP.get(tgt, "unknown")
            is_within = ga == gb

            cross_domain_results[pair_key] = {
                "source": src,
                "target": tgt,
                "source_group": ga,
                "target_group": gb,
                "within_group": is_within,
                "proto_auroc": proto_auroc,
                "mahal_auroc": mah_auroc,
            }

            marker = "W" if is_within else "B"
            print(f"  [{marker}] {src:15s} -> {tgt:15s}: "
                  f"proto={proto_auroc:.3f}  mahal={mah_auroc:.3f}")

    # Aggregate within vs between
    print(f"\n{'='*70}")
    print("  Aggregate: Within-Group vs Between-Group")
    print(f"{'='*70}")

    within_proto = [v["proto_auroc"] for v in cross_domain_results.values() if v["within_group"]]
    between_proto = [v["proto_auroc"] for v in cross_domain_results.values() if not v["within_group"]]
    within_mah = [v["mahal_auroc"] for v in cross_domain_results.values() if v["within_group"]]
    between_mah = [v["mahal_auroc"] for v in cross_domain_results.values() if not v["within_group"]]

    print(f"\n  Prototype Scorer:")
    print(f"    Within-group AUROC:  {np.mean(within_proto):.3f} +/- {np.std(within_proto):.3f} "
          f"(n={len(within_proto)})")
    print(f"    Between-group AUROC: {np.mean(between_proto):.3f} +/- {np.std(between_proto):.3f} "
          f"(n={len(between_proto)})")

    print(f"\n  Mahalanobis Scorer:")
    print(f"    Within-group AUROC:  {np.mean(within_mah):.3f} +/- {np.std(within_mah):.3f}")
    print(f"    Between-group AUROC: {np.mean(between_mah):.3f} +/- {np.std(between_mah):.3f}")

    # Count anti-predictions (AUROC < 0.5)
    n_anti_proto = sum(1 for v in cross_domain_results.values() if v["proto_auroc"] < 0.5)
    n_anti_mah = sum(1 for v in cross_domain_results.values() if v["mahal_auroc"] < 0.5)
    total_pairs = len(cross_domain_results)
    print(f"\n  Anti-predictions (AUROC < 0.5):")
    print(f"    Prototype: {n_anti_proto}/{total_pairs} ({n_anti_proto/total_pairs*100:.1f}%)")
    print(f"    Mahalanobis: {n_anti_mah}/{total_pairs} ({n_anti_mah/total_pairs*100:.1f}%)")

    # Check if within-group transfers better
    from scipy.stats import mannwhitneyu
    if within_proto and between_proto:
        stat, p = mannwhitneyu(within_proto, between_proto, alternative="greater")
        print(f"\n  Mann-Whitney (within > between, prototype): U={stat:.1f}, p={p:.4f}")
    if within_mah and between_mah:
        stat, p = mannwhitneyu(within_mah, between_mah, alternative="greater")
        print(f"  Mann-Whitney (within > between, mahalanobis): U={stat:.1f}, p={p:.4f}")

    # Cohen's d polarity analysis on prototype scores
    print(f"\n{'='*70}")
    print("  Polarity Analysis: Prototype Score d-vectors")
    print(f"{'='*70}")

    domain_d_proto: dict[str, float] = {}
    for domain in loaded:
        emb, lab = domain_data[domain]
        scores = prototype_scorer(emb, lab, emb)  # self-score
        d = cohens_d(scores[lab == 1], scores[lab == 0])
        domain_d_proto[domain] = d
        print(f"  {domain:15s}: d={d:+.3f}")

    # Check sign reversals
    n_within_rev = 0
    n_within_total = 0
    n_between_rev = 0
    n_between_total = 0

    for da, db in combinations(loaded, 2):
        ga = DOMAIN_TO_GROUP.get(da, "unknown")
        gb = DOMAIN_TO_GROUP.get(db, "unknown")
        d_a = domain_d_proto[da]
        d_b = domain_d_proto[db]
        reversed_flag = abs(d_a) > 0.1 and abs(d_b) > 0.1 and (d_a > 0) != (d_b > 0)
        if ga == gb:
            n_within_total += 1
            if reversed_flag:
                n_within_rev += 1
        else:
            n_between_total += 1
            if reversed_flag:
                n_between_rev += 1

    within_rate = n_within_rev / max(n_within_total, 1)
    between_rate = n_between_rev / max(n_between_total, 1)
    print(f"\n  Prototype score polarity reversal:")
    print(f"    Within-group:  {within_rate:.1%} ({n_within_rev}/{n_within_total})")
    print(f"    Between-group: {between_rate:.1%} ({n_between_rev}/{n_between_total})")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Save results
    output = {
        "experiment": "v31_prototype_scorer",
        "model": args.model,
        "n_domains": len(loaded),
        "domains": loaded,
        "cross_domain_results": cross_domain_results,
        "aggregate": {
            "within_proto_mean": float(np.mean(within_proto)) if within_proto else None,
            "within_proto_std": float(np.std(within_proto)) if within_proto else None,
            "between_proto_mean": float(np.mean(between_proto)) if between_proto else None,
            "between_proto_std": float(np.std(between_proto)) if between_proto else None,
            "within_mah_mean": float(np.mean(within_mah)) if within_mah else None,
            "between_mah_mean": float(np.mean(between_mah)) if between_mah else None,
            "n_anti_proto": n_anti_proto,
            "n_anti_mah": n_anti_mah,
            "total_pairs": total_pairs,
        },
        "polarity_reversal": {
            "within_rate": within_rate,
            "between_rate": between_rate,
            "within_rev": n_within_rev,
            "within_total": n_within_total,
            "between_rev": n_between_rev,
            "between_total": n_between_total,
        },
        "domain_d_proto": domain_d_proto,
        "elapsed_s": elapsed,
    }

    out_path = OUTPUT_DIR / "v31_prototype_scorer.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
