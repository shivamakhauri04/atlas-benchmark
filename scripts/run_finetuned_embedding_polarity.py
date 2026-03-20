"""Fine-tuned embedding polarity analysis.

Addresses reviewer concern: "only one frozen sentence-transformer tested."
Fine-tunes a sentence-transformer on each source domain's anomaly labels
using contrastive learning (SetFit-style), then re-runs the cosine-of-d-vectors
polarity analysis across all 9 domains.

If fine-tuning recovers structured polarity (within < between reversal),
it strengthens the "richer representations sidestep the problem" claim.
If not, it shows polarity reversal is fundamental, not an artifact of
frozen embeddings.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_finetuned_embedding_polarity.py
    PYTHONPATH=../../lib:src uv run python scripts/run_finetuned_embedding_polarity.py --quick
"""

from __future__ import annotations

import argparse
import copy
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

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_domain_traces(domain_name: str, *, max_traces: int = 5000) -> list:
    """Load traces as Trace objects (reused from run_llm_polarity)."""
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


def embed_traces(traces: list, model, batch_size: int = 256) -> np.ndarray:
    """Embed traces by mean-pooling action-level sentence embeddings."""
    all_embeddings = []
    emb_dim = model.get_sentence_embedding_dimension()
    for trace in traces:
        action_texts = trace_to_texts(trace)
        if not action_texts:
            all_embeddings.append(np.zeros(emb_dim))
            continue
        action_embs = model.encode(
            action_texts, batch_size=batch_size,
            show_progress_bar=False, normalize_embeddings=True,
        )
        trace_emb = action_embs.mean(axis=0)
        all_embeddings.append(trace_emb)
    return np.array(all_embeddings)


def compute_domain_d_vector(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cohen's d per dimension."""
    d_hidden = features.shape[1]
    d_vec = np.zeros(d_hidden)
    normal = labels == 0
    anomalous = labels == 1
    if normal.sum() < 2 or anomalous.sum() < 2:
        return d_vec * float("nan")
    for dim in range(d_hidden):
        d_vec[dim] = cohens_d(features[anomalous, dim], features[normal, dim])
    return d_vec


def finetune_on_domain(
    base_model,
    traces: list,
    labels: np.ndarray,
    *,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
) -> object:
    """Fine-tune sentence-transformer on domain labels via contrastive loss.

    Uses cosine similarity loss: same-class pairs should be close,
    different-class pairs should be far. This is a simplified SetFit approach.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    set_seeds(seed)

    # Create training pairs: sample pairs within and across classes
    rng = np.random.RandomState(seed)
    normal_idx = np.where(labels == 0)[0]
    anom_idx = np.where(labels == 1)[0]

    # Create balanced pairs
    n_pairs = min(2000, len(traces) * 2)
    examples = []
    for _ in range(n_pairs):
        if rng.random() < 0.5:
            # Same class pair (label=1.0 = similar)
            if rng.random() < 0.5 and len(normal_idx) >= 2:
                i, j = rng.choice(normal_idx, 2, replace=False)
            elif len(anom_idx) >= 2:
                i, j = rng.choice(anom_idx, 2, replace=False)
            else:
                continue
            sim_label = 1.0
        else:
            # Different class pair (label=0.0 = dissimilar)
            if len(normal_idx) < 1 or len(anom_idx) < 1:
                continue
            i = rng.choice(normal_idx)
            j = rng.choice(anom_idx)
            sim_label = 0.0

        text_i = " [SEP] ".join(trace_to_texts(traces[i])[:10])
        text_j = " [SEP] ".join(trace_to_texts(traces[j])[:10])
        examples.append(InputExample(texts=[text_i, text_j], label=sim_label))

    if len(examples) < 10:
        print(f"    Too few training pairs ({len(examples)}), skipping fine-tuning")
        return base_model

    # Deep copy the model for fine-tuning
    ft_model = SentenceTransformer(base_model[0].auto_model.config._name_or_path)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(ft_model)

    ft_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(train_dataloader) * 0.1),
        show_progress_bar=False,
    )

    return ft_model


def count_reversals(d_a: np.ndarray, d_b: np.ndarray, threshold: float = 0.1):
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


def run_polarity_analysis(
    domain_d: dict[str, np.ndarray],
    loaded_domains: list[str],
    label: str,
) -> dict:
    """Run cosine-of-d-vectors polarity analysis on a set of d-vectors."""
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
        valid = ~(np.isnan(d_a) | np.isnan(d_b))
        da_v = d_a[valid]
        db_v = d_b[valid]

        norm_a = np.linalg.norm(da_v)
        norm_b = np.linalg.norm(db_v)
        if norm_a < 1e-10 or norm_b < 1e-10:
            continue
        cos = float(np.dot(da_v, db_v) / (norm_a * norm_b))

        n_rev, n_comp = count_reversals(d_a, d_b)
        rev_rate = n_rev / max(n_comp, 1)

        pair_results[f"{da}-{db}"] = {
            "group_a": group_a, "group_b": group_b,
            "is_within_group": is_within,
            "cosine": cos, "reversal_rate": float(rev_rate),
        }

        if is_within:
            within_cosines.append(cos)
        else:
            between_cosines.append(cos)

    within_cos_mean = float(np.mean(within_cosines)) if within_cosines else float("nan")
    between_cos_mean = float(np.mean(between_cosines)) if between_cosines else float("nan")
    within_neg = sum(1 for c in within_cosines if c < 0)
    between_neg = sum(1 for c in between_cosines if c < 0)

    mw_p = float("nan")
    if len(within_cosines) >= 3 and len(between_cosines) >= 3:
        try:
            from scipy.stats import mannwhitneyu
            _, mw_p = mannwhitneyu(within_cosines, between_cosines, alternative="greater")
        except Exception:
            pass

    print(f"\n  [{label}] COSINE SIMILARITY SUMMARY:")
    print(f"    Within-group ({len(within_cosines)} pairs):  mean cos = {within_cos_mean:+.3f}")
    print(f"    Between-group ({len(between_cosines)} pairs): mean cos = {between_cos_mean:+.3f}")
    print(f"    Negative cosine: within {within_neg}/{len(within_cosines)}, "
          f"between {between_neg}/{len(between_cosines)}")

    return {
        "label": label,
        "n_within": len(within_cosines),
        "n_between": len(between_cosines),
        "within_cos_mean": within_cos_mean,
        "between_cos_mean": between_cos_mean,
        "within_neg_rate": within_neg / max(len(within_cosines), 1),
        "between_neg_rate": between_neg / max(len(between_cosines), 1),
        "mannwhitney_p": float(mw_p) if not np.isnan(mw_p) else None,
        "pair_results": pair_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuned embedding polarity analysis")
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ft-epochs", type=int, default=3)
    parser.add_argument(
        "--source-domains", nargs="+", default=None,
        help="Domains to fine-tune on (default: all with enough data)",
    )
    args = parser.parse_args()

    if args.quick:
        args.max_traces = 500
        args.ft_epochs = 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    set_seeds(42)

    print("=" * 70)
    print("  FINE-TUNED EMBEDDING POLARITY ANALYSIS")
    print(f"  Fine-tune epochs: {args.ft_epochs}")
    print("=" * 70)

    # Load base model
    print(f"\nLoading base model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    base_model = SentenceTransformer(MODEL_NAME)
    emb_dim = base_model.get_sentence_embedding_dimension()
    print(f"  Embedding dimension: {emb_dim}")

    # Load all domain traces
    domain_traces: dict[str, tuple[list, np.ndarray]] = {}
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
            domain_traces[domain] = (traces, labels)
        except Exception as e:
            print(f"    FAILED: {e}")

    loaded_domains = sorted(domain_traces.keys())
    print(f"\nLoaded {len(loaded_domains)} domains: {loaded_domains}")

    # 1) Frozen baseline (for comparison)
    print(f"\n{'='*70}")
    print("  PHASE 1: FROZEN EMBEDDINGS (baseline)")
    print(f"{'='*70}")

    frozen_d: dict[str, np.ndarray] = {}
    for domain in loaded_domains:
        traces, labels = domain_traces[domain]
        embeddings = embed_traces(traces, base_model)
        frozen_d[domain] = compute_domain_d_vector(embeddings, labels)

    frozen_results = run_polarity_analysis(frozen_d, loaded_domains, "frozen")

    # 2) Fine-tuned per source domain
    print(f"\n{'='*70}")
    print("  PHASE 2: FINE-TUNED EMBEDDINGS (per source domain)")
    print(f"{'='*70}")

    source_domains = args.source_domains or [
        d for d in loaded_domains
        if int((domain_traces[d][1] == 1).sum()) >= 10
    ]
    print(f"  Source domains for fine-tuning: {source_domains}")

    per_source_results: dict[str, dict] = {}

    for src_domain in source_domains:
        print(f"\n  --- Fine-tuning on {src_domain} ---")
        src_traces, src_labels = domain_traces[src_domain]

        st = time.time()
        ft_model = finetune_on_domain(
            base_model, src_traces, src_labels,
            epochs=args.ft_epochs, seed=42,
        )
        print(f"    Fine-tuned in {time.time() - st:.1f}s")

        # Embed ALL domains with the fine-tuned model
        ft_d: dict[str, np.ndarray] = {}
        for domain in loaded_domains:
            traces, labels = domain_traces[domain]
            embeddings = embed_traces(traces, ft_model)
            ft_d[domain] = compute_domain_d_vector(embeddings, labels)

        result = run_polarity_analysis(ft_d, loaded_domains, f"ft_{src_domain}")
        per_source_results[src_domain] = result

        # Cleanup
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3) Summary comparison
    print(f"\n{'='*70}")
    print("  SUMMARY: FROZEN vs FINE-TUNED")
    print(f"{'='*70}")

    print(f"\n  {'Source':<15s} {'Within cos':>12s} {'Between cos':>12s} "
          f"{'Within neg%':>12s} {'Between neg%':>12s}")
    print("  " + "-" * 65)
    print(f"  {'frozen':<15s} {frozen_results['within_cos_mean']:>+12.3f} "
          f"{frozen_results['between_cos_mean']:>+12.3f} "
          f"{frozen_results['within_neg_rate']:>12.1%} "
          f"{frozen_results['between_neg_rate']:>12.1%}")

    for src, res in per_source_results.items():
        print(f"  ft_{src:<11s} {res['within_cos_mean']:>+12.3f} "
              f"{res['between_cos_mean']:>+12.3f} "
              f"{res['within_neg_rate']:>12.1%} "
              f"{res['between_neg_rate']:>12.1%}")

    elapsed = time.time() - t0

    output = {
        "experiment": "v31_finetuned_embedding_polarity",
        "model": MODEL_NAME,
        "embedding_dim": emb_dim,
        "ft_epochs": args.ft_epochs,
        "source_domains": source_domains,
        "loaded_domains": loaded_domains,
        "frozen_results": {
            k: v for k, v in frozen_results.items() if k != "pair_results"
        },
        "frozen_pair_results": frozen_results["pair_results"],
        "per_source_results": {
            src: {k: v for k, v in res.items() if k != "pair_results"}
            for src, res in per_source_results.items()
        },
        "per_source_pair_results": {
            src: res["pair_results"]
            for src, res in per_source_results.items()
        },
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "v31_finetuned_embedding_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
