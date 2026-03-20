"""CodeBERT and CodeT5+ encoder polarity analysis.

Tests whether code-specialized encoders (125M-220M params, 768-dim) break
the polarity reversal pattern observed with the general-purpose MiniLM (22M,
384-dim). Runs both polarity analysis (cosine of Cohen's d vectors) and
prototype-based anomaly scoring.

Addresses DA reviewer Major #1: "Feature space too narrow — only a 22M
parameter sentence-transformer tested. Larger code-specialized encoders
could produce different results."

Designed to run on Kaggle T4 GPU (~2-4 hours).

Usage (local):
    PYTHONPATH=../../lib:src uv run python scripts/run_codebert_polarity.py

Usage (Kaggle):
    # In a Kaggle notebook with GPU enabled:
    !git clone <repo> && cd research-workspace/experiments/trustlens-auditable-agent-trust
    !pip install transformers torch sentence-transformers orjsonl orjson scipy scikit-learn
    !PYTHONPATH=../../lib:src python scripts/run_codebert_polarity.py --kaggle
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


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

MODELS = {
    "codebert": {
        "name": "microsoft/codebert-base",
        "dim": 768,
        "params": "125M",
        "type": "encoder",
    },
    "codet5p": {
        "name": "Salesforce/codet5p-220m",
        "dim": 768,
        "params": "220M",
        "type": "encoder-decoder",
    },
}


def load_domain_traces(domain_name: str, *, max_traces: int = 5000) -> list:
    """Load traces as Trace objects (reuses existing loaders)."""
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
    elif domain_name == "agentdojo":
        try:
            from data.trace_loader import load_agentdojo
            return load_agentdojo(Path("data"), max_traces=max_traces)
        except (ImportError, Exception):
            return []
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
    """Convert trace actions to embeddable text strings."""
    texts = []
    for action in trace.actions:
        parts = [action.tool]
        if hasattr(action, "args") and action.args:
            parts.append(str(action.args)[:200])
        texts.append(" ".join(parts))
    return texts


class CodeBERTEmbedder:
    """Embed texts using CodeBERT (or any HuggingFace encoder model).

    Uses CLS token pooling with L2 normalization.
    """

    def __init__(self, model_name: str, device: str = "auto"):
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        print(f"  Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.dim = self.model.config.hidden_size
        print(f"  Loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params, "
              f"{self.dim}-dim")

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """Encode texts to normalized embeddings via CLS pooling."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            # CLS token embedding
            cls_emb = outputs.last_hidden_state[:, 0, :]
            # L2 normalize
            cls_emb = cls_emb / (cls_emb.norm(dim=1, keepdim=True) + 1e-10)
            all_embs.append(cls_emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


class CodeT5Embedder:
    """Embed texts using CodeT5+ encoder."""

    def __init__(self, model_name: str, device: str = "auto"):
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        print(f"  Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self.dim = self.model.config.hidden_size
        print(f"  Loaded: {sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params, "
              f"{self.dim}-dim")

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> np.ndarray:
        """Encode texts via encoder mean pooling."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model.encoder(**inputs) if hasattr(self.model, "encoder") else self.model(**inputs)
            # Mean pool over non-padding tokens
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = outputs.last_hidden_state
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            pooled = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-10)
            all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0)


def embed_traces(traces: list, embedder, batch_size: int = 32) -> np.ndarray:
    """Embed traces by mean-pooling action-level embeddings."""
    all_embeddings = []
    for trace in traces:
        action_texts = trace_to_texts(trace)
        if not action_texts:
            all_embeddings.append(np.zeros(embedder.dim))
            continue
        embs = embedder.encode(action_texts, batch_size=batch_size)
        trace_emb = embs.mean(axis=0)
        all_embeddings.append(trace_emb)
    return np.array(all_embeddings)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


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


def prototype_scorer(
    source_emb: np.ndarray,
    source_lab: np.ndarray,
    target_emb: np.ndarray,
) -> np.ndarray:
    """Score by cosine distance to source normal prototype."""
    normal_embs = source_emb[source_lab == 0]
    if len(normal_embs) == 0:
        return np.zeros(len(target_emb))
    prototype = normal_embs.mean(axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-10)
    target_norms = target_emb / (np.linalg.norm(target_emb, axis=1, keepdims=True) + 1e-10)
    return 1.0 - (target_norms @ prototype)


def run_analysis_for_model(
    model_key: str,
    model_info: dict,
    domain_traces: dict[str, tuple[list, np.ndarray]],
    device: str,
) -> dict:
    """Run full polarity + prototype analysis for one encoder model."""
    print(f"\n{'='*70}")
    print(f"  Model: {model_key} ({model_info['name']}, {model_info['params']})")
    print(f"{'='*70}")

    # Load encoder
    if model_info["type"] == "encoder":
        embedder = CodeBERTEmbedder(model_info["name"], device=device)
    else:
        embedder = CodeT5Embedder(model_info["name"], device=device)

    # Embed all domains
    domain_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for domain in sorted(domain_traces.keys()):
        traces, labels = domain_traces[domain]
        print(f"\n  Embedding {domain} ({len(traces)} traces)...")
        st = time.time()
        embeddings = embed_traces(traces, embedder, batch_size=16)
        elapsed = time.time() - st
        print(f"    Done in {elapsed:.1f}s, shape: {embeddings.shape}")
        domain_data[domain] = (embeddings, labels)

    loaded = sorted(domain_data.keys())

    # --- Polarity Analysis ---
    print(f"\n  {'='*60}")
    print(f"  POLARITY ANALYSIS (cosine of Cohen's d vectors)")
    print(f"  {'='*60}")

    domain_d: dict[str, np.ndarray] = {}
    for domain in loaded:
        emb, lab = domain_data[domain]
        d_vec = compute_domain_d_vector(emb, lab)
        domain_d[domain] = d_vec
        abs_d = np.abs(d_vec)
        n_strong = int((abs_d > 0.2).sum())
        print(f"    {domain:15s}: {n_strong} dims with |d|>0.2")

    within_cosines: list[float] = []
    between_cosines: list[float] = []
    within_neg = 0
    within_total = 0
    between_neg = 0
    between_total = 0

    for da, db in combinations(loaded, 2):
        ga = DOMAIN_TO_GROUP.get(da, "unk")
        gb = DOMAIN_TO_GROUP.get(db, "unk")
        is_within = ga == gb

        d_a, d_b = domain_d[da], domain_d[db]
        valid = ~(np.isnan(d_a) | np.isnan(d_b))
        da_v, db_v = d_a[valid], d_b[valid]

        norm_a = np.linalg.norm(da_v)
        norm_b = np.linalg.norm(db_v)
        if norm_a < 1e-10 or norm_b < 1e-10:
            continue
        cos = float(np.dot(da_v, db_v) / (norm_a * norm_b))

        if is_within:
            within_cosines.append(cos)
            within_total += 1
            if cos < 0:
                within_neg += 1
        else:
            between_cosines.append(cos)
            between_total += 1
            if cos < 0:
                between_neg += 1

    within_neg_pct = within_neg / max(within_total, 1) * 100
    between_neg_pct = between_neg / max(between_total, 1) * 100
    within_cos_mean = float(np.mean(within_cosines)) if within_cosines else 0.0
    between_cos_mean = float(np.mean(between_cosines)) if between_cosines else 0.0

    print(f"\n  Within-group:  cos={within_cos_mean:+.3f}, neg={within_neg_pct:.1f}% "
          f"({within_neg}/{within_total})")
    print(f"  Between-group: cos={between_cos_mean:+.3f}, neg={between_neg_pct:.1f}% "
          f"({between_neg}/{between_total})")

    mw_p = None
    if within_cosines and between_cosines:
        _, mw_p = mannwhitneyu(within_cosines, between_cosines, alternative="greater")
        print(f"  Mann-Whitney (within > between): p={mw_p:.4f}")

    # --- Prototype Scorer ---
    print(f"\n  {'='*60}")
    print(f"  PROTOTYPE SCORER (cross-domain AUROC)")
    print(f"  {'='*60}")

    cross_results: dict[str, dict] = {}
    for src in loaded:
        for tgt in loaded:
            if src == tgt:
                continue
            src_emb, src_lab = domain_data[src]
            tgt_emb, tgt_lab = domain_data[tgt]
            scores = prototype_scorer(src_emb, src_lab, tgt_emb)
            try:
                auroc = float(roc_auc_score(tgt_lab, scores))
            except ValueError:
                auroc = 0.5
            ga = DOMAIN_TO_GROUP.get(src, "unk")
            gb = DOMAIN_TO_GROUP.get(tgt, "unk")
            cross_results[f"{src}->{tgt}"] = {
                "auroc": auroc,
                "within": ga == gb,
            }

    within_aurocs = [v["auroc"] for v in cross_results.values() if v["within"]]
    between_aurocs = [v["auroc"] for v in cross_results.values() if not v["within"]]
    n_anti = sum(1 for v in cross_results.values() if v["auroc"] < 0.5)
    total = len(cross_results)

    print(f"\n  Within AUROC:  {np.mean(within_aurocs):.3f} +/- {np.std(within_aurocs):.3f}")
    print(f"  Between AUROC: {np.mean(between_aurocs):.3f} +/- {np.std(between_aurocs):.3f}")
    print(f"  Anti-predict:  {n_anti}/{total} ({n_anti/total*100:.1f}%)")

    # Prototype score d-values (single score per domain, check sign reversal)
    domain_d_proto: dict[str, float] = {}
    for domain in loaded:
        emb, lab = domain_data[domain]
        scores = prototype_scorer(emb, lab, emb)
        d = cohens_d(scores[lab == 1], scores[lab == 0])
        domain_d_proto[domain] = d

    proto_within_rev = 0
    proto_within_total = 0
    proto_between_rev = 0
    proto_between_total = 0
    for da, db in combinations(loaded, 2):
        ga = DOMAIN_TO_GROUP.get(da, "unk")
        gb = DOMAIN_TO_GROUP.get(db, "unk")
        d_a = domain_d_proto[da]
        d_b = domain_d_proto[db]
        rev = abs(d_a) > 0.1 and abs(d_b) > 0.1 and (d_a > 0) != (d_b > 0)
        if ga == gb:
            proto_within_total += 1
            if rev:
                proto_within_rev += 1
        else:
            proto_between_total += 1
            if rev:
                proto_between_rev += 1

    print(f"\n  Proto score reversal: W={proto_within_rev}/{proto_within_total}, "
          f"B={proto_between_rev}/{proto_between_total}")

    return {
        "model": model_key,
        "model_name": model_info["name"],
        "model_params": model_info["params"],
        "embedding_dim": embedder.dim,
        "n_domains": len(loaded),
        "domains": loaded,
        "polarity": {
            "within_cos_mean": within_cos_mean,
            "between_cos_mean": between_cos_mean,
            "within_neg_pct": within_neg_pct,
            "between_neg_pct": between_neg_pct,
            "within_neg": within_neg,
            "within_total": within_total,
            "between_neg": between_neg,
            "between_total": between_total,
            "mannwhitney_p": float(mw_p) if mw_p is not None else None,
        },
        "prototype_scorer": {
            "within_auroc_mean": float(np.mean(within_aurocs)),
            "between_auroc_mean": float(np.mean(between_aurocs)),
            "anti_predict_pct": n_anti / total * 100,
            "n_anti": n_anti,
            "total_pairs": total,
        },
        "prototype_reversal": {
            "within_rev": proto_within_rev,
            "within_total": proto_within_total,
            "between_rev": proto_between_rev,
            "between_total": proto_between_total,
        },
        "domain_d_proto": domain_d_proto,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeBERT/CodeT5+ polarity analysis")
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--kaggle", action="store_true", help="Running on Kaggle (use CUDA)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--models", nargs="+", default=["codebert"],
        choices=list(MODELS.keys()),
        help="Which models to test",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    device = args.device
    if args.kaggle:
        device = "cuda"

    print("=" * 70)
    print("  Code-Specialized Encoder Polarity Analysis")
    print(f"  Models: {args.models}")
    print(f"  Device: {device}")
    print("=" * 70)

    # Load all domain traces (once, reuse across models)
    print("\nLoading domain traces...")
    domain_traces: dict[str, tuple[list, np.ndarray]] = {}
    for domain in ALL_DOMAINS:
        print(f"  Loading {domain}...")
        try:
            traces = load_domain_traces(domain, max_traces=args.max_traces)
            labels = np.array([1 if t.has_anomaly else 0 for t in traces])
            n_norm = int((labels == 0).sum())
            n_anom = int((labels == 1).sum())
            if n_anom < 2 or n_norm < 2:
                print(f"    Skipping: insufficient balance ({n_norm} normal, {n_anom} anomalous)")
                continue
            print(f"    {len(traces)} traces ({n_norm} normal, {n_anom} anomalous)")
            domain_traces[domain] = (traces, labels)
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nLoaded {len(domain_traces)} domains: {sorted(domain_traces.keys())}")

    # Run analysis for each model
    all_results = {}
    for model_key in args.models:
        model_info = MODELS[model_key]
        result = run_analysis_for_model(model_key, model_info, domain_traces, device)
        all_results[model_key] = result

    # Also run MiniLM for direct comparison
    print(f"\n{'='*70}")
    print(f"  Reference: all-MiniLM-L6-v2 (22M, 384-dim)")
    print(f"{'='*70}")

    from sentence_transformers import SentenceTransformer

    class STWrapper:
        def __init__(self, model_name: str, device: str):
            dev = device if device != "auto" else None
            self.model = SentenceTransformer(model_name, device=dev)
            self.dim = self.model.get_sentence_embedding_dimension()

        def encode(self, texts, batch_size=32, **kwargs):
            return self.model.encode(
                texts, batch_size=batch_size,
                show_progress_bar=False, normalize_embeddings=True,
            )

    minilm = STWrapper("all-MiniLM-L6-v2", device)
    minilm_info = {"name": "all-MiniLM-L6-v2", "params": "22M", "type": "sentence-transformer"}

    # Embed with MiniLM
    minilm_domain_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for domain in sorted(domain_traces.keys()):
        traces, labels = domain_traces[domain]
        print(f"  Embedding {domain} with MiniLM...")
        embs = embed_traces(traces, minilm, batch_size=64)
        minilm_domain_data[domain] = (embs, labels)

    # Quick polarity check for MiniLM
    loaded = sorted(minilm_domain_data.keys())
    minilm_d = {}
    for domain in loaded:
        emb, lab = minilm_domain_data[domain]
        minilm_d[domain] = compute_domain_d_vector(emb, lab)

    minilm_w_cos, minilm_b_cos = [], []
    for da, db in combinations(loaded, 2):
        ga = DOMAIN_TO_GROUP.get(da, "unk")
        gb = DOMAIN_TO_GROUP.get(db, "unk")
        d_a, d_b = minilm_d[da], minilm_d[db]
        valid = ~(np.isnan(d_a) | np.isnan(d_b))
        da_v, db_v = d_a[valid], d_b[valid]
        norm_a, norm_b = np.linalg.norm(da_v), np.linalg.norm(db_v)
        if norm_a < 1e-10 or norm_b < 1e-10:
            continue
        cos = float(np.dot(da_v, db_v) / (norm_a * norm_b))
        if ga == gb:
            minilm_w_cos.append(cos)
        else:
            minilm_b_cos.append(cos)

    print(f"\n  MiniLM within cos:  {np.mean(minilm_w_cos):+.3f}, "
          f"neg={sum(1 for c in minilm_w_cos if c < 0)}/{len(minilm_w_cos)}")
    print(f"  MiniLM between cos: {np.mean(minilm_b_cos):+.3f}, "
          f"neg={sum(1 for c in minilm_b_cos if c < 0)}/{len(minilm_b_cos)}")

    all_results["minilm_reference"] = {
        "model": "minilm",
        "model_name": "all-MiniLM-L6-v2",
        "model_params": "22M",
        "embedding_dim": 384,
        "polarity": {
            "within_cos_mean": float(np.mean(minilm_w_cos)),
            "between_cos_mean": float(np.mean(minilm_b_cos)),
            "within_neg_pct": sum(1 for c in minilm_w_cos if c < 0) / max(len(minilm_w_cos), 1) * 100,
            "between_neg_pct": sum(1 for c in minilm_b_cos if c < 0) / max(len(minilm_b_cos), 1) * 100,
        },
    }

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Model':25s} {'Params':>8s} {'Dim':>5s} {'W cos':>8s} {'B cos':>8s} "
          f"{'W neg%':>8s} {'B neg%':>8s} {'Anti%':>8s}")
    print(f"  {'-'*90}")

    for key, res in all_results.items():
        pol = res["polarity"]
        proto = res.get("prototype_scorer", {})
        anti = f"{proto.get('anti_predict_pct', 0):.1f}%" if proto else "N/A"
        print(f"  {res['model_name']:25s} {res['model_params']:>8s} "
              f"{res.get('embedding_dim', '?'):>5} "
              f"{pol['within_cos_mean']:+8.3f} {pol['between_cos_mean']:+8.3f} "
              f"{pol['within_neg_pct']:7.1f}% {pol['between_neg_pct']:7.1f}% "
              f"{anti:>8s}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    output = {
        "experiment": "v31_codebert_polarity",
        "models_tested": args.models,
        "results": all_results,
        "elapsed_s": elapsed,
    }
    out_path = OUTPUT_DIR / "v31_codebert_polarity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
