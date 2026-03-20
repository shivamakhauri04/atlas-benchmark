"""v24 Diagnostics: ATBench Cohen's d + LSTM training diagnostics.

Addresses DA#22 weaknesses:
  M2: LSTM vocabulary mismatch / undertrained confound
  M4: repeat_rate not convincingly inverted
  m2: ATBench Cohen's d missing from 3-way table

Analyses:
  1. ATBench trace property distributions + Cohen's d (completes 3-way table)
  2. Cross-domain vocabulary overlap / UNK rates
  3. LSTM training convergence on ATBench (per-epoch loss)
  4. NLL distribution analysis per domain (normal vs anomalous)
  5. Trace examples (5 per domain) grounding mechanistic explanation

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_v24_diagnostics.py
    PYTHONPATH=../../lib:src uv run python scripts/run_v24_diagnostics.py --skip-swe
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from research_utils.device import get_device
from research_utils.reproducibility import set_seeds

OUTPUT_DIR = Path("output/tables")
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}


# ---------------------------------------------------------------------------
# Reused from run_hybrid_eval.py and run_domain_analysis.py
# ---------------------------------------------------------------------------


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled_std)


def d_magnitude(d: float) -> str:
    if np.isnan(d):
        return "undefined"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def compute_trace_properties(traces: list[dict]) -> dict[str, np.ndarray]:
    """Compute trace properties from evaluation-format dicts."""
    n = len(traces)
    lengths = np.zeros(n)
    diversities = np.zeros(n)
    repeat_rates = np.zeros(n)
    labels = np.zeros(n, dtype=np.int32)

    for i, t in enumerate(traces):
        actions = t["actions"]
        na = len(actions)
        lengths[i] = float(na)
        unique = len(set(actions))
        diversities[i] = 1.0 - (unique / max(na, 1))
        if na > 1:
            reps = sum(1 for j in range(1, na) if actions[j] == actions[j - 1])
            repeat_rates[i] = reps / (na - 1)
        labels[i] = t.get("label", 0 if t.get("safe", t.get("solved", False)) else 1)

    return {
        "lengths": lengths,
        "diversities": diversities,
        "repeat_rates": repeat_rates,
        "labels": labels,
    }


def analyze_properties(
    props: dict[str, np.ndarray],
    domain_name: str,
) -> dict:
    """Full property analysis with Cohen's d and KS tests."""
    labels = props["labels"]
    normal_mask = labels == 0
    anomalous_mask = labels == 1
    n_normal = int(normal_mask.sum())
    n_anom = int(anomalous_mask.sum())

    result = {
        "domain": domain_name,
        "n_normal": n_normal,
        "n_anomalous": n_anom,
        "properties": {},
    }

    for prop_name in ["lengths", "diversities", "repeat_rates"]:
        values = props[prop_name]
        normal_vals = values[normal_mask]
        anom_vals = values[anomalous_mask]

        d = cohens_d(anom_vals, normal_vals)
        ks_stat, ks_p = stats.ks_2samp(normal_vals, anom_vals)

        result["properties"][prop_name] = {
            "normal_mean": float(normal_vals.mean()),
            "normal_std": float(normal_vals.std()),
            "anomalous_mean": float(anom_vals.mean()),
            "anomalous_std": float(anom_vals.std()),
            "cohens_d": d,
            "cohens_d_magnitude": d_magnitude(d),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
        }

    return result


# ---------------------------------------------------------------------------
# 1. ATBench Cohen's d
# ---------------------------------------------------------------------------


def run_atbench_cohens_d() -> dict:
    """Compute Cohen's d for ATBench — completes the 3-way table."""
    from run_hybrid_eval import load_atbench_traces

    print("\n--- 1. ATBench Cohen's d ---")
    traces = load_atbench_traces()
    props = compute_trace_properties(traces)
    analysis = analyze_properties(props, "atbench")

    print(f"  ATBench: {analysis['n_normal']} normal, {analysis['n_anomalous']} anomalous")
    for pn, pv in analysis["properties"].items():
        print(
            f"    {pn}: normal={pv['normal_mean']:.3f}+/-{pv['normal_std']:.3f}, "
            f"anom={pv['anomalous_mean']:.3f}+/-{pv['anomalous_std']:.3f}, "
            f"d={pv['cohens_d']:.3f} ({pv['cohens_d_magnitude']}), "
            f"KS p={pv['ks_p_value']:.4f}"
        )

    return analysis


# ---------------------------------------------------------------------------
# 2. Cross-domain vocabulary overlap and UNK rates
# ---------------------------------------------------------------------------


def build_vocab_from_traces(traces: list[dict], min_freq: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for t in traces:
        counter.update(t["actions"])
    vocab = dict(SPECIAL_TOKENS)
    for cmd, freq in counter.most_common():
        if freq >= min_freq:
            vocab[cmd] = len(vocab)
    return vocab


def compute_unk_rate(
    traces: list[dict], vocab: dict[str, int],
) -> dict:
    """Compute UNK token rate when applying a vocabulary to traces."""
    total_tokens = 0
    unk_tokens = 0
    unk_id = vocab.get("<UNK>", 1)
    for t in traces:
        for action in t["actions"]:
            total_tokens += 1
            if action not in vocab:
                unk_tokens += 1

    rate = unk_tokens / max(total_tokens, 1)
    return {
        "total_tokens": total_tokens,
        "unk_tokens": unk_tokens,
        "unk_rate": rate,
        "vocab_size": len(vocab),
    }


def run_vocab_analysis(
    domain_traces: dict[str, list[dict]],
    min_freq_map: dict[str, int],
) -> dict:
    """Cross-domain vocabulary overlap and UNK rate analysis."""
    print("\n--- 2. Cross-Domain Vocabulary Analysis ---")

    vocabs = {}
    for domain, traces in domain_traces.items():
        mf = min_freq_map.get(domain, 2)
        vocabs[domain] = build_vocab_from_traces(traces, min_freq=mf)

    result = {"per_domain": {}, "cross_domain_unk": {}}

    for domain, vocab in vocabs.items():
        n_special = len(SPECIAL_TOKENS)
        n_commands = len(vocab) - n_special
        unique_actions = set()
        for t in domain_traces[domain]:
            unique_actions.update(t["actions"])
        n_unique = len(unique_actions)
        in_vocab = sum(1 for a in unique_actions if a in vocab)
        oov = n_unique - in_vocab

        result["per_domain"][domain] = {
            "vocab_size": len(vocab),
            "n_commands": n_commands,
            "unique_actions_in_data": n_unique,
            "in_vocab": in_vocab,
            "out_of_vocab": oov,
            "oov_rate": oov / max(n_unique, 1),
        }
        print(
            f"  {domain}: vocab={len(vocab)} ({n_commands} commands), "
            f"unique_actions={n_unique}, OOV={oov} ({oov / max(n_unique, 1):.1%})"
        )

    # Cross-domain: apply each domain's vocab to every other domain
    for src_domain, src_vocab in vocabs.items():
        for tgt_domain, tgt_traces in domain_traces.items():
            key = f"{src_domain}_vocab_on_{tgt_domain}"
            unk_info = compute_unk_rate(tgt_traces, src_vocab)
            result["cross_domain_unk"][key] = unk_info
            print(
                f"  {key}: UNK rate={unk_info['unk_rate']:.1%} "
                f"({unk_info['unk_tokens']}/{unk_info['total_tokens']})"
            )

    return result


# ---------------------------------------------------------------------------
# 3. LSTM training convergence on ATBench
# ---------------------------------------------------------------------------


def train_lstm_with_diagnostics(
    train_traces: list[dict],
    vocab: dict[str, int],
    device: torch.device,
    epochs: int = 30,
    d_model: int = 64,
    domain_name: str = "unknown",
) -> tuple[object, list[dict]]:
    """Train LSTM and record per-epoch diagnostics."""
    from run_hybrid_eval import (
        ActionSeqDataset,
        LSTMScorer,
        collate_fn,
    )

    print(f"\n  Training LSTM on {domain_name} ({len(train_traces)} traces, "
          f"d_model={d_model}, epochs={epochs})...")

    model = LSTMScorer(len(vocab), d_model=d_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    from torch.utils.data import DataLoader

    ds = ActionSeqDataset(train_traces, vocab, [0] * len(train_traces))
    loader = DataLoader(ds, batch_size=64, shuffle=True, collate_fn=collate_fn)

    epoch_logs = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for seqs, _ in loader:
            seqs = seqs.to(device)
            logits = model(seqs[:, :-1])
            targets = seqs[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        lr = optimizer.param_groups[0]["lr"]
        epoch_logs.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "lr": lr,
        })
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, lr={lr:.6f}")

    return model, epoch_logs


# ---------------------------------------------------------------------------
# 4. NLL distribution analysis
# ---------------------------------------------------------------------------


def compute_nll_distributions(
    model: object,
    traces: list[dict],
    vocab: dict[str, int],
    labels: np.ndarray,
    device: torch.device,
    domain_name: str,
) -> dict:
    """Compute NLL distributions for normal vs anomalous traces."""
    from run_hybrid_eval import compute_ncs

    print(f"\n  Computing NLL distributions for {domain_name}...")
    nll_scores = compute_ncs(model, traces, vocab, device)

    normal_mask = labels == 0
    anomalous_mask = labels == 1
    normal_nll = nll_scores[normal_mask]
    anom_nll = nll_scores[anomalous_mask]

    d = cohens_d(anom_nll, normal_nll)

    result = {
        "domain": domain_name,
        "n_traces": len(traces),
        "n_normal": int(normal_mask.sum()),
        "n_anomalous": int(anomalous_mask.sum()),
        "normal_nll": {
            "mean": float(normal_nll.mean()),
            "std": float(normal_nll.std()),
            "median": float(np.median(normal_nll)),
            "q25": float(np.percentile(normal_nll, 25)),
            "q75": float(np.percentile(normal_nll, 75)),
        },
        "anomalous_nll": {
            "mean": float(anom_nll.mean()),
            "std": float(anom_nll.std()),
            "median": float(np.median(anom_nll)),
            "q25": float(np.percentile(anom_nll, 25)),
            "q75": float(np.percentile(anom_nll, 75)),
        },
        "cohens_d": d,
        "cohens_d_magnitude": d_magnitude(d),
        "anti_predictive": d < 0,
    }

    print(f"    Normal NLL: {result['normal_nll']['mean']:.4f}+/-{result['normal_nll']['std']:.4f}")
    print(f"    Anomalous NLL: {result['anomalous_nll']['mean']:.4f}+/-{result['anomalous_nll']['std']:.4f}")
    print(f"    Cohen's d: {d:.3f} ({d_magnitude(d)}) {'[ANTI-PREDICTIVE]' if d < 0 else ''}")

    return result


# ---------------------------------------------------------------------------
# 5. Trace examples
# ---------------------------------------------------------------------------


def extract_trace_examples(
    traces: list[dict],
    labels: np.ndarray,
    domain_name: str,
    n_examples: int = 5,
) -> dict:
    """Extract representative trace examples for each class."""
    normal_mask = labels == 0
    anomalous_mask = labels == 1

    normal_traces = [traces[i] for i in range(len(traces)) if normal_mask[i]]
    anom_traces = [traces[i] for i in range(len(traces)) if anomalous_mask[i]]

    # Sort by trace length to show diversity
    normal_sorted = sorted(normal_traces, key=lambda t: len(t["actions"]))
    anom_sorted = sorted(anom_traces, key=lambda t: len(t["actions"]))

    # Pick evenly spaced examples
    def pick_examples(sorted_list: list, n: int) -> list:
        if len(sorted_list) <= n:
            return sorted_list
        step = len(sorted_list) / n
        return [sorted_list[int(i * step)] for i in range(n)]

    normal_ex = pick_examples(normal_sorted, n_examples)
    anom_ex = pick_examples(anom_sorted, n_examples)

    def summarize(t: dict) -> dict:
        actions = t["actions"]
        na = len(actions)
        unique = len(set(actions))
        repeat = 0.0
        if na > 1:
            reps = sum(1 for j in range(1, na) if actions[j] == actions[j - 1])
            repeat = reps / (na - 1)
        return {
            "trace_id": t.get("trace_id", t.get("instance_id", "?")),
            "n_actions": na,
            "unique_tools": unique,
            "low_diversity": 1.0 - (unique / max(na, 1)),
            "repeat_rate": repeat,
            "first_5_actions": actions[:5],
            "last_5_actions": actions[-5:] if na > 5 else actions,
        }

    return {
        "domain": domain_name,
        "normal_examples": [summarize(t) for t in normal_ex],
        "anomalous_examples": [summarize(t) for t in anom_ex],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="v24 Diagnostics")
    parser.add_argument("--skip-swe", action="store_true", help="Skip SWE-agent (slow)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atbench-epochs", type=int, default=30)
    parser.add_argument("--atbench-d-model", type=int, default=64)
    args = parser.parse_args()

    set_seeds(args.seed)
    device = get_device()
    print(f"Device: {device}")

    t0 = time.time()
    all_results: dict = {"version": "v24"}

    # --- Load datasets ---
    from run_hybrid_eval import load_atbench_traces, load_synthetic_traces

    atb_traces = load_atbench_traces()
    synth_traces = load_synthetic_traces()

    swe_traces = None
    if not args.skip_swe:
        swe_cache = Path("data/swe_agent_cache.json")
        if swe_cache.exists():
            print("Loading cached SWE-agent data...")
            swe_traces = json.loads(swe_cache.read_text())
            print(f"  Loaded {len(swe_traces)} cached traces")
        else:
            try:
                from run_hybrid_eval import load_swe_agent_data
                swe_traces = load_swe_agent_data(5000)
            except Exception as e:
                print(f"  WARNING: Could not load SWE-agent data: {e}")

    # --- 1. ATBench Cohen's d ---
    atb_analysis = run_atbench_cohens_d()
    all_results["atbench_cohens_d"] = atb_analysis

    # --- 2. Cross-domain vocabulary analysis ---
    domain_traces = {"atbench": atb_traces, "synthetic": synth_traces}
    min_freq_map = {"atbench": 2, "synthetic": 2, "swe_agent": 5}
    if swe_traces:
        domain_traces["swe_agent"] = swe_traces

    vocab_analysis = run_vocab_analysis(domain_traces, min_freq_map)
    all_results["vocabulary_analysis"] = vocab_analysis

    # --- 3. ATBench LSTM training convergence ---
    print("\n--- 3. ATBench LSTM Training Convergence ---")
    rng = np.random.RandomState(args.seed)

    atb_safe = [t for t in atb_traces if t["safe"]]
    atb_unsafe = [t for t in atb_traces if not t["safe"]]
    rng.shuffle(atb_safe)
    rng.shuffle(atb_unsafe)

    ns, nu = len(atb_safe), len(atb_unsafe)
    nts, ntu = int(0.6 * ns), int(0.6 * nu)
    ncs, ncu = int(0.2 * ns), int(0.2 * nu)

    atb_train = atb_safe[:nts] + atb_unsafe[:ntu]
    atb_cal = atb_safe[nts:nts + ncs] + atb_unsafe[ntu:ntu + ncu]
    atb_test = atb_safe[nts + ncs:] + atb_unsafe[ntu + ncu:]
    atb_train_safe = [t for t in atb_train if t["safe"]]

    print(f"  ATBench splits: train_safe={len(atb_train_safe)}, "
          f"cal={len(atb_cal)}, test={len(atb_test)}")

    atb_vocab = build_vocab_from_traces(atb_traces, min_freq=2)

    # UNK rate in training data itself
    train_unk = compute_unk_rate(atb_train_safe, atb_vocab)
    print(f"  ATBench train UNK rate: {train_unk['unk_rate']:.1%}")

    atb_model, atb_epoch_logs = train_lstm_with_diagnostics(
        atb_train_safe, atb_vocab, device,
        epochs=args.atbench_epochs,
        d_model=args.atbench_d_model,
        domain_name="atbench",
    )

    convergence_result = {
        "domain": "atbench",
        "n_train_safe": len(atb_train_safe),
        "d_model": args.atbench_d_model,
        "epochs": args.atbench_epochs,
        "train_unk_rate": train_unk["unk_rate"],
        "epoch_logs": atb_epoch_logs,
        "final_loss": atb_epoch_logs[-1]["avg_loss"],
        "converged": (
            atb_epoch_logs[-1]["avg_loss"] < atb_epoch_logs[0]["avg_loss"] * 0.5
        ),
        "loss_ratio_final_vs_initial": (
            atb_epoch_logs[-1]["avg_loss"] / max(atb_epoch_logs[0]["avg_loss"], 1e-10)
        ),
    }
    all_results["atbench_convergence"] = convergence_result
    print(f"  Final/Initial loss ratio: "
          f"{convergence_result['loss_ratio_final_vs_initial']:.3f}")
    print(f"  Converged: {convergence_result['converged']}")

    # --- 4. NLL distribution analysis ---
    print("\n--- 4. NLL Distribution Analysis ---")
    atb_test_labels = np.array([t["label"] for t in atb_test])
    atb_nll = compute_nll_distributions(
        atb_model, atb_test, atb_vocab, atb_test_labels, device, "atbench",
    )
    all_results["atbench_nll_distributions"] = atb_nll

    # Also check NLL on cal set (for CP context)
    atb_cal_labels = np.array([t["label"] for t in atb_cal])
    atb_cal_nll = compute_nll_distributions(
        atb_model, atb_cal, atb_vocab, atb_cal_labels, device, "atbench_cal",
    )
    all_results["atbench_cal_nll_distributions"] = atb_cal_nll

    # --- 5. Trace examples ---
    print("\n--- 5. Trace Examples ---")
    atb_props = compute_trace_properties(atb_test)
    atb_examples = extract_trace_examples(
        atb_test, atb_test_labels, "atbench", n_examples=5,
    )
    all_results["atbench_trace_examples"] = atb_examples

    synth_labels = np.array([t["label"] for t in synth_traces])
    synth_examples = extract_trace_examples(
        synth_traces, synth_labels, "synthetic", n_examples=5,
    )
    all_results["synthetic_trace_examples"] = synth_examples

    if swe_traces:
        swe_labels = np.array([0 if t["solved"] else 1 for t in swe_traces])
        swe_examples = extract_trace_examples(
            swe_traces, swe_labels, "swe_agent", n_examples=5,
        )
        all_results["swe_trace_examples"] = swe_examples

    # --- 3-way Cohen's d summary ---
    print("\n--- 3-Way Cohen's d Summary ---")
    # Load existing domain analysis for synthetic + SWE-agent
    existing_file = OUTPUT_DIR / "domain_analysis_v18_10seed.json"
    three_way = {"atbench": atb_analysis}
    if existing_file.exists():
        existing = json.loads(existing_file.read_text())
        if "synthetic_properties" in existing:
            three_way["synthetic"] = existing["synthetic_properties"]
        if "swe_properties" in existing:
            three_way["swe_agent"] = existing["swe_properties"]

    # Print 3-way comparison table
    prop_names = ["lengths", "diversities", "repeat_rates"]
    display_names = {
        "lengths": "trace_length",
        "diversities": "low_diversity",
        "repeat_rates": "repeat_rate",
    }
    print(f"\n  {'Feature':<15} {'Synthetic d':>12} {'SWE-agent d':>12} {'ATBench d':>12}")
    print(f"  {'-' * 51}")
    for pn in prop_names:
        syn_d = three_way.get("synthetic", {}).get("properties", {}).get(pn, {}).get("cohens_d", float("nan"))
        swe_d = three_way.get("swe_agent", {}).get("properties", {}).get(pn, {}).get("cohens_d", float("nan"))
        atb_d = atb_analysis["properties"].get(pn, {}).get("cohens_d", float("nan"))
        print(f"  {display_names[pn]:<15} {syn_d:>12.3f} {swe_d:>12.3f} {atb_d:>12.3f}")

    all_results["three_way_cohens_d"] = {
        display_names[pn]: {
            "synthetic": three_way.get("synthetic", {}).get("properties", {}).get(pn, {}).get("cohens_d"),
            "swe_agent": three_way.get("swe_agent", {}).get("properties", {}).get(pn, {}).get("cohens_d"),
            "atbench": atb_analysis["properties"].get(pn, {}).get("cohens_d"),
        }
        for pn in prop_names
    }

    # --- Save ---
    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "v24_diagnostics.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n{'=' * 60}")
    print(f"v24 Diagnostics saved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
