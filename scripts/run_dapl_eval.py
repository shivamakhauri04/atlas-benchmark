"""v28: DAPL vs DA baselines cross-domain evaluation.

Trains DAPL, DANN, CORAL, LSTM-only, and PAFC on source domain
(Synthetic), evaluates cross-domain on SWE-agent and ATBench.
10 seeds with Wilcoxon signed-rank + BH correction.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_dapl_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_dapl_eval.py --seeds 42 --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from research_utils.reproducibility import set_seeds
from scipy import stats
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = Path("output/tables")


def _get_device_str() -> str:
    import torch

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
    """Load domain traces and encode to integer sequences + labels."""
    from data.trace_loader import generate_synthetic_traces, load_atbench

    if domain_name == "synthetic":
        traces = generate_synthetic_traces(
            n_traces=2000, anomaly_rate=0.15, seed=42,
        )
    elif domain_name == "atbench":
        traces = load_atbench(max_traces=max_traces)
    elif domain_name == "swe_agent":
        traces = _load_swe_traces(max_traces)
    elif domain_name == "injecagent":
        from data.trace_loader import load_injecagent

        traces = load_injecagent(Path("data"), max_traces=max_traces)
    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in traces]
    labels = [1 if t.has_anomaly else 0 for t in traces]
    return seqs, labels


def _load_swe_traces(max_traces: int) -> list:
    """Load SWE-agent traces as Trace objects."""
    import re

    from data.schema import Action, Trace

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets: uv add --dev datasets") from exc

    ds = load_dataset(
        "nebius/SWE-agent-trajectories", split="train", streaming=True,
    )
    traces: list[Trace] = []
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions_raw: list[str] = []
        for turn in example["trajectory"]:
            if turn.get("role") == "ai" and turn.get("text"):
                blocks = re.findall(
                    r"```\s*\n?(.*?)\n?```", turn["text"], re.DOTALL,
                )
                for block in blocks:
                    block = block.strip()
                    if not block:
                        continue
                    first_line = block.split("\n")[0].strip()
                    cmd = first_line.split()[0] if first_line.split() else ""
                    cmd = cmd.lstrip("./!$")
                    if cmd:
                        actions_raw.append(cmd)
        if 3 <= len(actions_raw) <= 200:
            solved = bool(example.get("target"))
            trace_actions = [
                Action(
                    tool=t, args="{}", context="swe",
                    label=0 if solved else 1,
                )
                for t in actions_raw
            ]
            traces.append(Trace(
                trace_id=f"swe_{len(traces):06d}",
                actions=trace_actions,
                source="swe_agent",
            ))
    return traces


def build_vocab_from_domains(domain_names: list[str]):
    """Build vocab from multiple domain traces."""
    from data.trace_loader import (
        TraceVocab,
        generate_synthetic_traces,
        load_atbench,
    )

    all_traces = []
    all_traces.extend(generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42))
    for name in domain_names:
        if name == "atbench":
            try:
                all_traces.extend(load_atbench(max_traces=500))
            except Exception as e:
                print(f"  Skipping {name} for vocab: {e}")
        elif name == "swe_agent":
            try:
                all_traces.extend(_load_swe_traces(500))
            except Exception as e:
                print(f"  Skipping {name} for vocab: {e}")
    vocab = TraceVocab()
    vocab.fit(all_traces)
    return vocab


def evaluate_method(
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
    if method_name == "LSTM-only":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.0, lambda_pol=0.0,
            epochs=epochs, patience=5,
        )
        device = torch.device(_get_device_str())
        trainer.to(device)
        trainer.train(train_seqs, train_labels, tgt_seqs, tgt_labels)
        tgt_scores = trainer.score_traces(tgt_seqs)

    elif method_name == "DANN":
        from baselines.dann_coral import DANNBaseline

        dann = DANNBaseline(epochs=epochs, patience=5)
        dann.fit_cross_domain(
            vocab_size, train_seqs, train_labels,
            tgt_seqs, tgt_labels, cal_seqs,
        )
        tgt_scores = dann._score_seqs(tgt_seqs)
        tgt_scores = tgt_scores.tolist()

    elif method_name == "CORAL":
        from baselines.dann_coral import CORALBaseline

        coral = CORALBaseline(epochs=epochs, patience=5)
        coral.fit_cross_domain(
            vocab_size, train_seqs, train_labels,
            tgt_seqs, tgt_labels, cal_seqs,
        )
        tgt_scores = coral._score_seqs(tgt_seqs)
        tgt_scores = tgt_scores.tolist()

    elif method_name == "PAFC":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.0, lambda_pol=0.0,
            epochs=epochs, patience=5,
        )
        device = torch.device(_get_device_str())
        trainer.to(device)
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
            vocab_size,
            lambda_adv=0.1,
            lambda_pol=0.5,
            epochs=epochs,
            patience=5,
        )
        device = torch.device(_get_device_str())
        trainer.to(device)
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


def run_comparison(
    src_name: str,
    tgt_name: str,
    vocab,
    seeds: list[int],
    methods: list[str],
    *,
    epochs: int = 50,
    max_traces: int = 5000,
) -> dict[str, list[float]]:
    """Run all methods across all seeds for one domain pair."""
    print(f"\n{'='*60}")
    print(f"  {src_name} -> {tgt_name}")
    print(f"{'='*60}")

    src_seqs, src_labels = load_domain_sequences(src_name, vocab, max_traces=max_traces)
    tgt_seqs, tgt_labels = load_domain_sequences(tgt_name, vocab, max_traces=max_traces)

    print(f"  Source: {len(src_seqs)} traces, {sum(src_labels)} anomalous")
    print(f"  Target: {len(tgt_seqs)} traces, {sum(tgt_labels)} anomalous")

    results: dict[str, list[float]] = {m: [] for m in methods}

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        for method in methods:
            t0 = time.time()
            res = evaluate_method(
                method, src_seqs, src_labels, tgt_seqs, tgt_labels,
                vocab.vocab_size, seed, epochs=epochs,
            )
            elapsed = time.time() - t0
            results[method].append(res["auroc"])
            print(f"    {method:12s}: AUROC={res['auroc']:.4f}  ({elapsed:.1f}s)")

    return results


def statistical_tests(
    results: dict[str, list[float]],
    reference: str = "DAPL",
) -> dict[str, dict]:
    """Wilcoxon signed-rank tests + BH correction vs reference."""
    ref_scores = np.array(results.get(reference, []))
    if len(ref_scores) < 3:
        return {}

    raw_pvals: list[tuple[str, float]] = []
    for method, scores in results.items():
        if method == reference:
            continue
        scores_np = np.array(scores)
        if len(scores_np) != len(ref_scores):
            continue
        try:
            stat, pval = stats.wilcoxon(ref_scores, scores_np, alternative="greater")
        except ValueError:
            pval = 1.0
        raw_pvals.append((method, pval))

    raw_pvals.sort(key=lambda x: x[1])
    n_tests = len(raw_pvals)
    adjusted: dict[str, dict] = {}
    for rank, (method, pval) in enumerate(raw_pvals, 1):
        bh_pval = min(pval * n_tests / rank, 1.0)
        ref_mean = float(ref_scores.mean())
        method_mean = float(np.mean(results[method]))
        adjusted[method] = {
            "raw_p": pval,
            "bh_adjusted_p": bh_pval,
            "significant": bh_pval < 0.05,
            "mean_diff": ref_mean - method_mean,
            "ref_mean": ref_mean,
            "method_mean": method_mean,
        }

    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="v28: DAPL cross-domain evaluation")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 5
        args.max_traces = 200

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = ["LSTM-only", "DANN", "CORAL", "PAFC", "DAPL"]
    target_domains = ["atbench"]

    try:
        from datasets import load_dataset  # noqa: F401

        target_domains.append("swe_agent")
    except ImportError:
        print("  datasets not available, skipping SWE-agent")

    print("Building shared vocabulary...")
    vocab = build_vocab_from_domains(target_domains)
    print(f"  Vocab size: {vocab.vocab_size}")

    all_results: dict[str, dict[str, list[float]]] = {}
    all_stats: dict[str, dict] = {}

    for tgt in target_domains:
        pair_key = f"synthetic->{tgt}"
        results = run_comparison(
            "synthetic", tgt, vocab, args.seeds, methods,
            epochs=args.epochs, max_traces=args.max_traces,
        )
        all_results[pair_key] = results

        print(f"\n  Summary for {pair_key}:")
        for method, aurocs in results.items():
            arr = np.array(aurocs)
            print(f"    {method:12s}: {arr.mean():.4f} +/- {arr.std():.4f}")

        if len(args.seeds) >= 3:
            stat_tests = statistical_tests(results, reference="DAPL")
            all_stats[pair_key] = stat_tests
            print("\n  Statistical tests (DAPL vs others):")
            for method, info in stat_tests.items():
                sig = "*" if info["significant"] else ""
                print(
                    f"    vs {method:12s}: "
                    f"diff={info['mean_diff']:+.4f}  "
                    f"p={info['bh_adjusted_p']:.4f}{sig}"
                )

    output = {
        "experiment": "v28_dapl_eval",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "methods": methods,
        "results": {
            pair: {m: aurocs for m, aurocs in res.items()}
            for pair, res in all_results.items()
        },
        "statistics": all_stats,
    }
    out_path = OUTPUT_DIR / "v28_dapl_eval.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
