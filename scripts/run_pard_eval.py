"""v31: PARD cross-domain evaluation.

Trains PARD (3 correction modes), DAPL, DANN, CORAL, LSTM-only, and PAFC
on source domain, evaluates cross-domain on target domains.
Reports AUROC, coverage, coverage gap, and polarity reversal diagnostics.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_pard_eval.py
    PYTHONPATH=../../lib:src uv run python scripts/run_pard_eval.py --seeds 42 --quick
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


def _get_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_domain_sequences(domain_name, vocab, *, max_traces=5000):
    """Load domain traces and encode to integer sequences + labels."""
    from data.trace_loader import generate_synthetic_traces, load_atbench

    if domain_name == "synthetic":
        traces = generate_synthetic_traces(n_traces=2000, anomaly_rate=0.15, seed=42)
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


def _load_swe_traces(max_traces):
    import re

    from data.schema import Action, Trace

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets: uv add --dev datasets") from exc

    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)
    traces = []
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions_raw = []
        for turn in example["trajectory"]:
            if turn.get("role") == "ai" and turn.get("text"):
                blocks = re.findall(r"```\s*\n?(.*?)\n?```", turn["text"], re.DOTALL)
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
                Action(tool=t, args="{}", context="swe", label=0 if solved else 1)
                for t in actions_raw
            ]
            traces.append(Trace(
                trace_id=f"swe_{len(traces):06d}",
                actions=trace_actions,
                source="swe_agent",
            ))
    return traces


def build_vocab_from_domains(domain_names):
    from data.trace_loader import TraceVocab, generate_synthetic_traces, load_atbench

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


def evaluate_pard(
    src_seqs, src_labels, tgt_seqs, tgt_labels, vocab_size, seed,
    *, mode="diagonal", epochs=50, d_bottleneck=32, use_encoder=False,
):
    """Evaluate PARD on heuristic features (optionally + LSTM features)."""
    import torch

    from trustlens.pard import (
        CorrectionMode,
        PARDTrainer,
        pard_calibrate_heuristic,
        pard_predict_heuristic,
    )

    set_seeds(seed)
    device = _get_device()

    n_tgt = len(tgt_seqs)
    n_cal = n_tgt // 2
    idx = np.random.permutation(n_tgt)
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
    cal_seqs = [tgt_seqs[i] for i in cal_idx]
    cal_labels = [tgt_labels[i] for i in cal_idx]
    test_seqs = [tgt_seqs[i] for i in test_idx]
    test_labels_arr = np.array([tgt_labels[i] for i in test_idx])

    encoder = None
    history = {"total": [0.0, 0.0]}
    if use_encoder:
        trainer = PARDTrainer(
            vocab_size, d_bottleneck=d_bottleneck,
            lambda_adv=0.1, lambda_pol=0.5, lambda_cls=0.3,
            epochs=epochs, patience=5,
        )
        trainer.to(device)
        history = trainer.train(src_seqs, src_labels, cal_seqs, cal_labels)
        encoder = trainer.encoder

    correction_mode = CorrectionMode(mode)
    correction, tau, clf, info = pard_calibrate_heuristic(
        src_seqs, src_labels, cal_seqs, cal_labels,
        encoder=encoder, mode=correction_mode,
        alpha=0.05, device=device,
    )

    preds, scores = pard_predict_heuristic(
        correction, clf, tau, test_seqs,
        encoder=encoder, device=device,
    )

    try:
        auroc = roc_auc_score(test_labels_arr, scores)
    except ValueError:
        auroc = 0.5

    coverage = 1.0 - float(np.mean(preds[test_labels_arr == 0]))
    coverage_gap = abs(coverage - 0.95)

    return {
        "auroc": auroc,
        "coverage": coverage,
        "coverage_gap": coverage_gap,
        "reversal_rate_pre": info["reversal_rate_pre"],
        "reversal_rate_post": info["reversal_rate_post"],
        "tau": tau,
        "d_features": info["d_features"],
        "train_loss_start": history["total"][0],
        "train_loss_end": history["total"][-1],
        "n_test": len(test_seqs),
        "seed": seed,
    }


def evaluate_baseline(
    method_name, src_seqs, src_labels, tgt_seqs, tgt_labels,
    vocab_size, seed, *, epochs=50,
):
    """Train and evaluate a baseline method."""
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
        trainer.to(_get_device())
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

    elif method_name == "DAPL":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.1, lambda_pol=0.5,
            epochs=epochs, patience=5,
        )
        trainer.to(_get_device())
        trainer.train(train_seqs, train_labels, tgt_seqs, tgt_labels)
        tgt_scores = trainer.score_traces(tgt_seqs)

    elif method_name == "PAFC":
        from trustlens.dapl import DAPLTrainer

        trainer = DAPLTrainer(
            vocab_size, lambda_adv=0.0, lambda_pol=0.0,
            epochs=epochs, patience=5,
        )
        trainer.to(_get_device())
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

    else:
        raise ValueError(f"Unknown method: {method_name}")

    tgt_scores_np = np.array(tgt_scores)
    tgt_labels_np = np.array(tgt_labels)
    try:
        auroc = roc_auc_score(tgt_labels_np, tgt_scores_np)
    except ValueError:
        auroc = 0.5

    return {"auroc": auroc, "n_test": len(tgt_labels), "seed": seed}


def run_pard_comparison(
    src_name, tgt_name, vocab, seeds, baseline_methods,
    *, epochs=50, max_traces=5000, d_bottleneck=32,
):
    """Run PARD + baselines across all seeds for one domain pair."""
    print(f"\n{'='*60}")
    print(f"  {src_name} -> {tgt_name}")
    print(f"{'='*60}")

    src_seqs, src_labels = load_domain_sequences(src_name, vocab, max_traces=max_traces)
    tgt_seqs, tgt_labels = load_domain_sequences(tgt_name, vocab, max_traces=max_traces)

    print(f"  Source: {len(src_seqs)} traces, {sum(src_labels)} anomalous")
    print(f"  Target: {len(tgt_seqs)} traces, {sum(tgt_labels)} anomalous")

    pard_methods = [
        ("PARD-diag", "diagonal", False),
        ("PARD-flip", "sign_flip", False),
        ("PARD-diag+LSTM", "diagonal", True),
    ]
    pard_results = {name: [] for name, _, _ in pard_methods}
    pard_details = {name: [] for name, _, _ in pard_methods}
    baseline_results = {m: [] for m in baseline_methods}
    baseline_results["LR-heuristic"] = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")

        for name, mode, use_enc in pard_methods:
            t0 = time.time()
            res = evaluate_pard(
                src_seqs, src_labels, tgt_seqs, tgt_labels,
                vocab.vocab_size, seed,
                mode=mode, epochs=epochs, d_bottleneck=d_bottleneck,
                use_encoder=use_enc,
            )
            elapsed = time.time() - t0
            pard_results[name].append(res["auroc"])
            pard_details[name].append(res)
            print(
                f"    {name:18s}: AUROC={res['auroc']:.4f}  "
                f"cov={res['coverage']:.3f}  "
                f"rev={res['reversal_rate_pre']:.2f}->{res['reversal_rate_post']:.2f}  "
                f"({elapsed:.1f}s)"
            )

        # Uncorrected LR baseline on heuristic features
        from trustlens.pard import compute_trace_heuristics
        from sklearn.linear_model import LogisticRegression as LR
        set_seeds(seed)
        n_tgt = len(tgt_seqs)
        idx = np.random.permutation(n_tgt)
        cal_s = [tgt_seqs[i] for i in idx[:n_tgt//2]]
        test_s = [tgt_seqs[i] for i in idx[n_tgt//2:]]
        test_l = np.array([tgt_labels[i] for i in idx[n_tgt//2:]])
        src_f = compute_trace_heuristics(src_seqs)
        test_f = compute_trace_heuristics(test_s)
        lr = LR(max_iter=1000).fit(src_f, src_labels)
        lr_auroc = roc_auc_score(test_l, lr.predict_proba(test_f)[:, 1])
        baseline_results["LR-heuristic"].append(lr_auroc)
        print(f"    {'LR-heuristic':18s}: AUROC={lr_auroc:.4f}")

        for method in baseline_methods:
            t0 = time.time()
            res = evaluate_baseline(
                method, src_seqs, src_labels, tgt_seqs, tgt_labels,
                vocab.vocab_size, seed, epochs=epochs,
            )
            elapsed = time.time() - t0
            baseline_results[method].append(res["auroc"])
            print(f"    {method:12s}: AUROC={res['auroc']:.4f}  ({elapsed:.1f}s)")

    all_results = {**pard_results, **baseline_results}
    return all_results, pard_details


def main():
    parser = argparse.ArgumentParser(description="v31: PARD cross-domain evaluation")
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-traces", type=int, default=5000)
    parser.add_argument("--d-bottleneck", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 5
        args.max_traces = 200

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baselines = ["LSTM-only", "DANN", "CORAL", "PAFC", "DAPL"]
    target_domains = ["atbench"]

    try:
        from datasets import load_dataset  # noqa: F401

        target_domains.append("swe_agent")
    except ImportError:
        print("  datasets not available, skipping SWE-agent")

    print("Building shared vocabulary...")
    vocab = build_vocab_from_domains(target_domains)
    print(f"  Vocab size: {vocab.vocab_size}")

    all_results = {}
    all_details = {}

    for tgt in target_domains:
        pair_key = f"synthetic->{tgt}"
        results, details = run_pard_comparison(
            "synthetic", tgt, vocab, args.seeds, baselines,
            epochs=args.epochs, max_traces=args.max_traces,
            d_bottleneck=args.d_bottleneck,
        )
        all_results[pair_key] = results
        all_details[pair_key] = details

        print(f"\n  Summary for {pair_key}:")
        for method, aurocs in results.items():
            arr = np.array(aurocs)
            print(f"    {method:20s}: {arr.mean():.4f} +/- {arr.std():.4f}")

    output = {
        "experiment": "v31_pard_eval",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "d_bottleneck": args.d_bottleneck,
        "results": all_results,
        "pard_details": {
            pair: {m: [d for d in dets] for m, dets in det.items()}
            for pair, det in all_details.items()
        },
    }
    out_path = OUTPUT_DIR / "v31_pard_eval.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
