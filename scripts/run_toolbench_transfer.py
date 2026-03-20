"""ToolBench transfer experiment: validate on real tool-use traces.

Tests whether the consequence-aware encoder improvement replicates on
real ToolBench traces (not just synthetic data). This addresses the
concern that the improvement could be a synthetic data artifact.

Protocol:
  1. Load ToolBench traces (normal, real tool-use trajectories)
  2. Split into train / cal / test
  3. Build vocab from ToolBench data
  4. Inject synthetic anomalies into cal/test traces using existing
     anomaly patterns (ToolBench has no anomaly labels)
  5. Train tool-only vs consequence-aware encoders (per seed)
  6. Compare AUROC under Split CP

Usage:
    python scripts/run_toolbench_transfer.py --data-dir data
    python scripts/run_toolbench_transfer.py --seeds 42,123,456,789,1024
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    _inject_anomaly,
    collate_traces,
    load_toolbench,
    split_traces,
)
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.encoder import build_consequence_encoder, build_encoder
from trustlens.stats import bootstrap_ci, paired_test


def inject_anomalies_into_traces(
    traces: list,
    anomaly_rate: float = 0.10,
    seed: int = 42,
) -> list:
    """Inject anomalies into a fraction of traces.

    Since ToolBench traces are all normal (no anomaly labels), we
    inject our synthetic anomaly patterns to create a labeled test set.
    This tests whether models trained on real normal traces can detect
    the same anomaly patterns used in synthetic experiments.
    """
    rng = random.Random(seed)
    result = []

    for trace in traces:
        if rng.random() < anomaly_rate:
            new_actions = _inject_anomaly(list(trace.actions), rng)
            from data.schema import Trace
            result.append(Trace(
                trace_id=trace.trace_id,
                actions=new_actions,
                source=trace.source,
                task=trace.task,
            ))
        else:
            result.append(trace)

    return result


def train_encoder_for_seed(
    seed: int,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 30,
    d_model: int = 128,
    use_args: bool = False,
) -> torch.nn.Module:
    """Train a standard encoder on ToolBench normal traces."""
    set_seeds(seed)
    train_ds = TraceDataset(train_traces, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )

    model = build_encoder(
        vocab_size=vocab.vocab_size,
        d_model=d_model,
        use_args=use_args,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model.training_step(tool_ids, args_hashes, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def train_consequence_encoder_for_seed(
    seed: int,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 30,
    d_model: int = 128,
    outcome_weight: float = 0.5,
) -> torch.nn.Module:
    """Train a consequence-aware encoder on ToolBench normal traces."""
    set_seeds(seed)
    train_ds = TraceDataset(train_traces, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )

    model = build_consequence_encoder(
        vocab_size=vocab.vocab_size,
        d_model=d_model,
        use_args=False,
        outcome_weight=outcome_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model.training_step(
                tool_ids, args_hashes, labels, mask,
                outcome_hashes=outcome_hashes,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def compute_scores(
    model: torch.nn.Module,
    dataset: TraceDataset,
    device: torch.device,
    use_outcome: bool = False,
    shuffle_outcomes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-trace anomaly scores.

    Args:
        shuffle_outcomes: If True, randomly permute outcome_hashes across
            all valid positions before scoring. Tests whether CQ advantage
            comes from outcome signal or is a leakage artifact.
    """
    dl = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]

            kwargs = {}
            if use_outcome:
                outcome_hashes = batch["outcome_hashes"].to(device)
                if shuffle_outcomes:
                    flat = outcome_hashes[mask].cpu().numpy()
                    np.random.shuffle(flat)
                    shuffled = torch.zeros_like(outcome_hashes)
                    idx = 0
                    for i in range(outcome_hashes.size(0)):
                        for j in range(outcome_hashes.size(1)):
                            if mask[i, j]:
                                shuffled[i, j] = int(flat[idx])
                                idx += 1
                    outcome_hashes = shuffled.to(device)
                kwargs["outcome_hashes"] = outcome_hashes

            scores = model.trace_score(tool_ids, args_hashes, mask, **kwargs)
            all_scores.append(scores.cpu().numpy())
            all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))

    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate_cp(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Evaluate with Split CP (mixed calibration for marginal guarantee)."""
    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n_pos = int(test_labels.sum())
    n_neg = len(test_labels) - n_pos

    coverage = float((preds[test_labels == 0] == 0).sum() / max(n_neg, 1))
    recall = float((preds[test_labels == 1] == 1).sum() / max(n_pos, 1))
    f1 = float(f1_score(test_labels, preds, zero_division=0.0))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = float("nan")

    return {
        "auroc": auroc,
        "f1": f1,
        "coverage": coverage,
        "recall": recall,
        "tau": float(tau),
    }


def run_single_seed(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    alpha: float,
    epochs: int,
    d_model: int,
    anomaly_rate: float,
) -> dict:
    """Run transfer comparison for one seed."""
    # Inject anomalies into cal/test with this seed
    cal_injected = inject_anomalies_into_traces(cal_traces, anomaly_rate, seed)
    test_injected = inject_anomalies_into_traces(test_traces, anomaly_rate, seed + 1000)

    cal_ds = TraceDataset(cal_injected, vocab)
    test_ds = TraceDataset(test_injected, vocab)

    cal_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_injected],
        dtype=np.int32,
    )
    test_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in test_injected],
        dtype=np.int32,
    )

    # Tool-only encoder (use_args=False for matched comparison)
    print(f"  Training tool-only encoder (seed={seed})...")
    tool_model = train_encoder_for_seed(
        seed, train_traces, vocab, device, epochs=epochs,
        d_model=d_model, use_args=False,
    )
    tool_cal_scores, _ = compute_scores(tool_model, cal_ds, device)
    tool_test_scores, _ = compute_scores(tool_model, test_ds, device)
    tool_results = evaluate_cp(
        tool_cal_scores, cal_labels, tool_test_scores, test_labels, alpha,
    )
    tool_results["scorer"] = "Tool-Only Encoder"
    tool_results["seed"] = seed

    # Consequence-aware encoder
    print(f"  Training consequence-aware encoder (seed={seed})...")
    cq_model = train_consequence_encoder_for_seed(
        seed, train_traces, vocab, device, epochs=epochs, d_model=d_model,
    )
    cq_cal_scores, _ = compute_scores(cq_model, cal_ds, device, use_outcome=True)
    cq_test_scores, _ = compute_scores(cq_model, test_ds, device, use_outcome=True)
    cq_results = evaluate_cp(
        cq_cal_scores, cal_labels, cq_test_scores, test_labels, alpha,
    )
    cq_results["scorer"] = "Consequence-Aware Encoder"
    cq_results["seed"] = seed

    # CQ with shuffled outcomes (leakage check)
    print(f"  Scoring CQ with shuffled outcomes (seed={seed})...")
    shuf_cal_scores, _ = compute_scores(
        cq_model, cal_ds, device, use_outcome=True, shuffle_outcomes=True,
    )
    shuf_test_scores, _ = compute_scores(
        cq_model, test_ds, device, use_outcome=True, shuffle_outcomes=True,
    )
    shuf_results = evaluate_cp(
        shuf_cal_scores, cal_labels, shuf_test_scores, test_labels, alpha,
    )
    shuf_results["scorer"] = "CQ Shuffled Outcomes"
    shuf_results["seed"] = seed

    # Leakage fraction on ToolBench
    denom = cq_results["auroc"] - tool_results["auroc"]
    if abs(denom) > 1e-6:
        leakage_frac = (cq_results["auroc"] - shuf_results["auroc"]) / denom
    else:
        leakage_frac = float("nan")

    return {
        "seed": seed,
        "n_cal": len(cal_injected),
        "n_test": len(test_injected),
        "n_cal_anomalous": int(cal_labels.sum()),
        "n_test_anomalous": int(test_labels.sum()),
        "tool_only": tool_results,
        "consequence_aware": cq_results,
        "cq_shuffled": shuf_results,
        "leakage_fraction": leakage_frac,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ToolBench transfer experiment")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--max-traces", type=int, default=5000,
                   help="Max ToolBench traces to load")
    p.add_argument("--anomaly-rate", type=float, default=0.10,
                   help="Fraction of cal/test traces to inject anomalies into")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)

    print(f"ToolBench Transfer Experiment on {device}")
    print(f"Seeds: {seeds}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load ToolBench data
    print(f"Loading ToolBench traces from {args.data_dir}...")
    try:
        traces = load_toolbench(args.data_dir, max_traces=args.max_traces)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Download ToolBench data and place as data/toolbench_train.json")
        return

    print(f"Loaded {len(traces)} ToolBench traces")

    # Build vocab from ToolBench
    vocab = TraceVocab().fit(traces)
    print(f"ToolBench vocab size: {vocab.vocab_size}")

    # Split
    train_traces, cal_traces, test_traces = split_traces(
        traces, train_frac=0.6, cal_frac=0.2, seed=42,
    )
    print(f"Split: train={len(train_traces)}, cal={len(cal_traces)}, "
          f"test={len(test_traces)}")

    # Run per-seed comparisons
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/{len(seeds)}) ---")
        result = run_single_seed(
            seed, train_traces, cal_traces, test_traces,
            vocab, device, args.alpha, args.epochs, args.d_model,
            args.anomaly_rate,
        )
        all_results.append(result)

        tool = result["tool_only"]
        cq = result["consequence_aware"]
        shuf = result["cq_shuffled"]
        delta_auroc = cq["auroc"] - tool["auroc"]
        print(f"  Tool-Only:          AUROC={tool['auroc']:.4f}  F1={tool['f1']:.4f}")
        print(f"  Consequence-Aware:  AUROC={cq['auroc']:.4f}  F1={cq['f1']:.4f}")
        print(f"  CQ Shuffled:        AUROC={shuf['auroc']:.4f}  F1={shuf['f1']:.4f}")
        print(f"  Delta AUROC: {delta_auroc:+.4f}")
        print(f"  Leakage fraction: {result['leakage_fraction']:.4f}")

    # Aggregate
    tool_aurocs = np.array([r["tool_only"]["auroc"] for r in all_results])
    cq_aurocs = np.array([r["consequence_aware"]["auroc"] for r in all_results])
    shuf_aurocs = np.array([r["cq_shuffled"]["auroc"] for r in all_results])
    tool_f1s = np.array([r["tool_only"]["f1"] for r in all_results])
    cq_f1s = np.array([r["consequence_aware"]["f1"] for r in all_results])
    leakage_fracs = np.array([r["leakage_fraction"] for r in all_results])

    print("\n" + "=" * 70)
    print("TOOLBENCH TRANSFER RESULTS (with leakage check)")
    print("=" * 70)

    tool_ci = bootstrap_ci(tool_aurocs)
    cq_ci = bootstrap_ci(cq_aurocs)
    shuf_ci = bootstrap_ci(shuf_aurocs)
    leakage_ci = bootstrap_ci(leakage_fracs)
    print(f"Tool-Only AUROC:          {tool_ci['mean']:.4f} "
          f"[{tool_ci['ci_lower']:.4f}, {tool_ci['ci_upper']:.4f}]")
    print(f"Consequence-Aware AUROC:  {cq_ci['mean']:.4f} "
          f"[{cq_ci['ci_lower']:.4f}, {cq_ci['ci_upper']:.4f}]")
    print(f"CQ Shuffled AUROC:        {shuf_ci['mean']:.4f} "
          f"[{shuf_ci['ci_lower']:.4f}, {shuf_ci['ci_upper']:.4f}]")
    print(f"Leakage fraction:         {leakage_ci['mean']:.4f} "
          f"[{leakage_ci['ci_lower']:.4f}, {leakage_ci['ci_upper']:.4f}]")

    if leakage_ci["mean"] > 0.8:
        print("LEAKAGE: CQ advantage on ToolBench is primarily outcome leakage.")
    elif leakage_ci["mean"] > 0.5:
        print("PARTIAL LEAKAGE: CQ advantage on ToolBench is partially outcome leakage.")
    else:
        print("GENUINE: CQ advantage on ToolBench survives outcome shuffling.")

    # Paired test
    auroc_test = paired_test(cq_aurocs, tool_aurocs)
    print(f"\nAUROC difference (CQ vs Tool): {auroc_test['difference_mean']:+.4f} "
          f"[{auroc_test['difference_ci_lower']:+.4f}, "
          f"{auroc_test['difference_ci_upper']:+.4f}]")
    print(f"Test: {auroc_test['test_used']}, "
          f"p={auroc_test['p_value']:.4f}" if not np.isnan(auroc_test['p_value'])
          else f"Test: {auroc_test['test_used']}")

    shuf_test = paired_test(shuf_aurocs, tool_aurocs)
    print(f"AUROC difference (Shuffled vs Tool): {shuf_test['difference_mean']:+.4f} "
          f"[{shuf_test['difference_ci_lower']:+.4f}, "
          f"{shuf_test['difference_ci_upper']:+.4f}]")

    if auroc_test["significant_by_ci"]:
        print("CONCLUSION: Consequence-aware improvement REPLICATES on real traces.")
    else:
        print("CONCLUSION: Difference not significant on ToolBench transfer.")

    # Save
    output = {
        "config": {
            "seeds": seeds,
            "alpha": args.alpha,
            "max_traces": args.max_traces,
            "anomaly_rate": args.anomaly_rate,
            "data_source": "toolbench",
        },
        "aggregate": {
            "tool_only_auroc": tool_ci,
            "consequence_auroc": cq_ci,
            "cq_shuffled_auroc": shuf_ci,
            "tool_only_f1": bootstrap_ci(tool_f1s),
            "consequence_f1": bootstrap_ci(cq_f1s),
            "auroc_paired_test": auroc_test,
            "f1_paired_test": paired_test(cq_f1s, tool_f1s),
            "shuffled_vs_tool_test": shuf_test,
            "leakage_fraction": leakage_ci,
        },
        "per_seed": all_results,
    }
    out_path = tables_dir / "toolbench_transfer.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
