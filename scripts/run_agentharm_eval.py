"""AgentHarm evaluation: detect harmful tool-use intent on real labels.

Unlike ToolBench and TRAIL, AgentHarm provides **real** benign/harmful
labels (not synthetic injection).  Each harmful prompt is paired with a
benign counterpart using the same target tool sequence — the only
difference is the *intent* expressed in the prompt.

Protocol:
  1. Load AgentHarm (176 benign + 176 harmful traces)
  2. Build vocab from AgentHarm tools
  3. Split into train / cal / test (stratified)
  4. Train tool-only vs consequence-aware encoders (per seed)
  5. Compare AUROC under Split CP

Usage:
    python scripts/run_agentharm_eval.py
    python scripts/run_agentharm_eval.py --seeds 42,123,456,789,1024
"""

import argparse
import json
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
    collate_traces,
    load_agentharm,
    split_traces,
)
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.encoder import build_consequence_encoder, build_encoder
from trustlens.stats import bootstrap_ci, paired_test


def train_encoder_for_seed(
    seed: int,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 30,
    d_model: int = 128,
) -> torch.nn.Module:
    """Train a standard encoder on AgentHarm traces."""
    set_seeds(seed)
    train_ds = TraceDataset(train_traces, vocab, max_len=50)
    train_dl = DataLoader(
        train_ds, batch_size=32, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )

    model = build_encoder(
        vocab_size=vocab.vocab_size,
        d_model=d_model,
        use_args=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for _epoch in range(epochs):
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
    """Train a consequence-aware encoder on AgentHarm traces."""
    set_seeds(seed)
    train_ds = TraceDataset(train_traces, vocab, max_len=50)
    train_dl = DataLoader(
        train_ds, batch_size=32, shuffle=True,
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
    for _epoch in range(epochs):
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-trace anomaly scores."""
    dl = DataLoader(
        dataset, batch_size=32, shuffle=False,
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
                kwargs["outcome_hashes"] = batch["outcome_hashes"].to(device)

            scores = model.trace_score(tool_ids, args_hashes, mask, **kwargs)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(
                (labels.sum(dim=1) > 0).numpy().astype(np.int32)
            )

    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate_cp(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Evaluate with Split CP."""
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
) -> dict:
    """Run evaluation for one seed."""
    cal_ds = TraceDataset(cal_traces, vocab, max_len=50)
    test_ds = TraceDataset(test_traces, vocab, max_len=50)

    cal_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_traces],
        dtype=np.int32,
    )
    test_labels = np.array(
        [1 if any(a.label == 1 for a in t.actions) else 0 for t in test_traces],
        dtype=np.int32,
    )

    print(f"  Training tool-only encoder (seed={seed})...")
    tool_model = train_encoder_for_seed(
        seed, train_traces, vocab, device, epochs=epochs, d_model=d_model,
    )
    tool_cal_scores, _ = compute_scores(tool_model, cal_ds, device)
    tool_test_scores, _ = compute_scores(tool_model, test_ds, device)
    tool_results = evaluate_cp(
        tool_cal_scores, cal_labels, tool_test_scores, test_labels, alpha,
    )
    tool_results["scorer"] = "Tool-Only Encoder"
    tool_results["seed"] = seed

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

    return {
        "seed": seed,
        "n_cal": len(cal_traces),
        "n_test": len(test_traces),
        "n_cal_anomalous": int(cal_labels.sum()),
        "n_test_anomalous": int(test_labels.sum()),
        "tool_only": tool_results,
        "consequence_aware": cq_results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentHarm evaluation")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--d-model", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)

    print(f"AgentHarm Evaluation on {device}")
    print(f"Seeds: {seeds}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AgentHarm traces...")
    traces = load_agentharm()
    print(f"Loaded {len(traces)} AgentHarm traces")

    n_benign = sum(1 for t in traces if not t.has_anomaly)
    n_harmful = sum(1 for t in traces if t.has_anomaly)
    print(f"Benign: {n_benign}, Harmful: {n_harmful}")

    vocab = TraceVocab().fit(traces)
    print(f"AgentHarm vocab size: {vocab.vocab_size}")

    train_traces, cal_traces, test_traces = split_traces(
        traces, train_frac=0.6, cal_frac=0.2, seed=42,
    )
    print(
        f"Split: train={len(train_traces)}, cal={len(cal_traces)}, "
        f"test={len(test_traces)}"
    )

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        result = run_single_seed(
            seed, train_traces, cal_traces, test_traces,
            vocab, device, args.alpha, args.epochs, args.d_model,
        )
        all_results.append(result)

        tool = result["tool_only"]
        cq = result["consequence_aware"]
        delta_auroc = cq["auroc"] - tool["auroc"]
        print(f"  Tool-Only:          AUROC={tool['auroc']:.4f}  F1={tool['f1']:.4f}")
        print(f"  Consequence-Aware:  AUROC={cq['auroc']:.4f}  F1={cq['f1']:.4f}")
        print(f"  Delta AUROC: {delta_auroc:+.4f}")

    tool_aurocs = np.array([r["tool_only"]["auroc"] for r in all_results])
    cq_aurocs = np.array([r["consequence_aware"]["auroc"] for r in all_results])

    print("\n" + "=" * 70)
    print("AGENTHARM EVALUATION RESULTS")
    print("=" * 70)

    tool_ci = bootstrap_ci(tool_aurocs)
    cq_ci = bootstrap_ci(cq_aurocs)
    print(
        f"Tool-Only AUROC:          {tool_ci['mean']:.4f} "
        f"[{tool_ci['ci_lower']:.4f}, {tool_ci['ci_upper']:.4f}]"
    )
    print(
        f"Consequence-Aware AUROC:  {cq_ci['mean']:.4f} "
        f"[{cq_ci['ci_lower']:.4f}, {cq_ci['ci_upper']:.4f}]"
    )

    auroc_test = paired_test(cq_aurocs, tool_aurocs)
    print(
        f"\nAUROC difference: {auroc_test['difference_mean']:+.4f} "
        f"[{auroc_test['difference_ci_lower']:+.4f}, "
        f"{auroc_test['difference_ci_upper']:+.4f}]"
    )
    if not np.isnan(auroc_test["p_value"]):
        print(f"Test: {auroc_test['test_used']}, p={auroc_test['p_value']:.4f}")
    else:
        print(f"Test: {auroc_test['test_used']}")

    if auroc_test["significant_by_ci"]:
        print("CONCLUSION: Consequence-aware improvement REPLICATES on AgentHarm.")
    else:
        print("CONCLUSION: No significant difference on AgentHarm.")

    output = {
        "dataset": "agentharm",
        "description": "Real benign/harmful labels from AgentHarm benchmark",
        "n_traces": len(traces),
        "n_benign": n_benign,
        "n_harmful": n_harmful,
        "seeds": seeds,
        "per_seed": all_results,
        "aggregate": {
            "tool_only_auroc": tool_ci,
            "consequence_auroc": cq_ci,
            "auroc_paired_test": auroc_test,
        },
    }

    out_path = tables_dir / "agentharm_eval.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
