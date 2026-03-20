"""E17: Outcome leakage quantification.

Three measurements to decompose the CQ encoder's AUROC advantage:

1. Mutual Information — MI(outcome_hash, anomaly_label) on training data.
   Expected: ~1.0 bit (near-perfect correlation in deterministic outcomes).

2. Outcome-Only Encoder — Train CQ encoder with tool_ids zeroed out
   during both training and inference. AUROC measures how much
   outcome hashes alone can predict anomalies.

3. Outcome-Shuffled CQ — Train a standard CQ encoder, then at scoring
   time randomly permute outcome_hashes across traces. If CQ drops to
   seq-only levels: 100% leakage. If elevated: structural learning exists.

Leakage fraction:
    leakage_frac = (cq_auroc - shuffled_cq_auroc) / (cq_auroc - seq_only_auroc)

Usage:
    python scripts/run_leakage_analysis.py
    python scripts/run_leakage_analysis.py --seeds 42,123,456,789,1024
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    load_traces,
)
from trustlens.encoder import build_consequence_encoder, build_encoder
from trustlens.stats import bootstrap_ci


def _hash_outcome(outcome: str) -> int:
    return (
        int(hashlib.md5(outcome.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        % (2**16)
    )


def compute_mutual_information(traces: list) -> dict:
    """Compute MI(outcome_hash, anomaly_label) on a set of traces.

    Uses plugin estimator: MI = H(Y) + H(O) - H(Y,O) where Y=anomaly,
    O=outcome_hash_bucket (bucketed to 256 bins for stable estimation).
    """
    n_bins = 256
    labels = []
    outcome_bins = []

    for trace in traces:
        for action in trace.actions:
            labels.append(action.label)
            h = _hash_outcome(action.outcome) % n_bins
            outcome_bins.append(h)

    labels = np.array(labels)
    outcome_bins = np.array(outcome_bins)
    n = len(labels)

    if n == 0:
        return {"mi_bits": 0.0, "n_actions": 0}

    def _entropy(arr):
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-15))

    h_y = _entropy(labels)
    h_o = _entropy(outcome_bins)

    joint = labels * n_bins + outcome_bins
    h_yo = _entropy(joint)

    mi = h_y + h_o - h_yo

    n_anomalous = int(labels.sum())
    n_normal = n - n_anomalous

    return {
        "mi_bits": float(mi),
        "h_label": float(h_y),
        "h_outcome": float(h_o),
        "h_joint": float(h_yo),
        "n_actions": n,
        "n_anomalous_actions": n_anomalous,
        "n_normal_actions": n_normal,
    }


def _train_encoder(
    seed: int,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int,
    patience: int,
    builder_fn,
    builder_kwargs: dict,
    use_outcome: bool = False,
) -> torch.nn.Module:
    """Thin wrapper around encoder training with early stopping."""
    from torch.optim.lr_scheduler import CosineAnnealingLR

    set_seeds(seed)
    normal_train = [t for t in train_traces if not t.has_anomaly]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(normal_train))
    n_val = max(1, int(len(normal_train) * 0.2))
    tr = [normal_train[i] for i in indices[n_val:]]
    val = [normal_train[i] for i in indices[:n_val]]

    train_ds = TraceDataset(tr, vocab, max_len=200)
    val_ds = TraceDataset(val, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds, batch_size=64, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )

    model = builder_fn(**builder_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for _epoch in range(epochs):
        model.train()
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)

            optimizer.zero_grad()
            if use_outcome:
                loss = model.training_step(
                    tool_ids, args_hashes, labels, mask,
                    outcome_hashes=outcome_hashes,
                )
            else:
                loss = model.training_step(tool_ids, args_hashes, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                tool_ids = batch["tool_ids"].to(device)
                args_hashes = batch["args_hashes"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                outcome_hashes = batch["outcome_hashes"].to(device)
                if use_outcome:
                    loss = model.training_step(
                        tool_ids, args_hashes, labels, mask,
                        outcome_hashes=outcome_hashes,
                    )
                else:
                    loss = model.training_step(tool_ids, args_hashes, labels, mask)
                vl += loss.item()
                vn += 1
        model.train()
        val_loss = vl / max(vn, 1)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def _score_traces(
    model: torch.nn.Module,
    ds: TraceDataset,
    device: torch.device,
    use_outcome: bool = False,
    shuffle_outcomes: bool = False,
    zero_tool_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Score traces, with optional outcome shuffling or tool zeroing."""
    dl = DataLoader(
        ds, batch_size=64, shuffle=False,
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
            outcome_hashes = batch["outcome_hashes"].to(device)

            if zero_tool_ids:
                tool_ids = torch.zeros_like(tool_ids)

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

            if use_outcome:
                scores = model.trace_score(
                    tool_ids, args_hashes, mask,
                    outcome_hashes=outcome_hashes,
                )
            else:
                scores = model.trace_score(tool_ids, args_hashes, mask)

            all_scores.append(scores.cpu().numpy())
            all_labels.append(
                (labels.sum(dim=1) > 0).numpy().astype(np.int32),
            )

    return np.concatenate(all_scores), np.concatenate(all_labels)


def run_leakage_analysis(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 30,
    patience: int = 5,
) -> dict:
    """Run all 3 leakage measurements for one seed."""
    test_ds = TraceDataset(test_traces, vocab)
    test_labels = np.array(
        [1 if t.has_anomaly else 0 for t in test_traces], dtype=np.int32,
    )

    results = {"seed": seed}

    # --- Measurement 1: Seq-Only Encoder (baseline) ---
    print(f"    [seed={seed}] Training seq-only encoder...")
    seq_model = _train_encoder(
        seed, train_traces, vocab, device, epochs, patience,
        builder_fn=build_encoder,
        builder_kwargs={"vocab_size": vocab.vocab_size, "use_args": False},
    )
    seq_test_scores, _ = _score_traces(seq_model, test_ds, device)
    seq_auroc = float(roc_auc_score(test_labels, seq_test_scores))
    results["seq_only_auroc"] = seq_auroc
    print(f"    [seed={seed}] Seq-only AUROC: {seq_auroc:.4f}")

    # --- Measurement 2: CQ Encoder (standard) ---
    print(f"    [seed={seed}] Training CQ encoder...")
    cq_model = _train_encoder(
        seed, train_traces, vocab, device, epochs, patience,
        builder_fn=build_consequence_encoder,
        builder_kwargs={"vocab_size": vocab.vocab_size, "outcome_weight": 0.5},
        use_outcome=True,
    )
    cq_test_scores, _ = _score_traces(
        cq_model, test_ds, device, use_outcome=True,
    )
    cq_auroc = float(roc_auc_score(test_labels, cq_test_scores))
    results["cq_auroc"] = cq_auroc
    print(f"    [seed={seed}] CQ AUROC: {cq_auroc:.4f}")

    # --- Measurement 3: CQ with shuffled outcomes ---
    print(f"    [seed={seed}] Scoring CQ with shuffled outcomes...")
    shuffled_scores, _ = _score_traces(
        cq_model, test_ds, device,
        use_outcome=True, shuffle_outcomes=True,
    )
    shuffled_auroc = float(roc_auc_score(test_labels, shuffled_scores))
    results["shuffled_cq_auroc"] = shuffled_auroc
    print(f"    [seed={seed}] Shuffled CQ AUROC: {shuffled_auroc:.4f}")

    # --- Measurement 4: Outcome-Only Encoder ---
    print(f"    [seed={seed}] Training outcome-only encoder (tool_ids zeroed)...")
    outcome_only_model = _train_encoder(
        seed, train_traces, vocab, device, epochs, patience,
        builder_fn=build_consequence_encoder,
        builder_kwargs={"vocab_size": vocab.vocab_size, "outcome_weight": 0.5},
        use_outcome=True,
    )
    outcome_only_scores, _ = _score_traces(
        outcome_only_model, test_ds, device,
        use_outcome=True, zero_tool_ids=True,
    )
    outcome_only_auroc = float(roc_auc_score(test_labels, outcome_only_scores))
    results["outcome_only_auroc"] = outcome_only_auroc
    print(f"    [seed={seed}] Outcome-only AUROC: {outcome_only_auroc:.4f}")

    # --- Leakage decomposition ---
    denom = cq_auroc - seq_auroc
    if abs(denom) > 1e-6:
        leakage_frac = (cq_auroc - shuffled_auroc) / denom
    else:
        leakage_frac = float("nan")

    results["leakage_fraction"] = float(leakage_frac)
    structural = float(1.0 - leakage_frac) if not np.isnan(leakage_frac) else float("nan")
    results["structural_fraction"] = structural
    results["cq_advantage"] = float(denom)

    print(
        f"    [seed={seed}] Leakage fraction: {leakage_frac:.3f}, "
        f"Structural: {1.0 - leakage_frac:.3f}"
    )

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E17: Outcome leakage analysis")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--encoder-epochs", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"E17: Outcome Leakage Analysis on {device}")
    print(f"Seeds: {seeds}")

    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    # --- Mutual Information (computed once, deterministic) ---
    print("\n--- Mutual Information ---")
    all_traces = train_traces + cal_traces + test_traces
    mi_result = compute_mutual_information(all_traces)
    print(f"  MI(outcome, label) = {mi_result['mi_bits']:.4f} bits")
    print(f"  H(label) = {mi_result['h_label']:.4f}, H(outcome) = {mi_result['h_outcome']:.4f}")

    # --- Per-seed analysis ---
    per_seed = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
        result = run_leakage_analysis(
            seed, train_traces, cal_traces, test_traces,
            vocab, device, epochs=args.encoder_epochs,
        )
        per_seed.append(result)

    # --- Aggregate ---
    seq_aurocs = np.array([r["seq_only_auroc"] for r in per_seed])
    cq_aurocs = np.array([r["cq_auroc"] for r in per_seed])
    shuffled_aurocs = np.array([r["shuffled_cq_auroc"] for r in per_seed])
    outcome_aurocs = np.array([r["outcome_only_auroc"] for r in per_seed])
    leakage_fracs = np.array([
        r["leakage_fraction"] for r in per_seed
        if not np.isnan(r["leakage_fraction"])
    ])

    aggregate = {
        "seq_only_auroc": bootstrap_ci(seq_aurocs),
        "cq_auroc": bootstrap_ci(cq_aurocs),
        "shuffled_cq_auroc": bootstrap_ci(shuffled_aurocs),
        "outcome_only_auroc": bootstrap_ci(outcome_aurocs),
        "leakage_fraction": bootstrap_ci(leakage_fracs) if len(leakage_fracs) > 0 else {},
    }

    print("\n" + "=" * 70)
    print("LEAKAGE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"{'Scorer':<25} {'AUROC':>12} {'CI':>22}")
    print("-" * 60)
    for label, key in [
        ("Seq-Only", "seq_only_auroc"),
        ("CQ (standard)", "cq_auroc"),
        ("CQ (shuffled)", "shuffled_cq_auroc"),
        ("Outcome-Only", "outcome_only_auroc"),
    ]:
        a = aggregate[key]
        print(
            f"{label:<25} {a['mean']:.4f}±{a['std']:.3f} "
            f"[{a['ci_lower']:.4f}, {a['ci_upper']:.4f}]"
        )
    if aggregate.get("leakage_fraction"):
        lf = aggregate["leakage_fraction"]
        print(f"\nLeakage fraction: {lf['mean']:.3f} [{lf['ci_lower']:.3f}, {lf['ci_upper']:.3f}]")
    print(f"MI(outcome, label): {mi_result['mi_bits']:.4f} bits")

    output = {
        "config": {
            "seeds": seeds,
            "encoder_epochs": args.encoder_epochs,
            "n_seeds": len(seeds),
        },
        "mutual_information": mi_result,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    out_path = tables_dir / "leakage_analysis.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
