"""E18: Noisy outcome robustness sweep.

For each noise level, generates fresh synthetic data with outcome_noise,
trains CQ + seq-only encoders, and compares AUROC degradation.

Key question: Does CQ retain advantage over seq-only when outcomes are
noisy (realistic deployment), or does the advantage vanish (pure leakage)?

Noise levels:
  - 0.0: Deterministic outcomes (matches current setup)
  - 0.3: 30% of anomalous actions get normal-looking outcomes
  - 0.5: Coin flip — moderate noise
  - 0.7: Heavy noise — most anomalous outcomes look normal
  - 1.0: Outcomes carry zero anomaly signal

Usage:
    python scripts/run_noisy_outcome_sweep.py
    python scripts/run_noisy_outcome_sweep.py --noise-levels 0.0,0.1,0.3,0.5,0.7,1.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    generate_synthetic_traces,
    split_traces,
)
from trustlens.encoder import build_consequence_encoder, build_encoder
from trustlens.stats import bootstrap_ci


def _build_vocab(traces: list) -> TraceVocab:
    """Build vocabulary from traces."""
    return TraceVocab().fit(traces)


def _train_and_score(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int,
    patience: int,
    use_outcome: bool,
) -> dict:
    """Train encoder and compute test metrics."""
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

    if use_outcome:
        model = build_consequence_encoder(
            vocab_size=vocab.vocab_size, outcome_weight=0.5,
        ).to(device)
    else:
        model = build_encoder(
            vocab_size=vocab.vocab_size, use_args=False,
        ).to(device)

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

    # Score test set
    test_ds = TraceDataset(test_traces, vocab, max_len=200)
    test_dl = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    test_labels = np.array(
        [1 if t.has_anomaly else 0 for t in test_traces], dtype=np.int32,
    )

    all_scores = []
    with torch.no_grad():
        for batch in test_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            mask = batch["mask"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)

            if use_outcome:
                scores = model.trace_score(
                    tool_ids, args_hashes, mask,
                    outcome_hashes=outcome_hashes,
                )
            else:
                scores = model.trace_score(tool_ids, args_hashes, mask)
            all_scores.append(scores.cpu().numpy())

    test_scores = np.concatenate(all_scores)
    auroc = float(roc_auc_score(test_labels, test_scores))
    return {"auroc": auroc}


def run_noise_level(
    noise_p: float,
    seeds: list[int],
    device: torch.device,
    n_traces: int,
    epochs: int,
    patience: int,
) -> dict:
    """Run CQ + seq-only at a single noise level across seeds."""
    print(f"\n--- noise_p={noise_p:.1f} ---")
    per_seed = []

    for seed in seeds:
        print(f"  [seed={seed}] Generating data with noise_p={noise_p}...")
        traces = generate_synthetic_traces(
            n_traces=n_traces,
            anomaly_rate=0.05,
            seed=seed,
            outcome_noise=noise_p,
        )
        train, cal, test = split_traces(traces, seed=seed)
        vocab = _build_vocab(train + cal + test)

        print(f"  [seed={seed}] Training seq-only...")
        seq_result = _train_and_score(
            seed, train, cal, test, vocab, device, epochs, patience,
            use_outcome=False,
        )

        print(f"  [seed={seed}] Training CQ...")
        cq_result = _train_and_score(
            seed, train, cal, test, vocab, device, epochs, patience,
            use_outcome=True,
        )

        advantage = cq_result["auroc"] - seq_result["auroc"]
        print(
            f"  [seed={seed}] Seq={seq_result['auroc']:.4f} "
            f"CQ={cq_result['auroc']:.4f} Δ={advantage:+.4f}"
        )

        per_seed.append({
            "seed": seed,
            "noise_p": noise_p,
            "seq_only_auroc": seq_result["auroc"],
            "cq_auroc": cq_result["auroc"],
            "cq_advantage": advantage,
        })

    seq_aurocs = np.array([r["seq_only_auroc"] for r in per_seed])
    cq_aurocs = np.array([r["cq_auroc"] for r in per_seed])
    advantages = np.array([r["cq_advantage"] for r in per_seed])

    return {
        "noise_p": noise_p,
        "seq_only_auroc": bootstrap_ci(seq_aurocs),
        "cq_auroc": bootstrap_ci(cq_aurocs),
        "cq_advantage": bootstrap_ci(advantages),
        "per_seed": per_seed,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E18: Noisy outcome sweep")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456")
    p.add_argument(
        "--noise-levels", type=str, default="0.0,0.3,0.5,0.7,1.0",
        help="Comma-separated noise_p values",
    )
    p.add_argument("--n-traces", type=int, default=5000)
    p.add_argument("--encoder-epochs", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    noise_levels = [float(x) for x in args.noise_levels.split(",")]
    device = get_device(args.device)

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print(f"E18: Noisy Outcome Sweep on {device}")
    print(f"Seeds: {seeds}, Noise levels: {noise_levels}")
    print(f"Traces per level: {args.n_traces}")

    results = []
    for noise_p in noise_levels:
        level_result = run_noise_level(
            noise_p, seeds, device,
            n_traces=args.n_traces,
            epochs=args.encoder_epochs,
            patience=5,
        )
        results.append(level_result)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("NOISY OUTCOME SWEEP SUMMARY")
    print("=" * 80)
    print(
        f"{'noise_p':>8} {'Seq-Only':>14} {'CQ':>14} "
        f"{'Advantage':>14} {'Adv CI':>22}"
    )
    print("-" * 75)
    for r in results:
        sq = r["seq_only_auroc"]
        cq = r["cq_auroc"]
        adv = r["cq_advantage"]
        print(
            f"{r['noise_p']:>8.1f} "
            f"{sq['mean']:.4f}±{sq['std']:.3f} "
            f"{cq['mean']:.4f}±{cq['std']:.3f} "
            f"{adv['mean']:>+.4f}±{adv['std']:.3f} "
            f"[{adv['ci_lower']:>+.4f}, {adv['ci_upper']:>+.4f}]"
        )

    output = {
        "config": {
            "seeds": seeds,
            "noise_levels": noise_levels,
            "n_traces": args.n_traces,
            "encoder_epochs": args.encoder_epochs,
            "n_seeds": len(seeds),
        },
        "results": results,
    }
    out_path = tables_dir / "noisy_outcome_sweep.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
