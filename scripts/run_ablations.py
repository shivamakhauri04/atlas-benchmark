"""Run ablation experiments to validate each component's contribution.

Ablation configs:
  full:           Transformer (3L) + args_hash + Split CP  (baseline — already trained)
  no-transformer: MLP (0L) + args_hash + Split CP          (tests sequential modeling)
  no-args:        Transformer (3L) + args=0 + Split CP      (tests argument information)
  no-CP:          Transformer (3L) + args_hash + raw 95th %  (tests conformal calibration)

Usage:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --epochs 30
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from research_utils.device import get_device
from research_utils.reproducibility import set_seeds
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.encoder import build_consequence_encoder, build_encoder


def compute_scores(model, dataset, device, batch_size=64, zero_args=False):
    """Compute per-trace anomaly scores."""
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_traces,
        num_workers=0,
    )
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]

            if zero_args:
                args_hashes = torch.zeros_like(args_hashes)

            scores = model.trace_score(tool_ids, args_hashes, mask)
            all_scores.append(scores.cpu().numpy())
            all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))

    return np.concatenate(all_scores), np.concatenate(all_labels)


def train_model(model, train_ds, val_ds, device, epochs=30, lr=1e-4, batch_size=64,
                zero_args=False):
    """Train a model variant and return best val_loss."""
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            if zero_args:
                args_hashes = torch.zeros_like(args_hashes)

            loss = model.training_step(tool_ids, args_hashes, labels, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                tool_ids = batch["tool_ids"].to(device)
                args_hashes = batch["args_hashes"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)

                if zero_args:
                    args_hashes = torch.zeros_like(args_hashes)

                logits = model.forward(tool_ids, args_hashes, mask)
                shift_logits = logits[:, :-1].contiguous()
                shift_targets = tool_ids[:, 1:].contiguous()
                shift_mask = mask[:, 1:].contiguous()
                loss_val = F.cross_entropy(
                    shift_logits.view(-1, model.vocab_size),
                    shift_targets.view(-1),
                    reduction="none",
                )
                loss_val = loss_val.view(shift_targets.shape)
                loss_val = (
                    (loss_val * shift_mask.float()).sum()
                    / shift_mask.float().sum().clamp(min=1)
                )
                val_loss += loss_val.item()
                val_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}/{epochs-1}  train={avg_train:.4f}  val={avg_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return best_val_loss


def evaluate_variant(model, cal_ds, test_ds, device, alpha=0.05, zero_args=False):
    """Evaluate a model variant with Split CP."""
    cal_scores, cal_labels = compute_scores(model, cal_ds, device, zero_args=zero_args)
    test_scores, test_labels = compute_scores(model, test_ds, device, zero_args=zero_args)

    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n_neg = (test_labels == 0).sum()
    n_pos = test_labels.sum()
    coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
    recall = (preds[test_labels == 1] == 1).sum() / max(n_pos, 1)
    f1 = f1_score(test_labels, preds, zero_division=0.0)

    try:
        auroc = roc_auc_score(test_labels, test_scores)
    except ValueError:
        auroc = float("nan")

    # No-CP variant: use 95th percentile threshold instead of conformal
    pctile_tau = np.percentile(cal_scores[cal_labels == 0], 95)
    preds_nocp = (test_scores > pctile_tau).astype(np.int32)
    f1_nocp = f1_score(test_labels, preds_nocp, zero_division=0.0)
    coverage_nocp = (preds_nocp[test_labels == 0] == 0).sum() / max(n_neg, 1)

    return {
        "auroc": float(auroc),
        "f1_cp": float(f1),
        "coverage_cp": float(coverage),
        "recall_cp": float(recall),
        "tau_cp": float(tau),
        "f1_nocp": float(f1_nocp),
        "coverage_nocp": float(coverage_nocp),
    }


def train_consequence_model(
    model, train_ds, val_ds, device, epochs=30, lr=1e-4, batch_size=64,
):
    """Train a consequence-aware model variant and return best val_loss."""
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_traces, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            loss = model.training_step(
                tool_ids, args_hashes, labels, mask,
                outcome_hashes=outcome_hashes,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                tool_ids = batch["tool_ids"].to(device)
                args_hashes = batch["args_hashes"].to(device)
                outcome_hashes = batch["outcome_hashes"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)

                loss_val = model.training_step(
                    tool_ids, args_hashes, labels, mask,
                    outcome_hashes=outcome_hashes,
                )
                val_loss += loss_val.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_train = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch:3d}/{epochs-1}  train={avg_train:.4f}  val={avg_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    return best_val_loss


def evaluate_consequence_variant(model, cal_ds, test_ds, device, alpha=0.05):
    """Evaluate a consequence-aware model variant with Split CP."""
    cal_dl = DataLoader(
        cal_ds, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    test_dl = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=collate_traces, num_workers=0,
    )
    model.eval()

    def _score(dl):
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in dl:
                tool_ids = batch["tool_ids"].to(device)
                args_hashes = batch["args_hashes"].to(device)
                outcome_hashes = batch["outcome_hashes"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"]
                scores = model.trace_score(
                    tool_ids, args_hashes, mask,
                    outcome_hashes=outcome_hashes,
                )
                all_scores.append(scores.cpu().numpy())
                all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))
        return np.concatenate(all_scores), np.concatenate(all_labels)

    cal_scores, cal_labels = _score(cal_dl)
    test_scores, test_labels = _score(test_dl)

    tau = split_conformal_calibrate(cal_scores, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n_neg = (test_labels == 0).sum()
    n_pos = test_labels.sum()
    coverage = (preds[test_labels == 0] == 0).sum() / max(n_neg, 1)
    recall = (preds[test_labels == 1] == 1).sum() / max(n_pos, 1)
    f1 = f1_score(test_labels, preds, zero_division=0.0)

    try:
        auroc = roc_auc_score(test_labels, test_scores)
    except ValueError:
        auroc = float("nan")

    pctile_tau = np.percentile(cal_scores[cal_labels == 0], 95)
    preds_nocp = (test_scores > pctile_tau).astype(np.int32)
    f1_nocp = f1_score(test_labels, preds_nocp, zero_division=0.0)
    coverage_nocp = (preds_nocp[test_labels == 0] == 0).sum() / max(n_neg, 1)

    return {
        "auroc": float(auroc),
        "f1_cp": float(f1),
        "coverage_cp": float(coverage),
        "recall_cp": float(recall),
        "tau_cp": float(tau),
        "f1_nocp": float(f1_nocp),
        "coverage_nocp": float(coverage_nocp),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation experiments")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Multiple seeds for multi-seed ablations")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def run_single_seed_ablation(
    seed, vocab, train_traces, cal_traces, test_traces, device, args, checkpoint_path=None,
):
    """Run ablation experiments for a single seed. Returns results dict."""
    set_seeds(seed)

    # Split training data 80/20 for train/val to avoid cal set leakage
    n_train_split = int(len(train_traces) * 0.8)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(train_traces))
    abl_train = [train_traces[i] for i in indices[:n_train_split]]
    abl_val = [train_traces[i] for i in indices[n_train_split:]]

    train_ds = TraceDataset(abl_train, vocab)
    val_ds = TraceDataset(abl_val, vocab)
    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    results = {}

    # --- Full model (already trained) ---
    full_model = build_encoder(vocab_size=vocab.vocab_size, n_layers=3)
    full_model = full_model.to(device)
    if checkpoint_path and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        full_model.load_state_dict(state["model_state_dict"])
    else:
        set_seeds(seed)
        train_model(full_model, train_ds, val_ds, device,
                     epochs=args.epochs, batch_size=args.batch_size)
    n_params_full = sum(p.numel() for p in full_model.parameters())
    metrics = evaluate_variant(full_model, cal_ds, test_ds, device, alpha=args.alpha)
    results["full"] = {"config": "3L transformer + args + Split CP",
                       "n_params": n_params_full, **metrics}

    # --- No-transformer (0L MLP) ---
    set_seeds(seed)
    notrans_model = build_encoder(vocab_size=vocab.vocab_size, n_layers=0)
    notrans_model = notrans_model.to(device)
    n_params_notrans = sum(p.numel() for p in notrans_model.parameters())
    t0 = time.time()
    best_val = train_model(
        notrans_model, train_ds, val_ds, device,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    dur = time.time() - t0
    metrics = evaluate_variant(notrans_model, cal_ds, test_ds, device, alpha=args.alpha)
    results["no_transformer"] = {
        "config": "0L (no attention) + args + Split CP",
        "n_params": n_params_notrans, "best_val_loss": float(best_val),
        "train_time_s": dur, **metrics,
    }

    # --- No-args (transformer + args=0) ---
    set_seeds(seed)
    noargs_model = build_encoder(vocab_size=vocab.vocab_size, n_layers=3)
    noargs_model = noargs_model.to(device)
    n_params_noargs = sum(p.numel() for p in noargs_model.parameters())
    t0 = time.time()
    best_val = train_model(
        noargs_model, train_ds, val_ds, device,
        epochs=args.epochs, batch_size=args.batch_size, zero_args=True,
    )
    dur = time.time() - t0
    metrics = evaluate_variant(
        noargs_model, cal_ds, test_ds, device, alpha=args.alpha, zero_args=True,
    )
    results["no_args"] = {
        "config": "3L transformer + no args + Split CP",
        "n_params": n_params_noargs, "best_val_loss": float(best_val),
        "train_time_s": dur, **metrics,
    }

    # --- 2x2 Factorial: Consequence-aware (no-args) ---
    # Key comparison: same use_args=False as no_args, adds outcome head
    print("  [consequence_no_args] Training consequence-aware encoder (no args)...")
    set_seeds(seed)
    cq_noargs = build_consequence_encoder(
        vocab_size=vocab.vocab_size, n_layers=3, use_args=False,
    )
    cq_noargs = cq_noargs.to(device)
    n_params_cq_noargs = sum(p.numel() for p in cq_noargs.parameters())
    t0 = time.time()
    best_val = train_consequence_model(
        cq_noargs, train_ds, val_ds, device,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    dur = time.time() - t0
    metrics = evaluate_consequence_variant(
        cq_noargs, cal_ds, test_ds, device, alpha=args.alpha,
    )
    results["consequence_no_args"] = {
        "config": "3L transformer + no args + outcome + Split CP",
        "n_params": n_params_cq_noargs, "best_val_loss": float(best_val),
        "train_time_s": dur, **metrics,
    }

    # --- 2x2 Factorial: Consequence-aware (full args) ---
    print("  [consequence_full] Training consequence-aware encoder (with args)...")
    set_seeds(seed)
    cq_full = build_consequence_encoder(
        vocab_size=vocab.vocab_size, n_layers=3, use_args=True,
    )
    cq_full = cq_full.to(device)
    n_params_cq_full = sum(p.numel() for p in cq_full.parameters())
    t0 = time.time()
    best_val = train_consequence_model(
        cq_full, train_ds, val_ds, device,
        epochs=args.epochs, batch_size=args.batch_size,
    )
    dur = time.time() - t0
    metrics = evaluate_consequence_variant(
        cq_full, cal_ds, test_ds, device, alpha=args.alpha,
    )
    results["consequence_full"] = {
        "config": "3L transformer + args + outcome + Split CP",
        "n_params": n_params_cq_full, "best_val_loss": float(best_val),
        "train_time_s": dur, **metrics,
    }

    return results


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Running ablations on {device}")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")
    train_traces = load_traces(args.data_dir / "train.jsonl")
    cal_traces = load_traces(args.data_dir / "cal.jsonl")
    test_traces = load_traces(args.data_dir / "test.jsonl")

    seeds = args.seeds if args.seeds else [args.seed]
    multi_seed = len(seeds) > 1
    all_seed_results = []

    for i, seed in enumerate(seeds):
        print(f"\n{'=' * 72}")
        print(f"Seed {seed} ({i + 1}/{len(seeds)})")
        print("=" * 72)

        # For single-seed mode with existing checkpoint, use it for full model
        ckpt = args.checkpoint if not multi_seed else None
        seed_results = run_single_seed_ablation(
            seed, vocab, train_traces, cal_traces, test_traces, device, args,
            checkpoint_path=ckpt,
        )
        all_seed_results.append({"seed": seed, **seed_results})

        for name, r in seed_results.items():
            print(
                f"  {r['config']:<35} AUROC={r['auroc']:.4f}  "
                f"F1(CP)={r['f1_cp']:.4f}  Cov={r['coverage_cp']:.4f}"
            )

    # --- Aggregate results ---
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY" + (" (multi-seed)" if multi_seed else ""))
    print("=" * 72)

    configs = [
        "full", "no_transformer", "no_args",
        "consequence_no_args", "consequence_full",
    ]
    metrics_keys = ["auroc", "f1_cp", "f1_nocp", "coverage_cp"]
    aggregate = {}

    hdr = f"{'Config':<35} {'Params':>10} {'AUROC':>12} {'F1(CP)':>12}"
    print(f"{hdr} {'F1(noCP)':>12} {'Cov':>12}")
    print("-" * 95)

    for cfg in configs:
        vals = {k: [] for k in metrics_keys}
        n_params = 0
        for sr in all_seed_results:
            if cfg in sr:
                n_params = sr[cfg]["n_params"]
                for k in metrics_keys:
                    vals[k].append(sr[cfg][k])

        if multi_seed:
            agg_entry = {"config": all_seed_results[0][cfg]["config"], "n_params": n_params,
                         "n_seeds": len(seeds), "seeds": seeds}
            cols = []
            for k in metrics_keys:
                arr = np.array(vals[k])
                agg_entry[f"{k}_mean"] = float(arr.mean())
                agg_entry[f"{k}_std"] = float(arr.std())
                agg_entry[f"{k}_values"] = vals[k]
                cols.append(f"{arr.mean():.4f}±{arr.std():.3f}")
            aggregate[cfg] = agg_entry
            row = " ".join(f"{c:>12}" for c in cols)
            print(f"{agg_entry['config']:<35} {n_params:>10,} {row}")
        else:
            r = all_seed_results[0][cfg]
            aggregate[cfg] = r
            print(
                f"{r['config']:<35} {r['n_params']:>10,} {r['auroc']:>12.4f} "
                f"{r['f1_cp']:>12.4f} {r['f1_nocp']:>12.4f} {r['coverage_cp']:>12.4f}"
            )

    output = {"seeds": seeds, "multi_seed": multi_seed, "aggregate": aggregate,
              "per_seed": all_seed_results}
    out_path = tables_dir / "ablations.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
