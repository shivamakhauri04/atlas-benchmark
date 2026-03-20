"""Train the TrustLens trace encoder.

Platform-agnostic: works on local CPU/MPS, Kaggle T4/P100, Colab T4, Vast.ai.
Supports checkpoint resume for free-tier session resilience.

Usage:
    python scripts/train_encoder.py
    python scripts/train_encoder.py --epochs 10 --device cpu
    python scripts/train_encoder.py --resume-from checkpoints/
"""

import argparse
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.trace_loader import TraceDataset, TraceVocab, collate_traces, load_traces
from research_utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from research_utils.device import get_device
from research_utils.reproducibility import log_environment, set_seeds
from trustlens.config import TrainConfig
from trustlens.encoder import build_encoder


def build_dataloaders(
    config: TrainConfig,
    vocab: TraceVocab,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders.

    Training uses only normal traces (no anomalies) — the encoder learns
    a language model of normal tool sequences. Anomalous traces score
    higher NLL at inference time.
    """
    data_dir = Path(config.data_dir)

    all_train = load_traces(data_dir / "train.jsonl")
    cal_traces = load_traces(data_dir / "cal.jsonl")

    train_traces = [t for t in all_train if not t.has_anomaly]
    print(f"Training on {len(train_traces)}/{len(all_train)} normal traces")

    train_ds = TraceDataset(train_traces, vocab, max_len=config.max_len)
    val_ds = TraceDataset(cal_traces, vocab, max_len=config.max_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_traces,
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_traces,
        num_workers=0,
    )
    return train_dl, val_dl


def validate(
    model: torch.nn.Module,
    val_dl: DataLoader,
    device: torch.device,
) -> dict:
    """Run validation and return metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            loss = model.training_step(tool_ids, args_hashes, labels, mask)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return {"val_loss": avg_loss}


def train(config: TrainConfig) -> None:
    """Run the full training loop."""
    set_seeds(config.seed)
    device = get_device(config.device)
    print(f"Training on {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_environment(output_dir / "env.json")

    data_dir = Path(config.data_dir)
    vocab = TraceVocab.load(data_dir / "vocab.json")
    print(f"Vocabulary size: {vocab.vocab_size}")

    model = build_encoder(
        vocab_size=vocab.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        use_args=config.use_args,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    start_epoch = 0

    if config.resume_from is not None:
        ckpt_path = find_latest_checkpoint(config.resume_from)
        if ckpt_path is not None:
            start_epoch, metrics = load_checkpoint(ckpt_path, model, optimizer, device=device)
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch - 1}, metrics: {metrics}")
            for _ in range(start_epoch):
                scheduler.step()
        else:
            print(f"No checkpoint found in {config.resume_from}, starting fresh")

    train_dl, val_dl = build_dataloaders(config, vocab)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    if len(train_dl) == 0:
        raise RuntimeError("Train DataLoader is empty. Check data pipeline.")

    best_val_loss = float("inf")
    last_ckpt_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

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

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        val_metrics = validate(model, val_dl, device)
        val_loss = val_metrics["val_loss"]

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{config.epochs - 1}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"lr={lr_now:.2e}"
        )

        metrics = {"train_loss": avg_loss, "val_loss": val_loss, "lr": lr_now}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, metrics,
                config.checkpoint_dir, run_name=config.run_name + "_best",
            )

        elapsed = time.time() - last_ckpt_time
        ckpt_interval = config.checkpoint_every_epochs
        time_trigger = elapsed >= config.checkpoint_every_minutes * 60
        if (epoch + 1) % ckpt_interval == 0 or time_trigger:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                config.checkpoint_dir, run_name=config.run_name,
            )
            last_ckpt_time = time.time()

    save_checkpoint(
        model, optimizer, config.epochs - 1, metrics,
        config.checkpoint_dir, run_name=config.run_name,
    )
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TrustLens trace encoder")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--checkpoint-every-epochs", type=int, default=None)
    p.add_argument("--checkpoint-every-minutes", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--no-args", action="store_true", help="Disable args embedding (tool-sequence only)")
    return p.parse_args()


def main():
    args = parse_args()
    config = TrainConfig()

    overrides = {
        "data_dir": args.data_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "output_dir": args.output_dir,
        "resume_from": args.resume_from,
        "device": args.device,
        "seed": args.seed,
        "run_name": args.run_name,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "dataset": args.dataset,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "checkpoint_every_minutes": args.checkpoint_every_minutes,
        "alpha": args.alpha,
        "use_args": False if args.no_args else None,
    }
    config = config.apply_cli_overrides(overrides)
    if args.no_args and args.run_name is None:
        config = config.model_copy(update={"run_name": "no_args"})
    train(config)


if __name__ == "__main__":
    main()
