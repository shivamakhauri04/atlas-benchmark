"""Scorer-agnostic conformal prediction comparison (v4 — unified protocol).

Runs Split CP with multiple anomaly scorers under a single unified
training protocol: 80/20 train/val split with early stopping (patience=5
on val loss). No more 100% train anywhere.

v4 improvements over v3:
  - Unified 80/20 train/val + early stopping for ALL encoder scorers
  - Iso-parameter "Large Seq-Only" baselines (~11M params) for capacity control
  - Single protocol, single table, no footnotes

Scorers:
  1. Transformer Encoder — next-token NLL (2.7M params)
  2. LSTM Next-Token — single-layer LSTM NLL
  3. Isolation Forest — sklearn IF on hand-crafted features
  4. N-gram LM — bigram language model NLL
  5. LSTM+Outcome — LSTM with joint tool+outcome NLL (v13)
  6. Large Seq-Only (wide) — d_model=256, 3 layers (~11M params)
  7. Large Seq-Only (deep) — d_model=128, 8 layers (~11M params)
  8. Consequence-Aware Encoder — joint tool+outcome NLL (2.7M params)

Usage:
    python scripts/run_cp_comparison.py
    python scripts/run_cp_comparison.py --seeds 42,123,456 --alpha 0.05
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

from baselines.all_baselines import (
    IsolationForestBaseline,
    LSTMConsequenceBaseline,
    LSTMNextTokenBaseline,
    NGramBaseline,
)
from data.trace_loader import (
    TraceDataset,
    TraceVocab,
    collate_traces,
    load_traces,
    split_traces,
)
from trustlens.conformal import split_conformal_calibrate, split_conformal_predict
from trustlens.encoder import build_consequence_encoder, build_encoder, build_large_encoder
from trustlens.stats import bootstrap_ci, compute_pairwise_stats


def _split_train_val(
    traces: list,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Split traces into train/val with fixed seed for reproducibility."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(traces))
    n_val = max(1, int(len(traces) * val_frac))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return [traces[i] for i in train_idx], [traces[i] for i in val_idx]


def _compute_val_loss(
    model: torch.nn.Module,
    val_dl: DataLoader,
    device: torch.device,
    use_outcome: bool = False,
) -> float:
    """Compute mean validation loss."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            if use_outcome:
                outcome_hashes = batch["outcome_hashes"].to(device)
                loss = model.training_step(
                    tool_ids,
                    args_hashes,
                    labels,
                    mask,
                    outcome_hashes=outcome_hashes,
                )
            else:
                loss = model.training_step(tool_ids, args_hashes, labels, mask)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train_encoder_for_seed(
    seed: int,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
    use_args: bool = True,
    d_model: int = 128,
    builder_fn=None,
    builder_kwargs: dict | None = None,
) -> torch.nn.Module:
    """Train an encoder with 80/20 val split and early stopping.

    Unified protocol for all encoder-based scorers. The train_traces
    are split 80/20 internally; early stopping uses patience on val loss.

    Args:
        builder_fn: Optional factory to build the model (default: build_encoder).
        builder_kwargs: Extra kwargs passed to builder_fn.
    """
    set_seeds(seed)
    normal_train = [t for t in train_traces if not t.has_anomaly]
    tr, val = _split_train_val(normal_train, val_frac=0.2, seed=seed)

    train_ds = TraceDataset(tr, vocab, max_len=200)
    val_ds = TraceDataset(val, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_traces,
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_traces,
        num_workers=0,
    )

    if builder_fn is not None:
        kw = {"vocab_size": vocab.vocab_size}
        if builder_kwargs:
            kw.update(builder_kwargs)
        model = builder_fn(**kw).to(device)
    else:
        model = build_encoder(
            vocab_size=vocab.vocab_size,
            d_model=d_model,
            use_args=use_args,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
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

        val_loss = _compute_val_loss(model, val_dl, device)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def compute_encoder_scores(
    model: torch.nn.Module,
    dataset: TraceDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-trace scores from the transformer encoder."""
    from torch.utils.data import DataLoader

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
            scores = model.trace_score(tool_ids, args_hashes, mask)
            all_scores.append(scores.cpu().numpy())
            all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))
    return np.concatenate(all_scores), np.concatenate(all_labels)


def train_consequence_encoder_for_seed(
    seed: int,
    train_traces: list,
    vocab: TraceVocab,
    device: torch.device,
    epochs: int = 50,
    patience: int = 5,
    d_model: int = 128,
    outcome_weight: float = 0.5,
    use_args: bool = False,
) -> torch.nn.Module:
    """Train a consequence-aware encoder with 80/20 val + early stopping."""
    set_seeds(seed)
    normal_train = [t for t in train_traces if not t.has_anomaly]
    tr, val = _split_train_val(normal_train, val_frac=0.2, seed=seed)

    train_ds = TraceDataset(tr, vocab, max_len=200)
    val_ds = TraceDataset(val, vocab, max_len=200)
    train_dl = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_traces,
        num_workers=0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_traces,
        num_workers=0,
    )

    model = build_consequence_encoder(
        vocab_size=vocab.vocab_size,
        d_model=d_model,
        use_args=use_args,
        outcome_weight=outcome_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            tool_ids = batch["tool_ids"].to(device)
            args_hashes = batch["args_hashes"].to(device)
            outcome_hashes = batch["outcome_hashes"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            loss = model.training_step(
                tool_ids,
                args_hashes,
                labels,
                mask,
                outcome_hashes=outcome_hashes,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_loss = _compute_val_loss(model, val_dl, device, use_outcome=True)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def compute_consequence_scores(
    model: torch.nn.Module,
    dataset: TraceDataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-trace scores from the consequence-aware encoder."""
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
            outcome_hashes = batch["outcome_hashes"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"]
            scores = model.trace_score(
                tool_ids,
                args_hashes,
                mask,
                outcome_hashes=outcome_hashes,
            )
            all_scores.append(scores.cpu().numpy())
            all_labels.append((labels.sum(dim=1) > 0).numpy().astype(np.int32))
    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate_cp(
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float = 0.05,
    normal_only_cal: bool = True,
) -> dict:
    """Run Split CP on a scorer's output and compute metrics.

    Args:
        normal_only_cal: If True, calibrate on normal cal scores only
            (conditional coverage guarantee). If False, calibrate on
            all cal scores (marginal coverage guarantee).
    """
    if normal_only_cal:
        cal_scores_for_cp = cal_scores[cal_labels == 0]
    else:
        cal_scores_for_cp = cal_scores

    tau = split_conformal_calibrate(cal_scores_for_cp, alpha=alpha)
    preds = split_conformal_predict(test_scores, tau)

    n = len(test_labels)
    n_pos = int(test_labels.sum())
    n_neg = n - n_pos

    coverage = float((preds[test_labels == 0] == 0).sum() / max(n_neg, 1))
    recall = float((preds[test_labels == 1] == 1).sum() / max(n_pos, 1))
    f1 = float(f1_score(test_labels, preds, zero_division=0.0))

    try:
        auroc = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        auroc = float("nan")

    n_cal = len(cal_scores_for_cp)
    bound = alpha + 1.0 / (n_cal + 1)
    conditional_miscoverage = float(preds[test_labels == 0].mean())
    marginal_miscoverage = float(preds.mean())

    if normal_only_cal:
        bound_holds = bool(conditional_miscoverage <= bound + 1e-9)
    else:
        bound_holds = bool(marginal_miscoverage <= bound + 1e-9)

    return {
        "alpha": alpha,
        "tau": float(tau),
        "auroc": auroc,
        "f1": f1,
        "recall": recall,
        "coverage": coverage,
        "miscoverage": conditional_miscoverage,
        "marginal_miscoverage": marginal_miscoverage,
        "theoretical_bound": float(bound),
        "bound_holds": bound_holds,
        "n_cal": n_cal,
        "n_test": n,
        "n_anomalous": n_pos,
        "calibration": "normal_only" if normal_only_cal else "mixed",
    }


def _append_normal_only_cal_row(
    results: list[dict],
    scorer_name: str,
    seed: int,
    cal_scores: np.ndarray,
    cal_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    alpha: float,
) -> None:
    """Append a normal-only calibration ablation row for a scorer."""
    r_normal = evaluate_cp(
        cal_scores,
        cal_labels,
        test_scores,
        test_labels,
        alpha=alpha,
        normal_only_cal=True,
    )
    r_normal["scorer"] = f"{scorer_name} (normal-only cal)"
    r_normal["seed"] = seed
    results.append(r_normal)


def run_single_seed(
    seed: int,
    train_traces: list,
    cal_traces: list,
    test_traces: list,
    cal_labels: np.ndarray,
    test_labels: np.ndarray,
    vocab: TraceVocab,
    device: torch.device,
    alpha: float,
    batch_size: int,
    encoder_epochs: int = 30,
    use_args: bool = True,
    d_model: int = 128,
    include_consequence: bool = False,
    outcome_weight: float = 0.5,
    include_mixed_cal: bool = False,
    outcome_weights: list[float] | None = None,
    consequence_only: bool = False,
) -> list[dict]:
    """Run CP comparison for one seed. Returns results for all scorers.

    All scorers (including the encoder) are trained/fit from scratch
    with the given seed, ensuring real variance across seeds.

    Args:
        include_mixed_cal: If True, add a second result row per scorer
            using normal-only calibration as ablation comparison.
        outcome_weights: If provided, run consequence-aware encoder at
            each weight (overrides outcome_weight). Used for weight sweeps.
        consequence_only: If True, skip the 4 base scorers — only run
            consequence-aware variants.
    """
    set_seeds(seed)

    cal_ds = TraceDataset(cal_traces, vocab)
    test_ds = TraceDataset(test_traces, vocab)

    results = []

    if not consequence_only:
        # --- Scorer 1: Transformer Encoder (retrained per seed) ---
        model = train_encoder_for_seed(
            seed,
            train_traces,
            vocab,
            device,
            epochs=encoder_epochs,
            use_args=use_args,
            d_model=d_model,
        )
        enc_cal_scores, _ = compute_encoder_scores(model, cal_ds, device, batch_size)
        enc_test_scores, _ = compute_encoder_scores(
            model,
            test_ds,
            device,
            batch_size,
        )

        r = evaluate_cp(
            enc_cal_scores,
            cal_labels,
            enc_test_scores,
            test_labels,
            alpha=alpha,
            normal_only_cal=False,
        )
        r["scorer"] = "Transformer Encoder"
        r["seed"] = seed
        results.append(r)

        if include_mixed_cal:
            _append_normal_only_cal_row(
                results,
                "Transformer Encoder",
                seed,
                enc_cal_scores,
                cal_labels,
                enc_test_scores,
                test_labels,
                alpha,
            )

        # --- Scorer 2: LSTM Next-Token (retrained per seed) ---
        set_seeds(seed)
        lstm_bl = LSTMNextTokenBaseline()
        lstm_bl.fit_traces(train_traces, cal_traces, vocab)
        lstm_cal_scores = lstm_bl.score_traces(cal_traces, vocab)
        lstm_test_scores = lstm_bl.score_traces(test_traces, vocab)

        r = evaluate_cp(
            lstm_cal_scores,
            cal_labels,
            lstm_test_scores,
            test_labels,
            alpha=alpha,
            normal_only_cal=False,
        )
        r["scorer"] = "LSTM Next-Token"
        r["seed"] = seed
        results.append(r)

        if include_mixed_cal:
            _append_normal_only_cal_row(
                results,
                "LSTM Next-Token",
                seed,
                lstm_cal_scores,
                cal_labels,
                lstm_test_scores,
                test_labels,
                alpha,
            )

        # --- Scorer 3: Isolation Forest (refit per seed) ---
        set_seeds(seed)
        if_bl = IsolationForestBaseline(random_state=seed)
        if_bl.fit_traces(train_traces, cal_traces, vocab)
        if_cal_scores = if_bl.score_traces(cal_traces, vocab)
        if_test_scores = if_bl.score_traces(test_traces, vocab)

        r = evaluate_cp(
            if_cal_scores,
            cal_labels,
            if_test_scores,
            test_labels,
            alpha=alpha,
            normal_only_cal=False,
        )
        r["scorer"] = "Isolation Forest"
        r["seed"] = seed
        results.append(r)

        if include_mixed_cal:
            _append_normal_only_cal_row(
                results,
                "Isolation Forest",
                seed,
                if_cal_scores,
                cal_labels,
                if_test_scores,
                test_labels,
                alpha,
            )

        # --- Scorer 4: N-gram LM (refit per seed) ---
        set_seeds(seed)
        ngram_bl = NGramBaseline()
        ngram_bl.fit_traces(train_traces, cal_traces, vocab)
        ngram_cal_scores = ngram_bl.score_traces(cal_traces, vocab)
        ngram_test_scores = ngram_bl.score_traces(test_traces, vocab)

        r = evaluate_cp(
            ngram_cal_scores,
            cal_labels,
            ngram_test_scores,
            test_labels,
            alpha=alpha,
            normal_only_cal=False,
        )
        r["scorer"] = "N-gram LM"
        r["seed"] = seed
        results.append(r)

        if include_mixed_cal:
            _append_normal_only_cal_row(
                results,
                "N-gram LM",
                seed,
                ngram_cal_scores,
                cal_labels,
                ngram_test_scores,
                test_labels,
                alpha,
            )

        # --- Scorer 5: LSTM+Outcome (retrained per seed) ---
        set_seeds(seed)
        lstm_cq_bl = LSTMConsequenceBaseline(outcome_weight=outcome_weight)
        lstm_cq_bl.fit_traces(train_traces, cal_traces, vocab)
        lstm_cq_cal_scores = lstm_cq_bl.score_traces(cal_traces, vocab)
        lstm_cq_test_scores = lstm_cq_bl.score_traces(test_traces, vocab)

        r = evaluate_cp(
            lstm_cq_cal_scores,
            cal_labels,
            lstm_cq_test_scores,
            test_labels,
            alpha=alpha,
            normal_only_cal=False,
        )
        r["scorer"] = "LSTM+Outcome"
        r["seed"] = seed
        results.append(r)

        if include_mixed_cal:
            _append_normal_only_cal_row(
                results,
                "LSTM+Outcome",
                seed,
                lstm_cq_cal_scores,
                cal_labels,
                lstm_cq_test_scores,
                test_labels,
                alpha,
            )

    # --- Scorers 6-7: Iso-Parameter Large Seq-Only Baselines ---
    if not consequence_only:
        for variant, label in [
            ("wide", "Large Seq-Only (wide)"),
            ("deep", "Large Seq-Only (deep)"),
        ]:
            set_seeds(seed)
            large_model = train_encoder_for_seed(
                seed,
                train_traces,
                vocab,
                device,
                epochs=encoder_epochs,
                use_args=False,
                builder_fn=build_large_encoder,
                builder_kwargs={"variant": variant, "use_args": False},
            )
            lg_cal_scores, _ = compute_encoder_scores(
                large_model,
                cal_ds,
                device,
                batch_size,
            )
            lg_test_scores, _ = compute_encoder_scores(
                large_model,
                test_ds,
                device,
                batch_size,
            )

            r = evaluate_cp(
                lg_cal_scores,
                cal_labels,
                lg_test_scores,
                test_labels,
                alpha=alpha,
                normal_only_cal=False,
            )
            r["scorer"] = label
            r["seed"] = seed
            results.append(r)

            if include_mixed_cal:
                _append_normal_only_cal_row(
                    results,
                    label,
                    seed,
                    lg_cal_scores,
                    cal_labels,
                    lg_test_scores,
                    test_labels,
                    alpha,
                )

    # --- Scorer 8: Consequence-Aware Encoder (optional) ---
    if include_consequence or consequence_only:
        weights_to_run = outcome_weights if outcome_weights else [outcome_weight]
        for ow in weights_to_run:
            set_seeds(seed)
            cq_model = train_consequence_encoder_for_seed(
                seed,
                train_traces,
                vocab,
                device,
                epochs=encoder_epochs,
                d_model=d_model,
                outcome_weight=ow,
                use_args=use_args,
            )
            cq_cal_scores, _ = compute_consequence_scores(
                cq_model,
                cal_ds,
                device,
                batch_size,
            )
            cq_test_scores, _ = compute_consequence_scores(
                cq_model,
                test_ds,
                device,
                batch_size,
            )

            scorer_label = (
                f"Consequence-Aware (w={ow})"
                if len(weights_to_run) > 1
                else "Consequence-Aware Encoder"
            )

            r = evaluate_cp(
                cq_cal_scores,
                cal_labels,
                cq_test_scores,
                test_labels,
                alpha=alpha,
                normal_only_cal=False,
            )
            r["scorer"] = scorer_label
            r["seed"] = seed
            r["outcome_weight"] = ow
            results.append(r)

            if include_mixed_cal:
                _append_normal_only_cal_row(
                    results,
                    scorer_label,
                    seed,
                    cq_cal_scores,
                    cal_labels,
                    cq_test_scores,
                    test_labels,
                    alpha,
                )

    return results


def aggregate_results(all_results: list[dict]) -> dict:
    """Aggregate per-seed results into mean±std per scorer with bootstrap CIs."""
    scorers = sorted(set(r["scorer"] for r in all_results))
    agg = {}

    for scorer in scorers:
        runs = [r for r in all_results if r["scorer"] == scorer]
        auroc_vals = np.array([r["auroc"] for r in runs])
        f1_vals = np.array([r["f1"] for r in runs])
        cov_vals = np.array([r["coverage"] for r in runs])
        recall_vals = np.array([r["recall"] for r in runs])

        auroc_ci = bootstrap_ci(auroc_vals)
        f1_ci = bootstrap_ci(f1_vals)
        cov_ci = bootstrap_ci(cov_vals)
        recall_ci = bootstrap_ci(recall_vals)

        agg[scorer] = {
            "auroc_mean": float(auroc_vals.mean()),
            "auroc_std": float(auroc_vals.std()),
            "auroc_ci_lower": auroc_ci["ci_lower"],
            "auroc_ci_upper": auroc_ci["ci_upper"],
            "f1_mean": float(f1_vals.mean()),
            "f1_std": float(f1_vals.std()),
            "f1_ci_lower": f1_ci["ci_lower"],
            "f1_ci_upper": f1_ci["ci_upper"],
            "coverage_mean": float(cov_vals.mean()),
            "coverage_std": float(cov_vals.std()),
            "coverage_ci_lower": cov_ci["ci_lower"],
            "coverage_ci_upper": cov_ci["ci_upper"],
            "recall_mean": float(recall_vals.mean()),
            "recall_std": float(recall_vals.std()),
            "recall_ci_lower": recall_ci["ci_lower"],
            "recall_ci_upper": recall_ci["ci_upper"],
            "bound_holds_fraction": float(np.mean([r["bound_holds"] for r in runs])),
            "n_seeds": len(runs),
            "n_effective": len(runs) if auroc_vals.std() > 0 else 1,
        }

    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scorer-agnostic CP comparison (v4 — unified protocol)",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seeds", type=str, default="42,123,456,789,1024")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--encoder-epochs", type=int, default=50, help="Max epochs (early stopping with patience=5)"
    )
    p.add_argument(
        "--no-args", action="store_true", help="Disable args embedding (tool-sequence only)"
    )
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument(
        "--consequence", action="store_true", help="Include consequence-aware encoder as scorer"
    )
    p.add_argument(
        "--outcome-weight",
        type=float,
        default=0.5,
        help="Weight for outcome NLL in consequence scorer",
    )
    p.add_argument(
        "--mixed-cal",
        action="store_true",
        help="Add mixed-calibration rows for diagnostic comparison",
    )
    p.add_argument(
        "--consequence-only",
        action="store_true",
        help="Skip base scorers, only run consequence-aware variants",
    )
    p.add_argument(
        "--resplit",
        action="store_true",
        help="Re-split data per seed (proper three-way splitting for coverage guarantee)",
    )
    p.add_argument(
        "--outcome-weights",
        type=str,
        default=None,
        help="Comma-separated weights for consequence sweep (e.g. 0.1,0.3,0.5,0.7)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = get_device(args.device)
    use_args = not args.no_args

    include_consequence = args.consequence
    consequence_only = args.consequence_only
    include_mixed_cal = args.mixed_cal
    outcome_weights = (
        [float(w) for w in args.outcome_weights.split(",")] if args.outcome_weights else None
    )

    if consequence_only:
        include_consequence = True

    print(f"Running scorer-agnostic CP comparison (v4, unified protocol) on {device}")
    print(f"Seeds: {seeds}")
    print(
        f"Encoder: d_model={args.d_model}, use_args={use_args}, "
        f"max_epochs={args.encoder_epochs}, early_stop=patience5"
    )
    print("Protocol: 80/20 train/val + early stopping for ALL encoders")
    print("Calibration: mixed (marginal coverage, v12 — formal bound holds)")
    if consequence_only:
        print("Mode: CONSEQUENCE-ONLY (base scorers skipped)")
    if include_consequence:
        if outcome_weights:
            print(f"Consequence-aware scorer: WEIGHT SWEEP {outcome_weights}")
        else:
            print(f"Consequence-aware scorer: ENABLED (outcome_weight={args.outcome_weight})")
    if include_mixed_cal:
        print("Normal-only cal ablation: ENABLED (dual rows per scorer)")

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    vocab = TraceVocab.load(args.data_dir / "vocab.json")

    resplit = args.resplit
    if resplit:
        all_traces = (
            load_traces(args.data_dir / "train.jsonl")
            + load_traces(args.data_dir / "cal.jsonl")
            + load_traces(args.data_dir / "test.jsonl")
        )
        print(f"Loaded {len(all_traces)} total traces for per-seed re-splitting")
        n_anom = sum(1 for t in all_traces if any(a.label == 1 for a in t.actions))
        print(f"  {n_anom} anomalous ({100 * n_anom / len(all_traces):.1f}%)")
    else:
        train_traces = load_traces(args.data_dir / "train.jsonl")
        cal_traces = load_traces(args.data_dir / "cal.jsonl")
        test_traces = load_traces(args.data_dir / "test.jsonl")

        cal_labels = np.array(
            [1 if any(a.label == 1 for a in t.actions) else 0 for t in cal_traces],
            dtype=np.int32,
        )
        test_labels = np.array(
            [1 if any(a.label == 1 for a in t.actions) else 0 for t in test_traces],
            dtype=np.int32,
        )

        print(
            f"Cal: {len(cal_traces)} ({cal_labels.sum()} anomalous, "
            f"{(cal_labels == 0).sum()} normal for calibration)"
        )
        print(f"Test: {len(test_traces)} ({test_labels.sum()} anomalous)")

    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i + 1}/{len(seeds)}) ---")

        if resplit:
            seed_train, seed_cal, seed_test = split_traces(all_traces, seed=seed)
            seed_cal_labels = np.array(
                [1 if any(a.label == 1 for a in t.actions) else 0 for t in seed_cal],
                dtype=np.int32,
            )
            seed_test_labels = np.array(
                [1 if any(a.label == 1 for a in t.actions) else 0 for t in seed_test],
                dtype=np.int32,
            )
            print(
                f"  Split: train={len(seed_train)}, "
                f"cal={len(seed_cal)} ({seed_cal_labels.sum()} anom), "
                f"test={len(seed_test)} ({seed_test_labels.sum()} anom)"
            )
        else:
            seed_train = train_traces
            seed_cal = cal_traces
            seed_test = test_traces
            seed_cal_labels = cal_labels
            seed_test_labels = test_labels

        seed_results = run_single_seed(
            seed,
            seed_train,
            seed_cal,
            seed_test,
            seed_cal_labels,
            seed_test_labels,
            vocab,
            device,
            args.alpha,
            args.batch_size,
            encoder_epochs=args.encoder_epochs,
            use_args=use_args,
            d_model=args.d_model,
            include_consequence=include_consequence,
            outcome_weight=args.outcome_weight,
            include_mixed_cal=include_mixed_cal,
            outcome_weights=outcome_weights,
            consequence_only=consequence_only,
        )
        for r in seed_results:
            status = "HOLDS" if r["bound_holds"] else "FAILS"
            print(
                f"  {r['scorer']:<35} AUROC={r['auroc']:.4f}  "
                f"F1={r['f1']:.4f}  Cov={r['coverage']:.4f}  {status}"
            )
        all_results.extend(seed_results)

    # Aggregate
    agg = aggregate_results(all_results)

    print("\n" + "=" * 90)
    cal_mode = "mixed + normal-only ablation" if include_mixed_cal else "mixed"
    print(f"SCORER-AGNOSTIC CP COMPARISON (MULTI-SEED, {cal_mode.upper()})")
    print("=" * 90)
    print(f"{'Scorer':<35} {'AUROC':>12} {'F1(CP)':>12} {'Coverage':>14} {'Bound %':>10}")
    print("-" * 85)
    for scorer, a in agg.items():
        print(
            f"{scorer:<35} "
            f"{a['auroc_mean']:.4f}±{a['auroc_std']:.3f} "
            f"{a['f1_mean']:.4f}±{a['f1_std']:.3f} "
            f"{a['coverage_mean']:.4f}±{a['coverage_std']:.4f} "
            f"{a['bound_holds_fraction'] * 100:>6.0f}%"
        )
    print("-" * 85)

    # Filter to non-mixed-cal for headline stats
    primary_agg = {k: v for k, v in agg.items() if "(normal-only cal)" not in k}
    if primary_agg:
        all_hold = all(a["bound_holds_fraction"] == 1.0 for a in primary_agg.values())
        most_hold = sum(1 for a in primary_agg.values() if a["bound_holds_fraction"] >= 0.8)
        print(f"\nCoverage holds for ALL primary scorers on ALL seeds: {all_hold}")
        print(f"Primary scorers where bound holds >=80%: {most_hold}/{len(primary_agg)}")

        best = max(primary_agg.items(), key=lambda x: x[1]["f1_mean"])
        print(f"Best F1 under CP: {best[0]} (F1={best[1]['f1_mean']:.4f}±{best[1]['f1_std']:.3f})")

    # --- Statistical Tests ---
    # Pairwise tests against Transformer Encoder as reference
    primary_scorers = [k for k in agg if "(normal-only cal)" not in k]
    reference = "Transformer Encoder" if "Transformer Encoder" in primary_scorers else None
    stat_results = {}

    if reference and len(seeds) >= 2:
        print(f"\n{'=' * 90}")
        print("STATISTICAL TESTS (vs Transformer Encoder)")
        print("=" * 90)

        for metric in ["auroc", "f1"]:
            comparisons = compute_pairwise_stats(
                agg,
                reference,
                metric=metric,
                per_seed_results=all_results,
            )
            stat_results[metric] = comparisons

            print(f"\n  {metric.upper()} pairwise comparisons:")
            print(
                f"  {'Scorer':<35} {'Diff':>10} {'95% CI':>22} {'p-val':>8} {'p-BH':>8} {'Sig?':>5}"
            )
            print("  " + "-" * 90)
            for c in comparisons:
                sig = "*" if c.get("rejected_bh", False) else ""
                sig_ci = "+" if c["significant_by_ci"] else ""
                p_raw = f"{c['p_value']:.4f}" if not np.isnan(c["p_value"]) else "N/A"
                p_bh = (
                    f"{c['p_value_corrected']:.4f}"
                    if not np.isnan(c.get("p_value_corrected", float("nan")))
                    else "N/A"
                )
                print(
                    f"  {c['scorer']:<35} "
                    f"{c['difference_mean']:>+10.4f} "
                    f"[{c['difference_ci_lower']:>+.4f}, {c['difference_ci_upper']:>+.4f}] "
                    f"{p_raw:>8} {p_bh:>8} {sig}{sig_ci:>4}"
                )

        if len(seeds) < 6:
            print(
                f"\n  NOTE: N={len(seeds)} seeds — Wilcoxon signed-rank cannot "
                "achieve p<0.05 with N<6."
            )
            print("  Bootstrap CIs are the primary evidence; '+' = CI excludes zero.")

        # Flag N-gram determinism
        ngram_aurocs = [r["auroc"] for r in all_results if r["scorer"] == "N-gram LM"]
        if ngram_aurocs and np.std(ngram_aurocs) == 0.0:
            print(
                f"\n  NOTE: N-gram LM has std=0.0 across {len(ngram_aurocs)} seeds "
                "(deterministic scorer)."
            )
            print("  Reported as N=1 effective observation with bootstrap CI on test set.")

    # Save main results
    output = {
        "config": {
            "seeds": seeds,
            "alpha": args.alpha,
            "calibration": cal_mode,
            "n_seeds": len(seeds),
            "use_args": use_args,
            "consequence_only": consequence_only,
            "mixed_cal": include_mixed_cal,
            "resplit_per_seed": resplit,
        },
        "aggregate": agg,
        "statistical_tests": stat_results,
        "per_seed": all_results,
    }
    out_path = tables_dir / "cp_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    # Save weight sweep separately if multiple weights
    if outcome_weights and len(outcome_weights) > 1:
        weight_results = [r for r in all_results if "outcome_weight" in r]
        weight_agg = aggregate_results(weight_results)
        weight_output = {
            "config": {
                "seeds": seeds,
                "alpha": args.alpha,
                "outcome_weights": outcome_weights,
                "n_seeds": len(seeds),
            },
            "aggregate": weight_agg,
            "per_seed": weight_results,
        }
        ws_path = tables_dir / "weight_sweep.json"
        ws_path.write_text(json.dumps(weight_output, indent=2))
        print(f"Weight sweep saved to {ws_path}")


if __name__ == "__main__":
    main()
