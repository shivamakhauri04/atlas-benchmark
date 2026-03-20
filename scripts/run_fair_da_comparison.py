"""Fair DA comparison: DANN and CORAL on heuristic features.

Addresses DA review F2: the original DANN/CORAL comparison used raw
sequences (AUROC ~0.50), which is unfair because the heuristic scorer
uses 3 pre-computed features. This script runs DANN and CORAL on the
SAME 3 heuristic features (trace_length, low_diversity, repeat_rate)
that PAFC uses, making the comparison apples-to-apples.

Architecture: 2-layer MLP (3 -> 32 -> 16 -> 1) with domain adaptation
loss instead of LSTM on raw sequences.

Usage:
    cd experiments/trustlens-auditable-agent-trust
    PYTHONPATH=../../lib:src uv run python scripts/run_fair_da_comparison.py
    PYTHONPATH=../../lib:src uv run python scripts/run_fair_da_comparison.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from research_utils.reproducibility import set_seeds
from sklearn.metrics import roc_auc_score

from data.domain_loader import load_all_domains
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")
FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FeatureMLP(nn.Module):
    """Small MLP: 3 features -> anomaly score."""

    def __init__(self, d_in: int = 3, d_hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden // 2)
        self.head = nn.Linear(d_hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x).squeeze(-1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class DomainClassifierMLP(nn.Module):
    """Binary domain classifier for DANN."""

    def __init__(self, d_in: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


def _z_score_features(
    features: dict[str, np.ndarray],
    mu: dict[str, float] | None = None,
    sigma: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """Z-score features into (N, 3) array."""
    cols = []
    if mu is None:
        mu = {}
        sigma = {}
        for name in FEATURE_NAMES:
            vals = features[name]
            mu[name] = float(vals.mean())
            sigma[name] = max(float(vals.std()), 1e-8)
    for name in FEATURE_NAMES:
        vals = features[name]
        cols.append((vals - mu[name]) / sigma[name])
    return np.column_stack(cols), mu, sigma


def _coral_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """CORAL loss: Frobenius norm of covariance difference."""
    d = src.size(1)
    s_mean = src.mean(0, keepdim=True)
    t_mean = tgt.mean(0, keepdim=True)
    s_c = src - s_mean
    t_c = tgt - t_mean
    n_s = max(src.size(0) - 1, 1)
    n_t = max(tgt.size(0) - 1, 1)
    cov_s = (s_c.T @ s_c) / n_s
    cov_t = (t_c.T @ t_c) / n_t
    diff = cov_s - cov_t
    return (diff * diff).sum() / (4 * d * d)


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def train_dann_on_features(
    src_X: np.ndarray,
    src_y: np.ndarray,
    tgt_X: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_adv: float = 0.1,
    seed: int = 42,
) -> FeatureMLP:
    """Train DANN on heuristic features."""
    set_seeds(seed)
    device = _get_device()
    d_in = src_X.shape[1]

    mlp = FeatureMLP(d_in=d_in).to(device)
    domain_clf = DomainClassifierMLP(d_in=16).to(device)

    params = list(mlp.parameters()) + list(domain_clf.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    src_t = torch.tensor(src_X, dtype=torch.float32, device=device)
    src_labels = torch.tensor(src_y, dtype=torch.float32, device=device)
    tgt_t = torch.tensor(tgt_X, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        mlp.train()
        domain_clf.train()
        progress = epoch / max(epochs - 1, 1)
        alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0

        scores = mlp(src_t)
        loss_task = F.binary_cross_entropy_with_logits(scores, src_labels)

        src_feat = mlp.features(src_t)
        tgt_feat = mlp.features(tgt_t)
        src_rev = GRL.apply(src_feat, alpha)
        tgt_rev = GRL.apply(tgt_feat, alpha)
        all_feat = torch.cat([src_rev, tgt_rev], dim=0)
        domain_labels = torch.cat([
            torch.zeros(src_feat.size(0), device=device),
            torch.ones(tgt_feat.size(0), device=device),
        ])
        loss_adv = F.binary_cross_entropy_with_logits(
            domain_clf(all_feat), domain_labels,
        )

        loss = loss_task + lambda_adv * loss_adv
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    mlp.eval()
    return mlp


def train_coral_on_features(
    src_X: np.ndarray,
    src_y: np.ndarray,
    tgt_X: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    lambda_coral: float = 0.5,
    seed: int = 42,
) -> FeatureMLP:
    """Train Deep CORAL on heuristic features."""
    set_seeds(seed)
    device = _get_device()
    d_in = src_X.shape[1]

    mlp = FeatureMLP(d_in=d_in).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=0.01)

    src_t = torch.tensor(src_X, dtype=torch.float32, device=device)
    src_labels = torch.tensor(src_y, dtype=torch.float32, device=device)
    tgt_t = torch.tensor(tgt_X, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        mlp.train()
        scores = mlp(src_t)
        loss_task = F.binary_cross_entropy_with_logits(scores, src_labels)

        src_feat = mlp.features(src_t)
        tgt_feat = mlp.features(tgt_t)
        loss_coral = _coral_loss(src_feat, tgt_feat)

        loss = loss_task + lambda_coral * loss_coral
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()

    mlp.eval()
    return mlp


def score_with_model(
    model: FeatureMLP,
    X: np.ndarray,
) -> np.ndarray:
    """Score target features with trained model."""
    device = next(model.parameters()).device
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=device)
        scores = torch.sigmoid(model(t)).cpu().numpy()
    return scores


def compute_domain_d(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for feat in FEATURE_NAMES:
        vals = features.get(feat, np.zeros(len(labels)))
        normal = vals[labels == 0]
        anomalous = vals[labels == 1]
        result[feat] = cohens_d(anomalous, normal)
    return result


def apply_pafc_correction(
    features: dict[str, np.ndarray],
    source_d: dict[str, float],
    target_d: dict[str, float],
) -> dict[str, np.ndarray]:
    corrected = {}
    for name, values in features.items():
        d_src = source_d.get(name, 0.0)
        d_tgt = target_d.get(name, 0.0)
        if abs(d_src) > 0.1 and abs(d_tgt) > 0.1 and (d_src > 0) != (d_tgt > 0):
            corrected[name] = -values
        else:
            corrected[name] = values
    return corrected


def heuristic_score(features: dict[str, np.ndarray]) -> np.ndarray:
    cols = []
    for name in sorted(features.keys()):
        vals = features[name]
        mu, sigma = vals.mean(), max(vals.std(), 1e-8)
        cols.append((vals - mu) / sigma)
    return np.mean(np.column_stack(cols), axis=1)


def evaluate_pair(
    src_features: dict[str, np.ndarray],
    src_labels: np.ndarray,
    tgt_features: dict[str, np.ndarray],
    tgt_labels: np.ndarray,
    seed: int,
    epochs: int = 100,
) -> dict[str, float]:
    """Evaluate all methods on one domain pair for one seed."""
    set_seeds(seed)
    results: dict[str, float] = {}

    src_d = compute_domain_d(src_features, src_labels)
    tgt_d = compute_domain_d(tgt_features, tgt_labels)

    raw_scores = heuristic_score(tgt_features)
    try:
        results["heuristic_raw"] = float(roc_auc_score(tgt_labels, raw_scores))
    except ValueError:
        results["heuristic_raw"] = 0.5

    corrected_tgt = apply_pafc_correction(tgt_features, src_d, tgt_d)
    pafc_scores = heuristic_score(corrected_tgt)
    try:
        results["heuristic_pafc"] = float(roc_auc_score(tgt_labels, pafc_scores))
    except ValueError:
        results["heuristic_pafc"] = 0.5

    src_X, mu, sigma = _z_score_features(src_features)
    tgt_X, _, _ = _z_score_features(tgt_features, mu, sigma)

    try:
        dann_model = train_dann_on_features(
            src_X, src_labels.astype(np.float32), tgt_X,
            epochs=epochs, seed=seed,
        )
        dann_scores = score_with_model(dann_model, tgt_X)
        results["dann_heuristic"] = float(roc_auc_score(tgt_labels, dann_scores))
    except Exception as e:
        print(f"      DANN failed: {e}")
        results["dann_heuristic"] = 0.5

    try:
        coral_model = train_coral_on_features(
            src_X, src_labels.astype(np.float32), tgt_X,
            epochs=epochs, seed=seed,
        )
        coral_scores = score_with_model(coral_model, tgt_X)
        results["coral_heuristic"] = float(roc_auc_score(tgt_labels, coral_scores))
    except Exception as e:
        print(f"      CORAL failed: {e}")
        results["coral_heuristic"] = 0.5

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair DA comparison: DANN/CORAL on heuristic features",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = 20

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  Fair DA Comparison: DANN/CORAL on Heuristic Features")
    print("=" * 70)

    print("\nLoading domains...")
    domain_data = load_all_domains()
    loaded = sorted(domain_data.keys())
    print(f"  Loaded: {loaded}")

    domain_d: dict[str, dict[str, float]] = {}
    for name, (features, labels) in domain_data.items():
        domain_d[name] = compute_domain_d(features, labels)

    all_pairs: list[tuple[str, str]] = []
    for da in loaded:
        for db in loaded:
            if da == db:
                continue
            n_reversed = sum(
                1 for f in FEATURE_NAMES
                if abs(domain_d[da].get(f, 0)) > 0.1
                and abs(domain_d[db].get(f, 0)) > 0.1
                and (domain_d[da][f] > 0) != (domain_d[db][f] > 0)
            )
            if n_reversed > 0:
                all_pairs.append((da, db))

    if not all_pairs:
        for da in loaded:
            for db in loaded:
                if da >= db:
                    continue
                all_pairs.append((da, db))
                all_pairs.append((db, da))

    print(f"\n  Evaluating {len(all_pairs)} domain pairs x {len(args.seeds)} seeds")

    results: dict[str, dict[str, list[float]]] = {}

    for pair_idx, (src, tgt) in enumerate(all_pairs):
        pair_key = f"{src}->{tgt}"
        print(f"\n  [{pair_idx + 1}/{len(all_pairs)}] {pair_key}")
        results[pair_key] = {
            "heuristic_raw": [],
            "heuristic_pafc": [],
            "dann_heuristic": [],
            "coral_heuristic": [],
        }

        src_feat, src_lab = domain_data[src]
        tgt_feat, tgt_lab = domain_data[tgt]

        for seed in args.seeds:
            r = evaluate_pair(
                src_feat, src_lab, tgt_feat, tgt_lab,
                seed=seed, epochs=args.epochs,
            )
            for method, val in r.items():
                results[pair_key][method].append(val)

            print(
                f"    seed={seed}: "
                f"raw={r['heuristic_raw']:.3f} "
                f"pafc={r['heuristic_pafc']:.3f} "
                f"dann={r['dann_heuristic']:.3f} "
                f"coral={r['coral_heuristic']:.3f}"
            )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for pair_key, methods in results.items():
        summary[pair_key] = {}
        for method, vals in methods.items():
            arr = np.array(vals)
            summary[pair_key][method] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }

    elapsed = time.time() - t0
    output = {
        "experiment": "fair_da_comparison",
        "description": "DANN/CORAL on heuristic features (3-dim) for fair comparison with PAFC",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "n_pairs": len(all_pairs),
        "domain_pairs": [f"{s}->{t}" for s, t in all_pairs],
        "results": results,
        "summary": summary,
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "fair_da_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  Summary (mean AUROC across seeds)")
    print(f"{'=' * 70}")
    print(f"  {'Pair':<30} {'Raw':>6} {'PAFC':>6} {'DANN':>6} {'CORAL':>6}")
    print(f"  {'-' * 54}")
    for pair_key, s in summary.items():
        print(
            f"  {pair_key:<30} "
            f"{s['heuristic_raw']['mean']:>6.3f} "
            f"{s['heuristic_pafc']['mean']:>6.3f} "
            f"{s['dann_heuristic']['mean']:>6.3f} "
            f"{s['coral_heuristic']['mean']:>6.3f}"
        )

    print(f"\n  Results saved to {out_path}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
