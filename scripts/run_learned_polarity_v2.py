"""v30: Learned representation polarity analysis across ALL 8 datasets.

Trains an autoencoder on synthetic normal traces (heuristic features as input),
extracts bottleneck features, computes Cohen's d per dimension across all 8
datasets. Computes within-group vs between-group reversal rates on learned
features to test whether polarity reversal persists beyond hand-crafted features.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_learned_polarity_v2.py
    PYTHONPATH=../../lib:src uv run python scripts/run_learned_polarity_v2.py --quick
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data.domain_loader import DomainData, load_all_domains
from research_utils.reproducibility import set_seeds
from trustlens.polarity import cohens_d

OUTPUT_DIR = Path("output/tables")

SEMANTIC_GROUPS = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "injecagent", "agentharm", "rjudge"],
    "api_misuse": ["toolbench"],
}

HEURISTIC_FEATURES = ["trace_length", "low_diversity", "repeat_rate"]


class DeepAutoEncoder(nn.Module):
    """Autoencoder with configurable bottleneck for learned representations."""

    def __init__(self, in_dim: int = 3, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    X_normal: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
    bottleneck: int = 8,
) -> DeepAutoEncoder:
    set_seeds(seed)
    model = DeepAutoEncoder(in_dim=X_normal.shape[1], bottleneck=bottleneck)
    X_t = torch.tensor(X_normal, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        x_hat = model(X_t)
        loss = ((X_t - x_hat) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def extract_bottleneck(model: DeepAutoEncoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model.encode(X_t).numpy()


def get_group_for_domain(domain: str) -> str:
    for group, members in SEMANTIC_GROUPS.items():
        if domain in members:
            return group
    return "unknown"


def same_group(d1: str, d2: str) -> bool:
    return get_group_for_domain(d1) == get_group_for_domain(d2)


def check_sign_reversal(d_a: float, d_b: float, threshold: float = 0.1) -> bool:
    if abs(d_a) < threshold or abs(d_b) < threshold:
        return False
    return (d_a > 0) != (d_b > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BOTTLENECK = 8
    N_SEEDS = 3 if args.quick else 10
    EPOCHS = 100 if args.quick else 200
    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768][:N_SEEDS]

    print("=" * 60)
    print("Learned Representation Polarity Analysis v2 (all 8 datasets)")
    print(f"Bottleneck: {BOTTLENECK}, Seeds: {N_SEEDS}, Epochs: {EPOCHS}")
    print("=" * 60)

    # Load all domains
    print("\nLoading all domains...")
    domain_data = load_all_domains(seeds=seeds)
    loaded_domains = sorted(domain_data.keys())
    print(f"Loaded {len(loaded_domains)} domains: {loaded_domains}")

    # Per-seed analysis
    all_seed_results: list[dict] = []
    all_seed_d_matrices: list[dict[str, dict[str, float]]] = []

    for seed in seeds:
        print(f"\n{'='*40} Seed {seed} {'='*40}")
        set_seeds(seed)

        # Build heuristic feature matrix for synthetic normal traces (training data)
        syn_features, syn_labels = domain_data["synthetic"]
        normal_mask = syn_labels == 0
        X_syn_normal = np.column_stack(
            [syn_features[f][normal_mask] for f in HEURISTIC_FEATURES]
        )

        scaler = StandardScaler()
        scaler.fit(X_syn_normal)

        # Train autoencoder on synthetic normal traces
        print("  Training autoencoder on synthetic normal traces...")
        model = train_autoencoder(
            scaler.transform(X_syn_normal),
            epochs=EPOCHS,
            seed=seed,
            bottleneck=BOTTLENECK,
        )

        # Extract bottleneck features for all domains
        seed_result: dict = {"seed": seed, "datasets": {}}
        seed_d_matrix: dict[str, dict[str, float]] = {}

        for domain_name in loaded_domains:
            features, labels = domain_data[domain_name]
            X_heuristic = np.column_stack(
                [features[f] for f in HEURISTIC_FEATURES]
            )
            X_scaled = scaler.transform(X_heuristic)
            Z = extract_bottleneck(model, X_scaled)

            normal_m = labels == 0
            anom_m = labels == 1
            if anom_m.sum() < 2 or normal_m.sum() < 2:
                print(f"  {domain_name}: insufficient labels, skipping")
                continue

            dims_result: dict = {}
            d_per_dim: dict[str, float] = {}

            for dim in range(BOTTLENECK):
                d = cohens_d(Z[anom_m, dim], Z[normal_m, dim])
                dim_key = f"z{dim}"
                dims_result[dim_key] = {
                    "d": float(d),
                    "sign": "+" if d > 0 else "-" if d < 0 else "0",
                }
                d_per_dim[dim_key] = float(d)

            seed_result["datasets"][domain_name] = {
                "n_traces": len(labels),
                "n_normal": int(normal_m.sum()),
                "n_anomalous": int(anom_m.sum()),
                "dims": dims_result,
            }
            seed_d_matrix[domain_name] = d_per_dim

            # Print summary for this domain
            d_vals = [d_per_dim[f"z{i}"] for i in range(BOTTLENECK)]
            print(
                f"  {domain_name:>12}: "
                f"d = [{', '.join(f'{v:+.2f}' for v in d_vals)}]"
            )

        all_seed_results.append(seed_result)
        all_seed_d_matrices.append(seed_d_matrix)

    # ---- Aggregate: reversal analysis across all domain pairs ----
    print("\n" + "=" * 60)
    print("REVERSAL ANALYSIS: Within-Group vs Between-Group")
    print("=" * 60)

    domains_analyzed = sorted(all_seed_d_matrices[0].keys())
    domain_pairs = list(combinations(domains_analyzed, 2))

    within_reversals = 0
    within_total = 0
    between_reversals = 0
    between_total = 0

    pair_results: list[dict] = []

    for d1, d2 in domain_pairs:
        is_within = same_group(d1, d2)
        pair_type = "within" if is_within else "between"

        dim_reversals = 0
        dim_total = 0

        for dim in range(BOTTLENECK):
            dim_key = f"z{dim}"
            # Average d across seeds for stability
            d1_vals = [m[d1][dim_key] for m in all_seed_d_matrices if d1 in m]
            d2_vals = [m[d2][dim_key] for m in all_seed_d_matrices if d2 in m]
            d1_mean = np.mean(d1_vals)
            d2_mean = np.mean(d2_vals)

            if check_sign_reversal(d1_mean, d2_mean):
                dim_reversals += 1
            dim_total += 1

        reversal_rate = dim_reversals / max(dim_total, 1)

        if is_within:
            within_reversals += dim_reversals
            within_total += dim_total
        else:
            between_reversals += dim_reversals
            between_total += dim_total

        pair_results.append({
            "pair": f"{d1}-{d2}",
            "type": pair_type,
            "group_d1": get_group_for_domain(d1),
            "group_d2": get_group_for_domain(d2),
            "dim_reversals": dim_reversals,
            "dim_total": dim_total,
            "reversal_rate": reversal_rate,
        })

    within_rate = within_reversals / max(within_total, 1)
    between_rate = between_reversals / max(between_total, 1)

    print(f"\nWithin-group reversal:  {within_reversals}/{within_total} = {within_rate:.1%}")
    print(f"Between-group reversal: {between_reversals}/{between_total} = {between_rate:.1%}")
    if within_rate > 0:
        print(f"Ratio (between/within): {between_rate / within_rate:.1f}x")
    else:
        print("Ratio: inf (0% within-group)")

    print(f"\nPer-pair breakdown:")
    print(f"{'Pair':<30} {'Type':<10} {'Reversals':<12} {'Rate':<8}")
    print("-" * 60)
    for pr in sorted(pair_results, key=lambda x: (x["type"], x["pair"])):
        print(
            f"{pr['pair']:<30} {pr['type']:<10} "
            f"{pr['dim_reversals']}/{pr['dim_total']:<10} "
            f"{pr['reversal_rate']:.1%}"
        )

    # ---- Compute mean d matrix (averaged across seeds) for the paper ----
    mean_d_matrix: dict[str, dict[str, float]] = {}
    for domain in domains_analyzed:
        mean_d_matrix[domain] = {}
        for dim in range(BOTTLENECK):
            dim_key = f"z{dim}"
            vals = [m[domain][dim_key] for m in all_seed_d_matrices if domain in m]
            mean_d_matrix[domain][dim_key] = float(np.mean(vals))

    elapsed = time.time() - t0

    results = {
        "experiment": "v30_learned_polarity_v2",
        "bottleneck": BOTTLENECK,
        "n_seeds": N_SEEDS,
        "epochs": EPOCHS,
        "domains_analyzed": domains_analyzed,
        "semantic_groups": SEMANTIC_GROUPS,
        "within_group_reversal": {
            "reversals": within_reversals,
            "total": within_total,
            "rate": within_rate,
        },
        "between_group_reversal": {
            "reversals": between_reversals,
            "total": between_total,
            "rate": between_rate,
        },
        "pair_results": pair_results,
        "mean_d_matrix": mean_d_matrix,
        "per_seed": all_seed_results,
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "v30_learned_polarity_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")

    # Key result for the paper
    print("\n" + "=" * 60)
    print("KEY RESULT FOR PAPER:")
    print(f"  Learned features ({BOTTLENECK}-dim autoencoder bottleneck)")
    print(f"  Within-group reversal:  {within_rate:.1%}")
    print(f"  Between-group reversal: {between_rate:.1%}")
    if within_rate > 0:
        print(f"  Gap: {between_rate / within_rate:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
