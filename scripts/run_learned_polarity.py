"""Experiment 3: Polarity reversal on learned representations.

Trains an autoencoder on synthetic normal traces, extracts bottleneck
features, computes Cohen's d on each bottleneck dimension across all
datasets. Shows polarity reversal persists in learned representations,
not just handcrafted features.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_learned_polarity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from research_utils.reproducibility import set_seeds

OUTPUT_DIR = Path("output/tables")
MIN_TRACE_LEN = 3
MAX_TRACE_LEN = 200


def compute_features(traces: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute heuristic features and labels."""
    n = len(traces)
    X = np.zeros((n, 3))
    y = np.zeros(n, dtype=np.int32)
    for i, t in enumerate(traces):
        actions = t["actions"]
        n_act = len(actions)
        X[i, 0] = float(n_act)
        unique = len(set(actions))
        X[i, 1] = 1.0 - (unique / max(n_act, 1))
        if n_act > 1:
            repeats = sum(
                1 for j in range(1, n_act) if actions[j] == actions[j - 1]
            )
            X[i, 2] = repeats / (n_act - 1)
        y[i] = t["label"]
    return X, y


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled < 1e-10:
        return float("nan")
    return float((group_a.mean() - group_b.mean()) / pooled)


class DeepAutoEncoder(nn.Module):
    """Deeper autoencoder with 8-dim bottleneck for learned representation analysis."""

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
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_deep_ae(
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
    for epoch in range(epochs):
        x_hat = model(X_t)
        loss = ((X_t - x_hat) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = float(loss.item())
    print(f"    AE training done: final reconstruction loss = {final_loss:.6f}")
    return model


def extract_bottleneck(model: DeepAutoEncoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        z = model.encode(X_t).numpy()
    return z


# --- Data loading ---

def load_swe_agent_data(max_traces: int = 5000) -> list[dict]:
    import re
    from datasets import load_dataset

    print(f"Loading up to {max_traces} SWE-agent trajectories...")
    ds = load_dataset(
        "nebius/SWE-agent-trajectories", split="train", streaming=True,
    )
    traces = []
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions = []
        for turn in example["trajectory"]:
            if turn.get("role") == "ai" and turn.get("text"):
                code_blocks = re.findall(
                    r"```\s*\n?(.*?)\n?```", turn["text"], re.DOTALL,
                )
                for block in code_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    first_line = block.split("\n")[0].strip()
                    cmd = first_line.split()[0] if first_line.split() else ""
                    cmd = cmd.lstrip("./!$")
                    if cmd:
                        actions.append(cmd)
        if len(actions) < MIN_TRACE_LEN:
            continue
        if len(actions) > MAX_TRACE_LEN:
            actions = actions[:MAX_TRACE_LEN]
        traces.append({
            "actions": actions,
            "label": 0 if bool(example["target"]) else 1,
            "n_actions": len(actions),
        })
    solved = sum(1 for t in traces if t["label"] == 0)
    print(f"Loaded {len(traces)} traces ({solved} solved)")
    return traces


def load_atbench_traces() -> list[dict]:
    from data.trace_loader import load_atbench

    print("Loading ATBench...")
    traces = load_atbench(max_traces=None)
    result = []
    for t in traces:
        actions = [a.tool for a in t.actions]
        if len(actions) < 2:
            continue
        result.append({
            "actions": actions[:50],
            "label": 1 if t.has_anomaly else 0,
            "n_actions": min(len(actions), 50),
        })
    safe = sum(1 for r in result if r["label"] == 0)
    print(f"Loaded {len(result)} traces ({safe} safe)")
    return result


def load_synthetic_traces() -> list[dict]:
    import orjson

    result = []
    for _, path in [("train", "data/train.jsonl"), ("test", "data/test.jsonl")]:
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "rb") as f:
            for line in f:
                obj = orjson.loads(line)
                raw_actions = obj.get("tools", obj.get("actions", []))
                actions = [
                    a["tool"] if isinstance(a, dict) else str(a)
                    for a in raw_actions
                ]
                if len(actions) >= MIN_TRACE_LEN:
                    label = obj.get("label", obj.get("anomaly", None))
                    if label is None:
                        label = int(any(
                            a.get("label", 0) == 1
                            for a in raw_actions
                            if isinstance(a, dict)
                        ))
                    result.append({
                        "actions": actions[:MAX_TRACE_LEN],
                        "label": int(label),
                        "n_actions": min(len(actions), MAX_TRACE_LEN),
                    })
    normal = sum(1 for r in result if r["label"] == 0)
    print(f"Loaded {len(result)} synthetic ({normal} normal)")
    return result


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BOTTLENECK = 8
    N_SEEDS = 5
    EPOCHS = 200
    seeds = [42, 123, 456, 789, 1024][:N_SEEDS]

    print("=" * 60)
    print(f"Learned Representation Polarity Analysis (v30)")
    print(f"Bottleneck dims: {BOTTLENECK}, Seeds: {N_SEEDS}, Epochs: {EPOCHS}")
    print("=" * 60)

    synthetic = load_synthetic_traces()
    swe_traces = load_swe_agent_data(max_traces=5000)
    atbench = load_atbench_traces()

    datasets = {
        "Synthetic": synthetic,
        "SWE-agent": swe_traces,
        "ATBench": atbench,
    }

    all_seed_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seeds(seed)

        X_syn, y_syn = compute_features(synthetic)
        normal_mask = y_syn == 0
        X_syn_normal = X_syn[normal_mask]

        scaler = StandardScaler()
        scaler.fit(X_syn_normal)

        print("  Training deep autoencoder on synthetic normal traces...")
        model = train_deep_ae(
            scaler.transform(X_syn_normal),
            epochs=EPOCHS,
            seed=seed,
            bottleneck=BOTTLENECK,
        )

        seed_result = {"seed": seed, "datasets": {}}

        for ds_name, traces in datasets.items():
            X, y = compute_features(traces)
            X_scaled = scaler.transform(X)

            Z = extract_bottleneck(model, X_scaled)

            normal_mask_ds = y == 0
            anom_mask_ds = y == 1

            if anom_mask_ds.sum() < 2 or normal_mask_ds.sum() < 2:
                print(f"  {ds_name}: insufficient labels, skipping")
                continue

            ds_result = {"n_traces": len(y), "dims": {}}
            print(f"\n  {ds_name} ({len(y)} traces):")
            print(f"    {'Dim':>6} | {'d':>8} | {'sign':>6} | {'|d|':>6}")
            print("    " + "-" * 40)

            for dim in range(BOTTLENECK):
                d = cohens_d(Z[anom_mask_ds, dim], Z[normal_mask_ds, dim])
                sign_str = "+" if d > 0 else "-" if d < 0 else "0"
                ds_result["dims"][f"z{dim}"] = {"d": d, "sign": sign_str}
                print(f"    z{dim:>5} | {d:>+8.3f} | {sign_str:>6} | {abs(d):>6.3f}")

            seed_result["datasets"][ds_name] = ds_result

        all_seed_results.append(seed_result)

    # --- Aggregate: count sign reversals across dims and seeds ---
    print("\n" + "=" * 60)
    print("SIGN REVERSAL SUMMARY (Synthetic → SWE-agent)")
    print("=" * 60)

    reversal_counts = np.zeros(BOTTLENECK)
    total_valid = np.zeros(BOTTLENECK)

    for sr in all_seed_results:
        syn_dims = sr["datasets"].get("Synthetic", {}).get("dims", {})
        swe_dims = sr["datasets"].get("SWE-agent", {}).get("dims", {})
        for dim in range(BOTTLENECK):
            key = f"z{dim}"
            if key in syn_dims and key in swe_dims:
                d_syn = syn_dims[key]["d"]
                d_swe = swe_dims[key]["d"]
                if not np.isnan(d_syn) and not np.isnan(d_swe):
                    total_valid[dim] += 1
                    if np.sign(d_syn) != np.sign(d_swe) and d_syn != 0 and d_swe != 0:
                        reversal_counts[dim] += 1

    print(f"{'Dim':>6} | {'Reversals':>10} | {'Total':>6} | {'Rate':>6}")
    print("-" * 40)
    n_reversed_dims = 0
    for dim in range(BOTTLENECK):
        rate = reversal_counts[dim] / max(total_valid[dim], 1)
        print(f"z{dim:>5} | {int(reversal_counts[dim]):>10} | {int(total_valid[dim]):>6} | {rate:>6.1%}")
        if rate > 0.5:
            n_reversed_dims += 1

    print(f"\nDimensions with >50% reversal rate: {n_reversed_dims}/{BOTTLENECK}")
    total_reversals = int(reversal_counts.sum())
    total_comparisons = int(total_valid.sum())
    overall_rate = total_reversals / max(total_comparisons, 1)
    print(f"Overall reversal rate: {total_reversals}/{total_comparisons} = {overall_rate:.1%}")

    # --- Same for ATBench ---
    print("\n" + "=" * 60)
    print("SIGN REVERSAL SUMMARY (Synthetic → ATBench)")
    print("=" * 60)

    reversal_counts_atb = np.zeros(BOTTLENECK)
    total_valid_atb = np.zeros(BOTTLENECK)

    for sr in all_seed_results:
        syn_dims = sr["datasets"].get("Synthetic", {}).get("dims", {})
        atb_dims = sr["datasets"].get("ATBench", {}).get("dims", {})
        for dim in range(BOTTLENECK):
            key = f"z{dim}"
            if key in syn_dims and key in atb_dims:
                d_syn = syn_dims[key]["d"]
                d_atb = atb_dims[key]["d"]
                if not np.isnan(d_syn) and not np.isnan(d_atb):
                    total_valid_atb[dim] += 1
                    if np.sign(d_syn) != np.sign(d_atb) and d_syn != 0 and d_atb != 0:
                        reversal_counts_atb[dim] += 1

    print(f"{'Dim':>6} | {'Reversals':>10} | {'Total':>6} | {'Rate':>6}")
    print("-" * 40)
    n_reversed_atb = 0
    for dim in range(BOTTLENECK):
        rate = reversal_counts_atb[dim] / max(total_valid_atb[dim], 1)
        print(f"z{dim:>5} | {int(reversal_counts_atb[dim]):>10} | {int(total_valid_atb[dim]):>6} | {rate:>6.1%}")
        if rate > 0.5:
            n_reversed_atb += 1

    print(f"\nDimensions with >50% reversal rate: {n_reversed_atb}/{BOTTLENECK}")

    elapsed = time.time() - t0
    results = {
        "bottleneck": BOTTLENECK,
        "n_seeds": N_SEEDS,
        "epochs": EPOCHS,
        "per_seed": all_seed_results,
        "syn_swe_reversal_rate": overall_rate,
        "syn_swe_reversed_dims": n_reversed_dims,
        "syn_atb_reversed_dims": n_reversed_atb,
        "elapsed_seconds": elapsed,
    }

    out_path = OUTPUT_DIR / "learned_polarity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
