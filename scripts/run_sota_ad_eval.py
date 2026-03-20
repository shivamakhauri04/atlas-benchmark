"""Experiment 1: SOTA AD baselines showing universal polarity reversal.

Trains Deep SVDD and Autoencoder on synthetic normal traces, scores all
4 datasets, computes Cohen's d on anomaly scores. Shows polarity reversal
is method-agnostic — not specific to our LSTM or heuristic features.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_sota_ad_eval.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from research_utils.reproducibility import set_seeds

OUTPUT_DIR = Path("output/tables")
MIN_TRACE_LEN = 3
MAX_TRACE_LEN = 200
FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]


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


def cohens_d(scores_anom: np.ndarray, scores_norm: np.ndarray) -> float:
    """Cohen's d: (mean_anom - mean_norm) / pooled_std."""
    n_a, n_b = len(scores_anom), len(scores_norm)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = scores_anom.var(ddof=1)
    var_b = scores_norm.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled < 1e-10:
        return float("nan")
    return float((scores_anom.mean() - scores_norm.mean()) / pooled)


# --- Deep SVDD ---

class DeepSVDDNet(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_deep_svdd(
    X_normal: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[DeepSVDDNet, np.ndarray]:
    """Train Deep SVDD on normal data. Returns (model, center)."""
    set_seeds(seed)
    model = DeepSVDDNet()
    X_t = torch.tensor(X_normal, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        center = model(X_t).mean(dim=0).numpy()
    center_t = torch.tensor(center, dtype=torch.float32)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        out = model(X_t)
        loss = ((out - center_t) ** 2).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, center


def score_deep_svdd(
    model: DeepSVDDNet, center: np.ndarray, X: np.ndarray,
) -> np.ndarray:
    """Score = distance to center."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    center_t = torch.tensor(center, dtype=torch.float32)
    with torch.no_grad():
        out = model(X_t)
        scores = ((out - center_t) ** 2).sum(dim=1).numpy()
    return scores


# --- Autoencoder ---

class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 16, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    X_normal: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> AutoEncoder:
    """Train autoencoder on normal data."""
    set_seeds(seed)
    model = AutoEncoder()
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


def score_autoencoder(model: AutoEncoder, X: np.ndarray) -> np.ndarray:
    """Score = reconstruction error (MSE per sample)."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        x_hat = model(X_t)
        scores = ((X_t - x_hat) ** 2).mean(dim=1).numpy()
    return scores


# --- Data loading (reuse from run_pafc_eval.py) ---

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


def evaluate_method(
    method_name: str,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute AUROC and Cohen's d for a method's scores."""
    normal_mask = labels == 0
    anom_mask = labels == 1
    n_pos = int(anom_mask.sum())
    n_neg = int(normal_mask.sum())
    if n_pos == 0 or n_neg == 0:
        return {"method": method_name, "auroc": float("nan"), "d": float("nan")}
    auroc = float(roc_auc_score(labels, scores))
    d = cohens_d(scores[anom_mask], scores[normal_mask])
    return {"method": method_name, "auroc": auroc, "d": d, "n_pos": n_pos, "n_neg": n_neg}


def main() -> None:
    parser = argparse.ArgumentParser(description="SOTA AD baselines polarity eval")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-swe-traces", type=int, default=5000)
    args = parser.parse_args()

    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random_seeds = [42, 123, 456, 789, 1024][:args.seeds]

    print("=" * 60)
    print("SOTA AD Baselines — Universal Polarity Reversal (v30)")
    print(f"Seeds: {len(random_seeds)}, Epochs: {args.epochs}")
    print("=" * 60)

    synthetic = load_synthetic_traces()
    swe_traces = load_swe_agent_data(max_traces=args.max_swe_traces)
    atbench = load_atbench_traces()

    datasets = {
        "Synthetic": synthetic,
        "SWE-agent": swe_traces,
        "ATBench": atbench,
    }

    all_results: dict = {"methods": {}, "polarity_table": []}

    for seed in random_seeds:
        print(f"\n--- Seed {seed} ---")
        set_seeds(seed)

        X_syn, y_syn = compute_features(synthetic)
        normal_mask = y_syn == 0
        X_syn_normal = X_syn[normal_mask]

        scaler = StandardScaler()
        scaler.fit(X_syn_normal)

        svdd_model, svdd_center = train_deep_svdd(
            scaler.transform(X_syn_normal), epochs=args.epochs, seed=seed,
        )
        ae_model = train_autoencoder(
            scaler.transform(X_syn_normal), epochs=args.epochs, seed=seed,
        )
        iso_forest = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=seed,
        )
        iso_forest.fit(scaler.transform(X_syn_normal))

        for ds_name, traces in datasets.items():
            X, y = compute_features(traces)
            X_scaled = scaler.transform(X)

            svdd_scores = score_deep_svdd(svdd_model, svdd_center, X_scaled)
            ae_scores = score_autoencoder(ae_model, X_scaled)
            if_scores = -iso_forest.score_samples(X_scaled)

            for method_name, scores in [
                ("Deep_SVDD", svdd_scores),
                ("Autoencoder", ae_scores),
                ("IsolationForest", if_scores),
            ]:
                result = evaluate_method(method_name, scores, y)
                result["dataset"] = ds_name
                result["seed"] = seed

                key = f"{method_name}_{ds_name}"
                if key not in all_results["methods"]:
                    all_results["methods"][key] = []
                all_results["methods"][key].append(result)

                print(
                    f"  {method_name:>16} on {ds_name:>10}: "
                    f"AUROC={result['auroc']:.3f}  d={result['d']:+.3f}"
                )

    print("\n" + "=" * 60)
    print("POLARITY REVERSAL TABLE (mean across seeds)")
    print("=" * 60)
    print(f"{'Method':>16} | {'Synthetic d':>12} | {'SWE-agent d':>12} | {'ATBench d':>12} | Reversal?")
    print("-" * 80)

    for method in ["Deep_SVDD", "Autoencoder", "IsolationForest"]:
        row = {"method": method}
        for ds_name in ["Synthetic", "SWE-agent", "ATBench"]:
            key = f"{method}_{ds_name}"
            entries = all_results["methods"].get(key, [])
            if entries:
                d_vals = [e["d"] for e in entries if not np.isnan(e["d"])]
                auroc_vals = [e["auroc"] for e in entries if not np.isnan(e["auroc"])]
                d_mean = float(np.mean(d_vals)) if d_vals else float("nan")
                auroc_mean = float(np.mean(auroc_vals)) if auroc_vals else float("nan")
                row[f"{ds_name}_d"] = d_mean
                row[f"{ds_name}_auroc"] = auroc_mean
            else:
                row[f"{ds_name}_d"] = float("nan")
                row[f"{ds_name}_auroc"] = float("nan")

        syn_d = row.get("Synthetic_d", 0)
        swe_d = row.get("SWE-agent_d", 0)
        atb_d = row.get("ATBench_d", 0)
        reversal = "YES" if (
            not np.isnan(syn_d) and not np.isnan(swe_d)
            and np.sign(syn_d) != np.sign(swe_d)
            and np.sign(syn_d) != 0 and np.sign(swe_d) != 0
        ) else "no"

        row["reversal_syn_swe"] = reversal
        all_results["polarity_table"].append(row)

        print(
            f"{method:>16} | {syn_d:>+12.3f} | {swe_d:>+12.3f} | "
            f"{atb_d:>+12.3f} | {reversal}"
        )

    print("\nAUROC Summary:")
    print(f"{'Method':>16} | {'Synthetic':>10} | {'SWE-agent':>10} | {'ATBench':>10}")
    print("-" * 60)
    for row in all_results["polarity_table"]:
        m = row["method"]
        print(
            f"{m:>16} | "
            f"{row.get('Synthetic_auroc', float('nan')):>10.3f} | "
            f"{row.get('SWE-agent_auroc', float('nan')):>10.3f} | "
            f"{row.get('ATBench_auroc', float('nan')):>10.3f}"
        )

    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = elapsed

    out_path = OUTPUT_DIR / "sota_ad_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
