"""A trained sequence monitor across AgentDojo configurations (prereg-v50 A14, descriptive).

A one-layer GRU reads each run as a sequence of (tool name, argument-length bucket) tokens. For each
source configuration and seed it is trained on the source's task half A (20% held out for early
stopping) and scored by AUROC on every target's task half B (transfer) and on the source's own half
B (within). A random forest on the six features uses the same halves. Seed-0 task split.
"""

import json
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch import nn

from data import domain_loader as dl
from data.trace_loader import load_agentdojo

OUT = Path("output/tables/v50_gru_monitor.json")
SEEDS = range(5)
BUCKETS = (0, 10, 50, 200)
MAX_LEN = 40
EPOCHS = 60
PATIENCE = 8


class GRUMonitor(nn.Module):
    def __init__(self, n_tools: int) -> None:
        super().__init__()
        self.tool = nn.Embedding(n_tools + 1, 32, padding_idx=0)
        self.bucket = nn.Embedding(len(BUCKETS) + 2, 32, padding_idx=0)
        self.gru = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, tools: torch.Tensor, buckets: torch.Tensor, lengths: torch.Tensor):
        h = self.tool(tools) + self.bucket(buckets)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, last = self.gru(packed)
        return self.head(last[-1]).squeeze(-1)


def _encode(traces, vocab: dict[str, int]):
    n = len(traces)
    tools = np.zeros((n, MAX_LEN), dtype=np.int64)
    buckets = np.zeros((n, MAX_LEN), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    for i, t in enumerate(traces):
        acts = t.actions[:MAX_LEN]
        lengths[i] = len(acts)
        for j, a in enumerate(acts):
            tools[i, j] = vocab.get(a.tool, 0)
            buckets[i, j] = 1 + int(np.searchsorted(BUCKETS, len(a.args), side="right"))
    return torch.tensor(tools), torch.tensor(buckets), torch.tensor(lengths)


def _fit(enc, y: np.ndarray, idx: np.ndarray, n_tools: int, seed: int) -> GRUMonitor:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    n_val = max(4, len(idx) // 5)
    val, tr = idx[:n_val], idx[n_val:]
    model = GRUMonitor(n_tools)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    pos = max(1, int(y[tr].sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(tr) - pos) / pos))
    yt = torch.tensor(y, dtype=torch.float32)
    best, best_state, wait = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(*(e[tr] for e in enc)), yt[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            s = model(*(e[val] for e in enc)).numpy()
        auc = roc_auc_score(y[val], s) if 0 < y[val].sum() < len(val) else 0.5
        if auc > best:
            best, wait = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval()


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def main() -> None:
    raw = {}
    for run in sorted(p.name for p in e1.RUNS.iterdir() if p.is_dir()):
        traces = load_agentdojo(e1.RUNS / run, min_tool_calls=1)
        y = np.array([int(t.has_anomaly) for t in traces])
        if len(y) and min(int(y.sum()), int(len(y) - y.sum())) >= e1.MIN_CLASS:
            raw[run] = (traces, y)
    vocab = {
        t: i + 1
        for i, t in enumerate(
            sorted({a.tool for tr, _ in raw.values() for t in tr for a in t.actions})
        )
    }
    enc = {r: _encode(tr, vocab) for r, (tr, _) in raw.items()}
    feats = {
        r: np.column_stack([np.asarray(f[k], float) for k in (*e1.FEATURES, "trace_length")])
        for r, (tr, _) in raw.items()
        for f in [dl._extract_all_features(tr)]
    }
    half = {r: e1._task_half([t.task for t in tr], 0) for r, (tr, _) in raw.items()}
    a_idx = {r: np.flatnonzero(half[r] == 0) for r in raw}
    b_idx = {r: np.flatnonzero(half[r] == 1) for r in raw}
    gru = {"within": [], "transfer": []}
    rf = {"within": [], "transfer": []}
    for src in raw:
        ys = raw[src][1]
        scores_gru = {t: [] for t in raw}
        scores_rf = {t: [] for t in raw}
        for seed in SEEDS:
            model = _fit(enc[src], ys, a_idx[src], len(vocab), seed)
            forest = RandomForestClassifier(
                n_estimators=50, max_depth=3, random_state=seed, class_weight="balanced"
            )
            forest.fit(feats[src][a_idx[src]], ys[a_idx[src]])
            with torch.no_grad():
                for tgt in raw:
                    b = b_idx[tgt]
                    scores_gru[tgt].append(model(*(e[b] for e in enc[tgt])).numpy())
                    scores_rf[tgt].append(forest.predict_proba(feats[tgt][b])[:, 1])
        for tgt in raw:
            yb = raw[tgt][1][b_idx[tgt]]
            key = "within" if tgt == src else "transfer"
            gru[key].append(_auc(yb, np.mean(scores_gru[tgt], axis=0)))
            rf[key].append(_auc(yb, np.mean(scores_rf[tgt], axis=0)))
        print(src, round(gru["within"][-1], 3), round(rf["within"][-1], 3), flush=True)

    def summary(v: dict) -> dict:
        tr = np.array(v["transfer"])
        return {
            "within_mean_auroc": float(np.mean(v["within"])),
            "transfer_mean_auroc": float(tr.mean()),
            "transfer_below_0.5": float((tr < 0.5).mean()),
            "n_pairs": int(len(tr)),
        }

    out = {
        "split": "seed-0 task halves",
        "seeds": len(SEEDS),
        "n_configurations": len(raw),
        "gru": summary(gru),
        "random_forest_all_six": summary(rf),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
