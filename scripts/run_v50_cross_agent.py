"""E1: same tasks, different agent. Does feature polarity flip across AgentDojo backbones?

Registered in prereg-v50 and the proposal:
- attack `important_instructions`; label 1 = injection succeeded
- length-free features only, because a successful attack adds tool calls by construction
- halves split by (user task, injection task), so no task is on both sides
- class balance matched by subsampling each backbone to 50/50
- backbones need >= 30 positives and >= 30 negatives
Refutation of "agent-driven reversal": cross-backbone flip rate <= within-backbone split-half rate.
Trace length is reported separately as the construction-tied reference.
"""

import itertools
import json
import zlib
from pathlib import Path

import numpy as np

from data import domain_loader as dl
from data.trace_loader import load_agentdojo
from traceshift.v50 import cohens_d_anomalous, is_reversed, transfer_auroc

RUNS = Path("/tmp/agentdojo_data/runs")
OUT = Path("output/tables/v50_cross_agent.json")
FEATURES = ("low_diversity", "repeat_rate", "mean_arg_length", "arg_diversity", "arg_empty_rate")
SEEDS = range(10)
TAU = 0.1
MIN_CLASS = 30


def _corpus(run: str) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    traces = load_agentdojo(RUNS / run, min_tool_calls=1)
    y = np.array([int(t.has_anomaly) for t in traces])
    if min(int(y.sum()), int(len(y) - y.sum())) < MIN_CLASS:
        return None
    feats = dl._extract_all_features(traces)
    x = np.column_stack([np.asarray(feats[f], float) for f in (*FEATURES, "trace_length")])
    return x, y, [t.task for t in traces]


def _task_half(tasks: list[str], seed: int) -> np.ndarray:
    return np.array([zlib.crc32(f"{seed}:{t}".encode()) % 2 for t in tasks])


def _balanced(y: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    n = min(len(pos), len(neg))
    return np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])


def _d(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.array([cohens_d_anomalous(x[idx, j], y[idx]) for j in range(x.shape[1])])


def main() -> None:
    corpora = {r: c for r in sorted(p.name for p in RUNS.iterdir()) if (c := _corpus(r))}
    print("backbones", len(corpora), flush=True)
    cross_flips, noise_flips, cross_len, aurocs = [], [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        halves = {}
        for r, (x, y, tasks) in corpora.items():
            h = _task_half(tasks, seed)
            halves[r] = {side: _balanced(y, np.flatnonzero(h == side), rng) for side in (0, 1)}
        d = {(r, s): _d(corpora[r][0], corpora[r][1], halves[r][s]) for r in corpora for s in (0, 1)}
        for r in corpora:
            noise_flips += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(r, 0)][:5], d[(r, 1)][:5])]
        for r1, r2 in itertools.combinations(corpora, 2):
            cross_flips += [is_reversed(a, b, tau=TAU) for a, b in zip(d[(r1, 0)][:5], d[(r2, 1)][:5])]
            cross_len.append(is_reversed(d[(r1, 0)][5], d[(r2, 1)][5], tau=TAU))
        if seed == 0:
            for r1, r2 in itertools.permutations(corpora, 2):
                (x1, y1, _), (x2, y2, _) = corpora[r1], corpora[r2]
                a, b = halves[r1][0], halves[r2][1]
                aurocs.append(transfer_auroc((x1[a][:, :5], y1[a]), (x2[b][:, :5], y2[b]),
                                             detector="rf", seed=0))
    cross, noise = float(np.mean(cross_flips)), float(np.mean(noise_flips))
    result = {
        "n_backbones": len(corpora), "backbones": list(corpora),
        "cross_backbone_flip_rate": cross, "within_backbone_split_half_flip_rate": noise,
        "agent_driven_reversal_refuted": cross <= noise,
        "trace_length_cross_flip_rate": float(np.mean(cross_len)),
        "rf_length_free_transfer": {"mean_auroc": float(np.mean(aurocs)),
                                    "anti_prediction_rate": float(np.mean(np.array(aurocs) < 0.5)),
                                    "n_pairs": len(aurocs)},
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
