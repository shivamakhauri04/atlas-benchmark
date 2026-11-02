"""E1 on learned features instead of hand-made statistics.

Paper review A4; exploratory, prereg A9.

Each trace becomes the text of its tool calls ("tool(args) | tool(args) ..."), embedded with
all-MiniLM-L6-v2. A PCA fitted on the pooled traces of all configurations, without labels,
reduces the
embeddings to five components, matching the five length-free statistics in E1. The reversal test is
then E1's: task-disjoint halves, class-balanced, ten splits, thresholds 0.1/0.2/0.3.

The label ties trace length by construction, and an embedding of the call text can carry length. So
the test also runs on components with log length partialled out within each configuration.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import run_v50_cross_agent as e1
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from data.trace_loader import load_agentdojo
from traceshift.v50 import is_reversed

OUT = Path("output/tables/v50_cross_agent_embed.json")
TAUS = (0.1, 0.2, 0.3)
N_COMPONENTS = 5
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _text(trace) -> str:
    return " | ".join(f"{a.tool}({a.args[:200]})" for a in trace.actions)[:2000]


def _residualize(x: np.ndarray, length: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones_like(length), np.log1p(length)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def _rates(corpora: dict) -> dict:
    pairs = list(itertools.combinations(corpora, 2))
    cross = {t: [] for t in TAUS}
    within = {t: [] for t in TAUS}
    for seed in e1.SEEDS:
        rng = np.random.default_rng(seed)
        halves = {}
        for r, (_, y, tasks) in corpora.items():
            h = e1._task_half(tasks, seed)
            halves[r] = {s: e1._balanced(y, np.flatnonzero(h == s), rng) for s in (0, 1)}
        d = {
            (r, s): e1._d(corpora[r][0], corpora[r][1], halves[r][s])
            for r in corpora
            for s in (0, 1)
        }
        for t in TAUS:
            within[t] += [
                is_reversed(a, b, tau=t) for r in corpora for a, b in zip(d[(r, 0)], d[(r, 1)])
            ]
            cross[t] += [
                is_reversed(a, b, tau=t) for p in pairs for a, b in zip(d[(p[0], 0)], d[(p[1], 1)])
            ]
    return {
        f"tau={t}": {"cross": float(np.mean(cross[t])), "within": float(np.mean(within[t]))}
        for t in TAUS
    }


def main() -> None:
    raw = {}
    for run in sorted(p.name for p in e1.RUNS.iterdir()):
        traces = load_agentdojo(e1.RUNS / run, min_tool_calls=1)
        y = np.array([int(t.has_anomaly) for t in traces])
        if len(y) and min(int(y.sum()), int(len(y) - y.sum())) >= e1.MIN_CLASS:
            raw[run] = (traces, y)
    model = SentenceTransformer(MODEL)
    emb = {
        r: model.encode([_text(t) for t in tr], batch_size=64, show_progress_bar=False)
        for r, (tr, _) in raw.items()
    }
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(np.vstack(list(emb.values())))
    comps = {r: pca.transform(emb[r]) for r in raw}
    lengths = {r: np.array([len(t.actions) for t in raw[r][0]], float) for r in raw}
    pooled_c = np.vstack(list(comps.values()))
    pooled_l = np.log1p(np.concatenate(list(lengths.values())))
    tasks = {r: [t.task for t in raw[r][0]] for r in raw}
    out = {
        "model": MODEL,
        "n_components": N_COMPONENTS,
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "n_configurations": len(raw),
        "corr_with_log_length": [
            float(np.corrcoef(pooled_c[:, j], pooled_l)[0, 1]) for j in range(N_COMPONENTS)
        ],
    }
    out.update(_rates({r: (comps[r], raw[r][1], tasks[r]) for r in raw}))
    out["length_partialled"] = _rates(
        {r: (_residualize(comps[r], lengths[r]), raw[r][1], tasks[r]) for r in raw}
    )
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
