"""Score the pre-registered label-free sign predictions (prereg-v50 A1-A3).

Primary metric: `sign_prediction_accuracy` per stratum, over construction-tied cells with |d| > tau,
against `pooled_sign_baseline` (each cell's sign predicted from the mean d of every other dataset).
Observed d is the mean over seeds 0-9 of Cohen's d on stratified half A. The predictions file was
committed in fa9840c, before this script existed.
"""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

from data import domain_loader as dl
from data.agentdrift_loader import load_agentdrift
from data.trace_loader import load_agentdojo, load_stabletoolbench, load_trail
from traceshift.v50 import cohens_d_anomalous, stratified_halves

PRED = Path("inputs/prereg-v50-sign-predictions.csv")
OUT = Path("output/tables/v50_decomposition.json")
AGENTDOJO = Path("/tmp/agentdojo_data/runs")
STB = Path("/tmp/stabletoolbench_baselines/data_baselines")
AGENTDRIFT = Path("/tmp/agentdrift")
SEEDS = range(10)
TAUS = (0.1, 0.2)
MIN_POSITIVES = 30
FEATURES = (
    "trace_length",
    "low_diversity",
    "repeat_rate",
    "mean_arg_length",
    "arg_diversity",
    "arg_empty_rate",
)
PRIMARY_SUB = {
    "agentdojo": "gpt-4o-2024-05-13",
    "stabletoolbench": "chatgpt_cot",
    "trail": "all",
    "agentdrift": "full_hijack/email",
}

Corpus = tuple[dict[str, np.ndarray], np.ndarray]


def _from_traces(traces: list) -> Corpus:
    labels = np.array([int(t.has_anomaly) for t in traces])
    return dl._extract_all_features(traces), labels


def _trail(split: str) -> Corpus:
    splits = None if split == "all" else {"gaia": ["GAIA"], "swe": ["SWE Bench"]}[split]
    traces = load_trail(Path("data/trail/TRAIL-Benchmark"), splits=splits)
    feats = dl._extract_all_features(traces)
    err = np.array([sum(a.label == 1 for a in t.actions) / max(len(t.actions), 1) for t in traces])
    return feats, (err > 0.3).astype(int)


def _loaders() -> dict[tuple[str, str], Callable[[], Corpus]]:
    out: dict[tuple[str, str], Callable[[], Corpus]] = {}
    for run in sorted(p.name for p in AGENTDOJO.iterdir()):
        out[("agentdojo", run)] = lambda r=run: _from_traces(
            load_agentdojo(AGENTDOJO / r, min_tool_calls=1)
        )
    for model in sorted(p.name for p in STB.iterdir()):
        out[("stabletoolbench", model)] = lambda m=model: _from_traces(
            load_stabletoolbench(STB / m, model=m)
        )
    for split in ("all", "gaia", "swe"):
        out[("trail", split)] = lambda s=split: _trail(s)
    for cat in ("full_hijack", "partial_hijack", "delayed_execution", "failed_attack"):
        for dom in ("email", "banking", "web", "coding", "medical"):
            out[("agentdrift", f"{cat}/{dom}")] = lambda c=cat, d=dom: _from_traces(
                load_agentdrift(AGENTDRIFT, category=c, domain=d)
            )
    out[("swe_agent", "")] = dl.load_swe_agent_domain
    out[("synthetic", "")] = dl.load_synthetic_domain
    out[("toolbench", "")] = dl.load_toolbench_domain
    out[("toolace", "")] = dl.load_toolace_domain
    out[("apibank", "")] = dl.load_apibank_domain
    out[("injecagent", "")] = dl.load_injecagent_domain
    out[("agentharm", "")] = dl.load_agentharm_domain
    out[("atbench", "")] = dl.load_atbench_domain
    out[("rjudge", "")] = dl.load_rjudge_domain
    return out


def half_a_d(feats: dict[str, np.ndarray], y: np.ndarray) -> dict[str, float]:
    """Mean over seeds of Cohen's d (anomalous minus normal) on stratified half A."""
    out: dict[str, float] = {}
    for f in FEATURES:
        x = np.asarray(feats[f], dtype=float)
        out[f] = float(
            np.mean(
                [
                    cohens_d_anomalous(x[a], y[a])
                    for a, _ in (stratified_halves(y, seed=s) for s in SEEDS)
                ]
            )
        )
    return out


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    lo = float(beta.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return lo, hi


EXTERNAL_BAR = 0.60


def score(
    preds: list[dict[str, str]],
    d: dict[tuple[str, str], dict[str, float]],
    stratum: str,
    tau: float,
) -> dict[str, object]:
    # Reason: the baseline is the pool's own "same meaning everywhere" null, so it pools only the
    # 12 datasets of the earlier benchmark; the external AgentDrift corpora never enter it.
    primary = {
        ds: d[(ds, PRIMARY_SUB.get(ds, ""))]
        for ds, _ in d
        if (ds, PRIMARY_SUB.get(ds, "")) in d and ds != "agentdrift"
    }
    hits = base_hits = 0
    cells: list[dict[str, object]] = []
    for p in preds:
        key = (
            p["dataset"],
            p["sub_corpus"] if p["dataset"] != "trail" or p["sub_corpus"] else "all",
        )
        if p["stratum"] != stratum or int(p["predicted_sign"]) == 0 or key not in d:
            continue
        obs = d[key][p["feature"]]
        if abs(obs) <= tau:
            continue
        others = [v[p["feature"]] for ds, v in primary.items() if ds != p["dataset"]]
        base = int(np.sign(np.mean(others)))
        hit, bhit = int(np.sign(obs)) == int(p["predicted_sign"]), int(np.sign(obs)) == base
        hits, base_hits = hits + hit, base_hits + bhit
        cells.append(
            {
                "dataset": key[0],
                "sub": key[1],
                "feature": p["feature"],
                "d": obs,
                "predicted": int(p["predicted_sign"]),
                "baseline": base,
                "hit": hit,
            }
        )
    n = len(cells)
    if n == 0:
        return {"n_cells": 0}
    acc, bacc = hits / n, base_hits / n
    return {
        "n_cells": n,
        "accuracy": acc,
        "accuracy_ci95": clopper_pearson(hits, n),
        "pooled_sign_baseline": bacc,
        # Reason: prereg A5 registered a fixed 60% bar for the external stratum (A8 corrects the
        # earlier script, which applied the executed-stratum rule to it).
        "rule": "accuracy > 0.60" if stratum == "ii_external" else "accuracy > baseline + 0.10",
        "refuted": acc <= EXTERNAL_BAR if stratum == "ii_external" else acc <= bacc + 0.10,
        "binom_p_vs_baseline_plus_10": float(
            binomtest(hits, n, min(bacc + 0.10, 0.999), alternative="greater").pvalue
        ),
        "cells": cells,
    }


def main() -> None:
    preds = list(csv.DictReader(PRED.open()))
    d: dict[tuple[str, str], dict[str, float]] = {}
    meta: dict[str, dict[str, int]] = {}
    for key, load in _loaders().items():
        try:
            feats, y = load()
        except Exception as exc:  # noqa: BLE001 — recorded, never silently dropped
            meta["/".join(key)] = {"error": 1}
            print(f"FAILED {key}: {exc}")
            continue
        n_pos = int(y.sum())
        meta["/".join(key)] = {"n": int(len(y)), "positives": n_pos}
        if (
            key[0] in ("agentdojo", "stabletoolbench", "trail", "agentdrift")
            and key[1] != PRIMARY_SUB[key[0]]
            and n_pos < MIN_POSITIVES
        ):
            continue
        d[key] = half_a_d(feats, y)
        print(key, len(y), n_pos)
    result = {
        "corpora": meta,
        "d_half_a": {"/".join(k): v for k, v in d.items()},
        "primary": {
            f"tau={t}": {
                s: score(preds, d, s, t) for s in ("i_executed", "ii_external", "iii_injected")
            }
            for t in TAUS
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: {
                    s: {
                        x: v[s].get(x)
                        for x in ("n_cells", "accuracy", "pooled_sign_baseline", "refuted")
                    }
                    for s in v
                }
                for k, v in result["primary"].items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
