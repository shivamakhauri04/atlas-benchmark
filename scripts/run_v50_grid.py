"""Corrected 12-domain transfer grid, provenance reversal ratio, and divergence regression (v50).

Analysis choices follow `docs/prereg-v50.md`:
- train on source half A, test on target half B, 10 seeds resampling both sides
- reversal of a (pair, feature) cell uses half-A d averaged over seeds, |d| > tau
- provenance comes from the committed predictions file; datasets the prereg excludes from headline
  analyses stay in the grid but not in the ratio
- divergence metrics use only unlabeled features; R^2 is leave-one-target-out
"""

import csv
import itertools
import json
from pathlib import Path

import numpy as np
import run_transfer_prediction as tp
import run_v50_decomposition as dec
from scipy.stats import wasserstein_distance

from traceshift.v50 import cohens_d_anomalous, stratified_halves, transfer_auroc

OUT = Path("output/tables/v50_grid.json")
CACHE = Path("output/cache/v50_primary_features.npz")
SEEDS = range(10)
TAUS = (0.1, 0.2)
DETECTORS = ("lr", "knn", "rf")
FEATURES = dec.FEATURES
LENGTH_FREE = tuple(f for f in FEATURES if f != "trace_length")
PRIMARY = {
    "agentdojo": "gpt-4o-2024-05-13",
    "stabletoolbench": "chatgpt_cot",
    "trail": "all",
    "swe_agent": "",
    "synthetic": "",
    "toolbench": "",
    "toolace": "",
    "apibank": "",
    "injecagent": "",
    "agentharm": "",
    "atbench": "",
    "rjudge": "",
}
EXCLUDED = {"injecagent", "agentharm", "atbench", "rjudge"}
N_PLACEBO = 1000

Matrix = dict[str, tuple[np.ndarray, np.ndarray]]


def load_primary() -> Matrix:
    """Load the 12 primary corpora as (n x 6 feature matrix, labels), caching to disk."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: (z[f"{k}__x"], z[f"{k}__y"]) for k in PRIMARY}
    loaders = dec._loaders()
    out: Matrix = {}
    for ds, sub in PRIMARY.items():
        feats, y = loaders[(ds, sub)]()
        out[ds] = (np.column_stack([np.asarray(feats[f], float) for f in FEATURES]), y.astype(int))
        print("loaded", ds, len(y), int(y.sum()), flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE,
        **{f"{k}__x": v[0] for k, v in out.items()},
        **{f"{k}__y": v[1] for k, v in out.items()},
    )
    return out


def half_a_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = [
        [cohens_d_anomalous(x[a, j], y[a]) for j in range(x.shape[1])]
        for a, _ in (stratified_halves(y, seed=s) for s in SEEDS)
    ]
    return np.mean(np.array(ds), axis=0)


def grid(data: Matrix) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cols = {
        "all": list(range(len(FEATURES))),
        "length_free": [FEATURES.index(f) for f in LENGTH_FREE],
    }
    for src, tgt in itertools.permutations(data, 2):
        (xs, ys), (xt, yt) = data[src], data[tgt]
        for det, (fs_name, idx) in itertools.product(DETECTORS, cols.items()):
            aurocs = []
            for s in SEEDS:
                a_s, _ = stratified_halves(ys, seed=s)
                _, b_t = stratified_halves(yt, seed=s)
                aurocs.append(
                    transfer_auroc(
                        (xs[a_s][:, idx], ys[a_s]), (xt[b_t][:, idx], yt[b_t]), detector=det, seed=s
                    )
                )
            rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "detector": det,
                    "features": fs_name,
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                }
            )
        print("pair", src, tgt, flush=True)
    return rows


def provenance(rule_filter: set[str]) -> dict[tuple[str, str], int]:
    """Map (dataset, feature) to 1 if construction-tied under the rule filter, from primary rows."""
    out: dict[tuple[str, str], int] = {}
    for r in csv.DictReader(dec.PRED.open()):
        if r["dataset"] not in PRIMARY or r["dataset"] in EXCLUDED:
            continue
        if r["sub_corpus"] not in ("", PRIMARY[r["dataset"]]):
            continue
        out[(r["dataset"], r["feature"])] = int(
            r["rule"] in rule_filter and r["predicted_sign"] != "0"
        )
    return out


def reversal_ratio(
    d: dict[str, np.ndarray], prov: dict[tuple[str, str], int], tau: float, rng: np.random.Generator
) -> dict[str, float]:
    datasets = sorted({k[0] for k in prov})
    cells: list[tuple[int, int]] = []
    keys = list(prov)
    for a, b in itertools.permutations(datasets, 2):
        for j, f in enumerate(FEATURES):
            rev = int(
                abs(d[a][j]) > tau and abs(d[b][j]) > tau and np.sign(d[a][j]) != np.sign(d[b][j])
            )
            cells.append((max(prov[(a, f)], prov[(b, f)]), rev))

    def ratio(tags: list[tuple[int, int]]) -> float:
        tied = [r for t, r in tags if t]
        free = [r for t, r in tags if not t]
        if not tied or not free or np.mean(free) == 0:
            return float("nan")
        return float(np.mean(tied) / np.mean(free))

    observed = ratio(cells)
    vals = np.array([prov[k] for k in keys])
    placebo = []
    for _ in range(N_PLACEBO):
        perm = dict(zip(keys, rng.permutation(vals)))
        tags = [
            (max(perm[(a, f)], perm[(b, f)]), rev)
            for (a, b, f), (_, rev) in zip(
                ((a, b, f) for a, b in itertools.permutations(datasets, 2) for f in FEATURES), cells
            )
        ]
        placebo.append(ratio(tags))
    p95 = float(np.nanpercentile(placebo, 95))
    n_rev = sum(r for _, r in cells)
    return {
        "ratio": observed,
        "placebo_p95": p95,
        "share_construction_tied": float(sum(r for t, r in cells if t) / n_rev)
        if n_rev
        else float("nan"),
        "n_cells": len(cells),
        "n_reversed": int(n_rev),
        "refuted": not (observed > 1.0 and observed > p95),
    }


def _sym_kl(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Symmetrized 50-bin KL, the definition used in the earlier version."""
    edges = np.linspace(min(a.min(), b.min()) - 1e-8, max(a.max(), b.max()) + 1e-8, n_bins + 1)
    pa = np.histogram(a, bins=edges)[0] + 1e-10
    pb = np.histogram(b, bins=edges)[0] + 1e-10
    pa, pb = pa / pa.sum(), pb / pb.sum()
    return float((np.sum(pa * np.log(pa / pb)) + np.sum(pb * np.log(pb / pa))) / 2)


def _pair_divergences(xs: np.ndarray, xt: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """KL (50-bin, symmetrized), Wasserstein, and RBF-MMD on per-domain z-scored features."""
    kl = [_sym_kl(xs[:, j], xt[:, j]) for j in range(xs.shape[1])]
    ws = [float(wasserstein_distance(xs[:, j], xt[:, j])) for j in range(xs.shape[1])]
    # Reason: the earlier run took the first 500 rows in file order, which for AgentDojo was mostly
    # one suite; a seeded random subsample keeps the cost and removes the ordering bias.
    sa = xs[rng.choice(len(xs), min(500, len(xs)), replace=False)]
    sb = xt[rng.choice(len(xt), min(500, len(xt)), replace=False)]
    return {
        "kl": float(np.mean(kl)),
        "wasserstein": float(np.mean(ws)),
        "mmd": tp.mmd_rbf(sa, sb),
    }


def _loto_r2(
    y: dict[tuple[str, str], float], x: dict[tuple[str, str], list[float]], targets: list[str]
) -> float:
    pred, true = [], []
    for t in targets:
        train = [k for k in y if k[1] != t]
        test = [k for k in y if k[1] == t]
        a = np.array([[1, *x[k]] for k in train])
        coef, *_ = np.linalg.lstsq(a, np.array([y[k] for k in train]), rcond=None)
        pred += [float(np.array([1, *x[k]]) @ coef) for k in test]
        true += [y[k] for k in test]
    ta, pa = np.array(true), np.array(pred)
    return float(1 - ((ta - pa) ** 2).sum() / ((ta - ta.mean()) ** 2).sum())


def divergence(data: Matrix, rows: list[dict[str, object]]) -> dict[str, object]:
    """Leave-one-target-out R^2 of genuine transfer AUROC on label-free divergence metrics."""

    def z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(0)) / np.where(x.std(0) < 1e-12, 1, x.std(0))

    rng = np.random.default_rng(0)
    div = {
        (s, t): _pair_divergences(z(data[s][0]), z(data[t][0]), rng)
        for s, t in itertools.permutations(data, 2)
    }
    out: dict[str, object] = {"pair_divergences": {f"{s}->{t}": v for (s, t), v in div.items()}}
    for det in DETECTORS:
        y = {
            (r["source"], r["target"]): r["auroc_mean"]
            for r in rows
            if r["detector"] == det and r["features"] == "all"
        }
        out[det] = {
            m: _loto_r2(y, {k: [v[m]] for k, v in div.items()}, list(data))
            for m in ("kl", "wasserstein", "mmd")
        }
        out[det]["all_three"] = _loto_r2(
            y, {k: [v["kl"], v["wasserstein"], v["mmd"]] for k, v in div.items()}, list(data)
        )
    return out


def main() -> None:
    data = load_primary()
    d = {ds: half_a_d(x, y) for ds, (x, y) in data.items()}
    rows = grid(data)
    OUT.with_name("v50_grid_rows.json").write_text(json.dumps(rows))
    rng = np.random.default_rng(0)
    arms = {"authors_lenient_R1-R3": {"R1", "R2", "R3"}, "authors_strict_R1": {"R1"}}
    ratio = {
        f"tau={t}/{arm}": reversal_ratio(d, provenance(rules), t, rng)
        for t in TAUS
        for arm, rules in arms.items()
    }
    result = {
        "grid": rows,
        "d_half_a": {k: dict(zip(FEATURES, map(float, v))) for k, v in d.items()},
        "reversal_ratio": ratio,
        "divergence": divergence(data, rows),
        "notes": {"ratio_datasets": sorted(set(PRIMARY) - EXCLUDED)},
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(ratio, indent=1))
    print(json.dumps(result["divergence"], indent=1))


if __name__ == "__main__":
    main()
