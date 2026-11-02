"""Earlier-version reanalyses -- all recomputed from stored artifacts, no training, no cost.

Produces every earlier-version number we cite, each traceable to a stored file:
  1. Clean permutation p-values (4-group / 5-group group-rate gap).
  2. (A) taxonomy-free reversal prevalence + threshold-sensitivity curve.
  3. (C) divergence-metric blindness (KL/MMD/Wasserstein vs AUROC), scoped to what exists.
  4. (D) mitigation improve/worsen split + the multi-source-vs-sign-flip mechanism check.

Reads output/tables/*.json only; writes output/tables/v49_reanalysis.json.
The organic-only causal gate (B) is not part of this release.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TABLES = Path(__file__).resolve().parent.parent / "output" / "tables"
OUT = TABLES / "v49_reanalysis.json"

STRUCTURAL = ["trace_length", "low_diversity", "repeat_rate"]
INJECTED = {"synthetic", "toolbench", "toolace", "apibank"}

GROUPS_4 = {
    "injected": ["synthetic", "toolace"],
    "failure": ["swe_agent", "trail", "stabletoolbench"],
    "safety": ["atbench", "agentharm", "rjudge", "injecagent", "agentdojo"],
    "api_misuse": ["toolbench", "apibank"],
}
GROUPS_5 = {
    "injected": ["synthetic", "toolace"],
    "failure": ["swe_agent", "trail", "stabletoolbench"],
    "safety": ["atbench", "agentharm", "rjudge"],
    "prompt_injection": ["injecagent", "agentdojo"],
    "api_misuse": ["toolbench", "apibank"],
}


def _rev(a: float, b: float, thr: float) -> bool:
    """Sign reversal with a minimum effect-size gate on both sides."""
    if abs(a) < thr or abs(b) < thr:
        return False
    return (a > 0) != (b > 0)


def _group_rate_gap(domains, d_mean, dom2group, thr=0.1):
    """Per-feature within/between reversal rates (the paper's headline metric)."""
    wr = wt = br = bt = 0
    for a, b in combinations(domains, 2):
        for f in STRUCTURAL:
            r = _rev(d_mean[a][f], d_mean[b][f], thr)
            if dom2group[a] == dom2group[b]:
                wt += 1
                wr += r
            else:
                bt += 1
                br += r
    within = wr / wt if wt else 0.0
    between = br / bt if bt else 0.0
    return within, between, (between / within if within else float("inf")), (wr, wt, br, bt)


def _permutation_p(domains, d_mean, groups, n_perm=20000, thr=0.1):
    """Permutation test: is the observed between-within gap larger than under random grouping?"""
    sizes = [len(v) for v in groups.values()]
    dom2group = {x: g for g, xs in groups.items() for x in xs}
    within, between, _, _ = _group_rate_gap(domains, d_mean, dom2group, thr)
    observed = between - within

    rng = np.random.RandomState(0)
    dl = list(domains)
    count = 0
    for _ in range(n_perm):
        perm = list(dl)
        rng.shuffle(perm)
        assign, i = {}, 0
        for g, sz in zip(groups.keys(), sizes):
            for _ in range(sz):
                assign[perm[i]] = g
                i += 1
        w, b, _, _ = _group_rate_gap(domains, d_mean, assign, thr)
        if (b - w) >= observed:
            count += 1
    return {
        "within_rate": within,
        "between_rate": between,
        "gap": observed,
        "ratio": (between / within if within else None),
        "p_value": (count + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


def _taxonomy_free_prevalence(domains, d_mean):
    """(A) Fraction of pairs with >=1 structural sign flip, across a threshold sweep -- no groups."""
    curve = {}
    for thr in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
        n_pairs = flip_pairs = 0
        perfeat_rev = perfeat_tot = 0
        for a, b in combinations(domains, 2):
            n_pairs += 1
            any_flip = False
            for f in STRUCTURAL:
                perfeat_tot += 1
                r = _rev(d_mean[a][f], d_mean[b][f], thr)
                perfeat_rev += r
                any_flip = any_flip or r
            flip_pairs += any_flip
        curve[f"{thr:.1f}"] = {
            "pair_flip_rate": flip_pairs / n_pairs,
            "n_flip_pairs": flip_pairs,
            "n_pairs": n_pairs,
            "per_feature_rate": perfeat_rev / perfeat_tot,
        }
    return curve


def _mitigation(v47_path, v48_path):
    """(D) improve/worsen split (v47, 10-seed) + sign-flip-vs-diversity mechanism (v48)."""
    v47 = json.loads(Path(v47_path).read_text())["results"]
    reversed_rows = [r for r in v47 if (not r["is_within"]) and r["n_reversed"] >= 1]
    out = {"reversed_pairs_10seed": {}}
    for clf in ["lr", "knn", "rf"]:
        deltas = np.array([r[f"multi_delta_{clf}"] for r in reversed_rows])
        out["reversed_pairs_10seed"][clf] = {
            "n_obs": len(deltas),
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std()),
            "improve_frac": float((deltas > 0.01).mean()),
            "flat_frac": float((np.abs(deltas) <= 0.01).mean()),
            "worsen_frac": float((deltas < -0.01).mean()),
        }

    v48 = json.loads(Path(v48_path).read_text())["results"]
    rev48 = [r for r in v48 if (not r["is_within"]) and r["n_reversed"] >= 1]
    mech = {}
    for clf in ["lr", "knn", "rf"]:
        multi_col = f"multi_{clf}"
        noflip_col = f"multi_noflip_{clf}"
        if rev48 and noflip_col in rev48[0]:
            multi = np.array([r[multi_col] for r in rev48])
            noflip = np.array([r[noflip_col] for r in rev48])
            mech[clf] = {
                "multi_mean": float(multi.mean()),
                "multi_noflip_mean": float(noflip.mean()),
                "noflip_ge_multi": bool(noflip.mean() >= multi.mean()),
            }
        else:
            mech[clf] = {"note": f"{noflip_col} not present in v48"}
    out["mechanism_flip_vs_diversity"] = mech
    return out


def _divergence(path):
    """(C) Do KL/MMD/Wasserstein predict transfer AUROC? Read stored correlations + meta-predictor."""
    p = Path(path)
    if not p.exists():
        return {"note": f"{p.name} not found"}
    data = json.loads(p.read_text())
    corr = data.get("correlations", {})
    metrics = {}
    for m in ["wasserstein_mean", "kl_mean", "mmd_rbf"]:
        if m in corr:
            metrics[m] = corr[m]["vs_auroc_raw"]
    mp = data.get("meta_predictor", {})
    return {
        "n_domains": data.get("n_domains"),
        "n_pairs_with_auroc": mp.get("n_samples"),
        "metric_vs_auroc_spearman": metrics,
        "all_divergence_p_gt_0.1": all(v["p"] > 0.1 for v in metrics.values()) if metrics else None,
        "meta_predictor_r_squared": mp.get("r_squared"),
        "group_only_r_squared": mp.get("group_only_r_squared"),
    }


def main() -> None:
    pol = json.loads((TABLES / "v30_expanded_polarity.json").read_text())
    pt = pol["polarity_table"]
    domains = sorted(pt.keys())
    d_mean = {dom: pt[dom]["d_mean"] for dom in domains}

    result = {
        "domains": domains,
        "structural_features": STRUCTURAL,
        "reversal_threshold_default": 0.1,
        "permutation_tests": {
            "4_group": _permutation_p(domains, d_mean, GROUPS_4),
            "5_group": _permutation_p(domains, d_mean, GROUPS_5),
        },
        "taxonomy_free_prevalence": _taxonomy_free_prevalence(domains, d_mean),
        "mitigation": _mitigation(
            TABLES / "v47_polarity_invariant_training.json",
            TABLES / "v48_polarity_invariant_training.json",
        ),
        "divergence_blindness": _divergence(TABLES / "transfer_prediction.json"),
    }
    OUT.write_text(json.dumps(result, indent=2))

    p4 = result["permutation_tests"]["4_group"]
    p5 = result["permutation_tests"]["5_group"]
    print(f"Wrote {OUT.name}\n")
    print("=== Permutation p-values (group-rate gap) ===")
    print(f"  4-group: within {p4['within_rate']*100:.1f}% between {p4['between_rate']*100:.1f}% "
          f"ratio {p4['ratio']:.2f}x  p={p4['p_value']:.4f}")
    print(f"  5-group: within {p5['within_rate']*100:.1f}% between {p5['between_rate']*100:.1f}% "
          f"ratio {p5['ratio']:.2f}x  p={p5['p_value']:.4f}")
    print("\n=== (A) taxonomy-free prevalence (pair flip rate by threshold) ===")
    for thr, v in result["taxonomy_free_prevalence"].items():
        print(f"  |d|>={thr}: {v['n_flip_pairs']}/{v['n_pairs']} pairs "
              f"= {v['pair_flip_rate']*100:.1f}% flip; per-feature {v['per_feature_rate']*100:.1f}%")
    print("\n=== (D) mitigation improve/worsen (reversed pairs, 10 seed) ===")
    for clf, v in result["mitigation"]["reversed_pairs_10seed"].items():
        print(f"  {clf.upper():4s} mean {v['mean_delta']:+.3f}+-{v['std_delta']:.3f}  "
              f"improve {v['improve_frac']*100:.0f}% flat {v['flat_frac']*100:.0f}% "
              f"worsen {v['worsen_frac']*100:.0f}%")
    print("\n=== (D) mechanism: sign-flip vs plain multi-source ===")
    for clf, v in result["mitigation"]["mechanism_flip_vs_diversity"].items():
        print(f"  {clf.upper():4s} {v}")
    print("\n=== (C) divergence blindness ===")
    print(f"  {json.dumps(result['divergence_blindness'])[:400]}")


if __name__ == "__main__":
    main()
