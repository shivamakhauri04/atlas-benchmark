"""Compile full cross-domain results table for appendix.

Combines v30_expanded_cross_domain.json (4 scorers x 26 pairs) with
v29_fewshot_baselines.json to produce the full results matrix.

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/compile_full_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("output/tables")

SEMANTIC_GROUPS = {
    "injected": ["synthetic"],
    "failure": ["swe_agent", "trail"],
    "safety": ["atbench", "injecagent", "agentharm", "rjudge"],
    "api_misuse": ["toolbench"],
}


def get_group(domain: str) -> str:
    for g, members in SEMANTIC_GROUPS.items():
        if domain in members:
            return g
    return "unknown"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load v30 cross-domain results (4 scorers x 26 pairs x 10 seeds)
    with open(OUTPUT_DIR / "v30_expanded_cross_domain.json") as f:
        v30 = json.load(f)

    # Load v29 fewshot baselines
    with open(OUTPUT_DIR / "v29_fewshot_baselines.json") as f:
        fewshot = json.load(f)

    # Load v29 conformal results
    with open(OUTPUT_DIR / "v29_pafc_conformal.json") as f:
        conformal = json.load(f)

    # Build the full results table
    full_table: list[dict] = []

    # v30 cross-domain: 4 scorers x 26 pairs
    v30_results = v30["results"]
    scorers_v30 = ["heuristic_raw", "heuristic_pafc", "lstm", "hybrid_pafc"]

    for pair_key, pair_data in v30_results.items():
        parts = pair_key.split("->")
        src, tgt = parts[0], parts[1]
        src_group = get_group(src)
        tgt_group = get_group(tgt)
        pair_type = "within" if src_group == tgt_group else "between"

        for scorer in scorers_v30:
            auroc_list = pair_data.get(scorer, [])
            if not auroc_list:
                continue
            mean_auroc = float(np.mean(auroc_list))
            std_auroc = float(np.std(auroc_list))
            full_table.append({
                "source": src,
                "target": tgt,
                "src_group": src_group,
                "tgt_group": tgt_group,
                "pair_type": pair_type,
                "method": scorer,
                "family": _method_family(scorer),
                "auroc_mean": round(mean_auroc, 3),
                "auroc_std": round(std_auroc, 3),
                "n_seeds": len(auroc_list),
            })

    # Fewshot baselines: aggregate per (src, tgt, method) at n=100
    fewshot_agg: dict[tuple, list[float]] = {}
    for r in fewshot["results"]:
        if int(r["n_labeled"]) != 100:
            continue
        key = (r["src"], r["tgt"], r["method"])
        fewshot_agg.setdefault(key, []).append(float(r["auroc"]))

    for (src, tgt, method), aurocs in fewshot_agg.items():
        src_group = get_group(src)
        tgt_group = get_group(tgt)
        pair_type = "within" if src_group == tgt_group else "between"
        full_table.append({
            "source": src,
            "target": tgt,
            "src_group": src_group,
            "tgt_group": tgt_group,
            "pair_type": pair_type,
            "method": f"fewshot_{method}",
            "family": "few-shot",
            "auroc_mean": round(float(np.mean(aurocs)), 3),
            "auroc_std": round(float(np.std(aurocs)), 3),
            "n_seeds": len(aurocs),
        })

    # Conformal results
    if "results" in conformal:
        conf_results = conformal["results"]
        if isinstance(conf_results, dict):
            for pair_key, pair_data in conf_results.items():
                if "->" not in pair_key:
                    continue
                src, tgt = pair_key.split("->")
                src_group = get_group(src)
                tgt_group = get_group(tgt)
                pair_type = "within" if src_group == tgt_group else "between"
                for method_name, method_data in pair_data.items():
                    if isinstance(method_data, dict) and "auroc" in method_data:
                        aurocs = method_data["auroc"]
                        if isinstance(aurocs, list):
                            full_table.append({
                                "source": src,
                                "target": tgt,
                                "src_group": src_group,
                                "tgt_group": tgt_group,
                                "pair_type": pair_type,
                                "method": f"conformal_{method_name}",
                                "family": "conformal",
                                "auroc_mean": round(float(np.mean(aurocs)), 3),
                                "auroc_std": round(float(np.std(aurocs)), 3),
                                "n_seeds": len(aurocs),
                            })

    # Sort by pair type, then source, target, method
    full_table.sort(key=lambda x: (x["pair_type"], x["source"], x["target"], x["method"]))

    # Print summary
    methods_seen = sorted(set(r["method"] for r in full_table))
    pairs_seen = sorted(set(f"{r['source']}->{r['target']}" for r in full_table))

    print(f"Full results table: {len(full_table)} entries")
    print(f"Methods ({len(methods_seen)}): {methods_seen}")
    print(f"Pairs ({len(pairs_seen)}): {pairs_seen[:10]}...")

    # Generate LaTeX table for appendix (condensed: 26 pairs x main methods)
    latex_lines = generate_latex_table(full_table, v30_results, scorers_v30)

    # Save
    out_path = OUTPUT_DIR / "v30_full_cross_domain_table.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "v30_full_cross_domain_compilation",
            "n_entries": len(full_table),
            "methods": methods_seen,
            "pairs": pairs_seen,
            "results": full_table,
        }, f, indent=2)

    latex_path = OUTPUT_DIR / "v30_full_results_latex.tex"
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\nSaved JSON to {out_path}")
    print(f"Saved LaTeX to {latex_path}")


def _method_family(method: str) -> str:
    if "heuristic" in method:
        return "heuristic"
    if "lstm" in method or "hybrid" in method:
        return "learned"
    if "fewshot" in method:
        return "few-shot"
    if "conformal" in method or "pafc" in method:
        return "conformal"
    return "other"


def generate_latex_table(
    full_table: list[dict],
    v30_results: dict,
    scorers: list[str],
) -> list[str]:
    """Generate LaTeX table with all 26 pairs x 4 primary scorers."""
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\caption{Full cross-domain AUROC (mean $\pm$ std, 10 seeds) for 4 primary scorers across all 26 transfer pairs. Bold: best per row. Pairs grouped by within/between semantic groups.}",
        r"\label{tab:full-crossdomain}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Source} & \textbf{Target} & \textbf{Heuristic} & \textbf{PAFC} & \textbf{LSTM} & \textbf{Hybrid} \\",
        r"\midrule",
    ]

    # Sort pairs: within-group first, then between-group
    pair_keys = sorted(
        v30_results.keys(),
        key=lambda p: (
            0 if get_group(p.split("->")[0]) == get_group(p.split("->")[1]) else 1,
            p,
        ),
    )

    prev_type = None
    for pair_key in pair_keys:
        src, tgt = pair_key.split("->")
        src_g = get_group(src)
        tgt_g = get_group(tgt)
        pair_type = "within" if src_g == tgt_g else "between"

        if prev_type != pair_type:
            if prev_type is not None:
                lines.append(r"\midrule")
            label = "Within-group" if pair_type == "within" else "Between-group"
            lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{label} transfers}}}} \\")
            prev_type = pair_type

        pair_data = v30_results[pair_key]
        vals = {}
        for scorer in scorers:
            arr = pair_data.get(scorer, [])
            if arr:
                vals[scorer] = (np.mean(arr), np.std(arr))
            else:
                vals[scorer] = (float("nan"), float("nan"))

        # Bold the best
        best_scorer = max(scorers, key=lambda s: vals[s][0] if not np.isnan(vals[s][0]) else -1)

        cells = []
        for scorer in scorers:
            m, s = vals[scorer]
            if np.isnan(m):
                cells.append("---")
            else:
                cell = f".{int(m*1000):03d}$\\pm$.{int(s*100):02d}"
                if scorer == best_scorer:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)

        lines.append(
            f"{src} & {tgt} & {' & '.join(cells)} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return lines


if __name__ == "__main__":
    main()
