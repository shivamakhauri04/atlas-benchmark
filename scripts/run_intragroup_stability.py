"""E5: Intra-group polarity stability analysis.

Shows that within each semantic group, Cohen's d profiles are consistent
(similar signs and magnitudes), while between-group profiles diverge.

Produces:
  - output/figures/intragroup_stability.pdf
  - output/tables/e5_intragroup_stability.json

Usage:
    PYTHONPATH=../../lib:src uv run python scripts/run_intragroup_stability.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TABLE_DIR = Path("output/tables")
FIG_DIR = Path("output/figures")
PAPER_FIG_DIR = Path(
    "../../papers/trustlens-auditable-agent-trust/submission/figures"
)

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]
FEATURE_DISPLAY = {
    "trace_length": "Trace Length",
    "low_diversity": "Low Diversity",
    "repeat_rate": "Repeat Rate",
}

GROUPINGS_5 = {
    "Injected": ["synthetic"],
    "Failure": ["swe_agent", "trail"],
    "Safety": ["atbench", "agentharm", "rjudge"],
    "Prompt-Inj.": ["injecagent", "agentdojo"],
    "API-Misuse": ["toolbench"],
}

DISPLAY_NAMES = {
    "synthetic": "Synthetic",
    "swe_agent": "SWE-agent",
    "trail": "TRAIL",
    "atbench": "ATBench",
    "rjudge": "R-Judge",
    "agentharm": "AgentHarm",
    "injecagent": "InjecAgent",
    "agentdojo": "AgentDojo",
    "toolbench": "ToolBench",
}

GROUP_COLORS = {
    "Injected": "#2196F3",
    "Failure": "#FF9800",
    "Safety": "#4CAF50",
    "Prompt-Inj.": "#E91E63",
    "API-Misuse": "#9C27B0",
}


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    polarity_path = TABLE_DIR / "v30_expanded_polarity.json"
    data = json.loads(polarity_path.read_text())
    polarity_table = data["polarity_table"]

    multi_groups = {
        g: members for g, members in GROUPINGS_5.items()
        if len([m for m in members if m in polarity_table]) >= 2
    }

    fig, axes = plt.subplots(
        1, len(multi_groups), figsize=(4 * len(multi_groups), 4),
        sharey=True,
    )
    if len(multi_groups) == 1:
        axes = [axes]

    intragroup_stats = {}

    for ax, (group_name, members) in zip(axes, multi_groups.items()):
        available = [m for m in members if m in polarity_table]
        x = np.arange(len(FEATURE_NAMES))
        width = 0.8 / len(available)

        group_d_vectors = []

        for i, domain in enumerate(available):
            d_vals = [polarity_table[domain]["d_mean"][f] for f in FEATURE_NAMES]
            group_d_vectors.append(d_vals)
            bars = ax.bar(
                x + i * width - 0.4 + width / 2,
                d_vals,
                width,
                label=DISPLAY_NAMES.get(domain, domain),
                alpha=0.85,
                edgecolor="white",
            )

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FEATURE_DISPLAY[f] for f in FEATURE_NAMES],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_title(group_name, fontsize=11, color=GROUP_COLORS.get(group_name, "black"))
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.2)

        d_matrix = np.array(group_d_vectors)
        sign_agreement = []
        for fi in range(len(FEATURE_NAMES)):
            signs = np.sign(d_matrix[:, fi])
            signs = signs[signs != 0]
            if len(signs) > 0:
                agreement = max(
                    (signs > 0).sum(), (signs < 0).sum()
                ) / len(signs)
                sign_agreement.append(agreement)

        avg_agreement = float(np.mean(sign_agreement)) if sign_agreement else 0.0

        if d_matrix.shape[0] >= 2:
            from itertools import combinations
            cosines = []
            for i, j in combinations(range(d_matrix.shape[0]), 2):
                norm_i = np.linalg.norm(d_matrix[i])
                norm_j = np.linalg.norm(d_matrix[j])
                if norm_i > 1e-8 and norm_j > 1e-8:
                    cos = float(np.dot(d_matrix[i], d_matrix[j]) / (norm_i * norm_j))
                    cosines.append(cos)
            mean_cosine = float(np.mean(cosines)) if cosines else 0.0
        else:
            mean_cosine = 1.0

        intragroup_stats[group_name] = {
            "n_datasets": len(available),
            "datasets": available,
            "sign_agreement": round(avg_agreement, 3),
            "mean_pairwise_cosine": round(mean_cosine, 3),
            "d_values": {
                domain: {f: polarity_table[domain]["d_mean"][f] for f in FEATURE_NAMES}
                for domain in available
            },
        }

    axes[0].set_ylabel("Cohen's d (anomalous − normal)")
    fig.suptitle(
        "Intra-Group Polarity Consistency (5-group taxonomy)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    for dest in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(dest / "intragroup_stability.pdf")
    plt.close()
    print("  Saved intragroup_stability.pdf")

    output = {
        "experiment": "e5_intragroup_stability",
        "grouping": "5group",
        "groups": intragroup_stats,
    }
    out_path = TABLE_DIR / "e5_intragroup_stability.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Results saved to {out_path}")

    print("\n  Intra-group summary:")
    for group, stats in intragroup_stats.items():
        print(f"    {group}: sign_agreement={stats['sign_agreement']:.1%}, "
              f"cosine={stats['mean_pairwise_cosine']:.3f}, "
              f"n={stats['n_datasets']}")


if __name__ == "__main__":
    main()
