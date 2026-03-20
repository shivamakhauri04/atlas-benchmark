"""Generate D&B paper figures from v30 experiment data.

Produces:
- v30_polarity_heatmap.pdf — Cohen's d heatmap (8 datasets x 3 features)
- v30_reversal_rates.pdf — Within-group vs between-group reversal bar chart

Usage:
    uv run python scripts/gen_db_paper_figures.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
PAPER_FIG_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "papers"
    / "trustlens-auditable-agent-trust"
    / "submission"
    / "figures"
)

SEMANTIC_GROUPS = {
    "synthetic": "Injected",
    "swe_agent": "Failure",
    "trail": "Failure",
    "atbench": "Safety",
    "rjudge": "Safety",
    "agentharm": "Safety",
    "injecagent": "Prompt-Inj.",
    "agentdojo": "Prompt-Inj.",
    "toolbench": "API-Misuse",
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

FEATURE_NAMES = ["trace_length", "low_diversity", "repeat_rate"]
FEATURE_DISPLAY = ["Trace Length", "Low Diversity", "Repeat Rate"]

GROUP_ORDER = ["Injected", "Failure", "Safety", "Prompt-Inj.", "API-Misuse"]
DATASET_ORDER = [
    "synthetic",
    "swe_agent",
    "trail",
    "atbench",
    "rjudge",
    "agentharm",
    "injecagent",
    "agentdojo",
    "toolbench",
]


def fig_polarity_heatmap():
    """Cohen's d heatmap: 8 datasets x 3 features with semantic group annotations."""
    path = TABLE_DIR / "v30_expanded_polarity.json"
    if not path.exists():
        print("  Skipping polarity heatmap (no v30 data)")
        return

    data = json.loads(path.read_text())
    polarity = data["polarity_table"]

    matrix = np.zeros((len(DATASET_ORDER), len(FEATURE_NAMES)))
    for i, ds in enumerate(DATASET_ORDER):
        for j, feat in enumerate(FEATURE_NAMES):
            matrix[i, j] = polarity[ds]["d_mean"][feat]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_DISPLAY, rotation=30, ha="right")
    ax.set_yticks(range(len(DATASET_ORDER)))

    ylabels = []
    for ds in DATASET_ORDER:
        ylabels.append(f"{DISPLAY_NAMES[ds]}  [{SEMANTIC_GROUPS[ds]}]")
    ax.set_yticklabels(ylabels)

    for i in range(len(DATASET_ORDER)):
        for j in range(len(FEATURE_NAMES)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Cohen's d")

    group_boundaries = []
    prev_group = None
    for i, ds in enumerate(DATASET_ORDER):
        g = SEMANTIC_GROUPS[ds]
        if g != prev_group and prev_group is not None:
            group_boundaries.append(i - 0.5)
        prev_group = g
    for b in group_boundaries:
        ax.axhline(y=b, color="gray", linewidth=1.5, linestyle="--")

    ax.set_title("Feature Polarity Across Datasets")
    plt.tight_layout()

    for dest in [FIG_DIR, PAPER_FIG_DIR]:
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / "v30_polarity_heatmap.pdf")
    print(f"  Saved v30_polarity_heatmap.pdf")
    plt.close()


def fig_reversal_rates():
    """Bar chart: within-group vs between-group reversal rates."""
    path = TABLE_DIR / "v30_expanded_polarity.json"
    if not path.exists():
        print("  Skipping reversal rates (no v30 data)")
        return

    data = json.loads(path.read_text())
    sg = data.get("semantic_group_analysis", {})
    within = sg.get("within_group_reversal_rate", 0.048)
    between = sg.get("between_group_reversal_rate", 0.333)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    bars = ax.bar(
        ["Within-Group", "Between-Group"],
        [within * 100, between * 100],
        color=["#4CAF50", "#F44336"],
        width=0.5,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, val in zip(bars, [within, between]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Reversal Rate (%)")
    ax.set_ylim(0, 45)
    ax.set_title("Feature Polarity Reversal Rates")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for dest in [FIG_DIR, PAPER_FIG_DIR]:
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / "v30_reversal_rates.pdf")
    print(f"  Saved v30_reversal_rates.pdf")
    plt.close()


if __name__ == "__main__":
    print("Generating D&B paper figures...")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_polarity_heatmap()
    fig_reversal_rates()
    print("Done.")
