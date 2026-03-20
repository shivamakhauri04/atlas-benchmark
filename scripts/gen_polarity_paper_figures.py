"""Generate paper figures for the polarity reversal paper.

Produces:
  1. Polarity reversal heatmap (8x8 matrix, signature figure)
  2. PAFC effect size analysis (bar chart: helps vs hurts)
  3. Cross-domain AUROC matrix (heuristic raw vs PAFC)
  4. Directional asymmetry scatter (A->B vs B->A)
  5. LSTM near-random demonstration

Usage:
    cd experiments/trustlens-auditable-agent-trust
    PYTHONPATH=../../lib:src uv run python scripts/gen_polarity_paper_figures.py
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
    "text.usetex": False,
})

TABLE_DIR = Path("output/tables")
FIG_DIR = Path("output/figures")
PAPER_FIG_DIR = Path("../../papers/trustlens-auditable-agent-trust/submission/figures")

DOMAIN_ORDER = [
    "synthetic",
    "swe_agent", "trail",
    "atbench", "rjudge", "agentharm",
    "injecagent", "agentdojo",
    "toolbench",
]
DOMAIN_LABELS = {
    "agentharm": "AgentHarm",
    "agentdojo": "AgentDojo",
    "atbench": "ATBench",
    "injecagent": "InjecAgent",
    "rjudge": "R-Judge",
    "swe_agent": "SWE-agent",
    "synthetic": "Synthetic",
    "toolbench": "ToolBench",
    "trail": "TRAIL",
}

SEMANTIC_GROUPS = {
    "agentharm": "safety",
    "agentdojo": "prompt_inj",
    "atbench": "safety",
    "injecagent": "prompt_inj",
    "rjudge": "safety",
    "swe_agent": "failure",
    "trail": "failure",
    "synthetic": "injected",
    "toolbench": "api_misuse",
}

GROUP_COLORS = {
    "safety": "#4CAF50",
    "failure": "#FF9800",
    "injected": "#2196F3",
    "prompt_inj": "#E91E63",
    "api_misuse": "#9C27B0",
}


def load_polarity_data() -> dict:
    path = TABLE_DIR / "v30_expanded_polarity.json"
    with open(path) as f:
        return json.load(f)


def load_cross_domain_data() -> dict:
    path = TABLE_DIR / "v30_expanded_cross_domain.json"
    with open(path) as f:
        return json.load(f)


def fig1_polarity_reversal_heatmap(polarity: dict) -> None:
    """8x8 heatmap: reversal rate per domain pair, colored by group."""
    reversal_matrix = polarity["reversal_matrix"]
    n = len(DOMAIN_ORDER)
    features = ["trace_length", "low_diversity", "repeat_rate"]

    matrix = np.zeros((n, n))
    for i, da in enumerate(DOMAIN_ORDER):
        for j, db in enumerate(DOMAIN_ORDER):
            if da == db:
                matrix[i, j] = 0.0
            else:
                reversals = reversal_matrix.get(da, {}).get(db, {})
                n_rev = sum(1 for f in features if reversals.get(f, False))
                matrix[i, j] = n_rev / len(features)

    fig, ax = plt.subplots(figsize=(7, 6))

    within_mask = np.zeros((n, n), dtype=bool)
    for i, da in enumerate(DOMAIN_ORDER):
        for j, db in enumerate(DOMAIN_ORDER):
            if SEMANTIC_GROUPS[da] == SEMANTIC_GROUPS[db]:
                within_mask[i, j] = True

    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=9, color="gray")
            else:
                val = matrix[i, j]
                color = "white" if val > 0.5 else "black"
                n_rev = int(val * len(features))
                ax.text(j, i, f"{n_rev}/3", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if within_mask[i, j] and i != j:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=1.5, edgecolor="#4CAF50", facecolor="none",
                    linestyle="--", alpha=0.8,
                )
                ax.add_patch(rect)

    labels = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Target Domain")
    ax.set_ylabel("Source Domain")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of Features Reversed", fontsize=10)

    sa = polarity["semantic_group_analysis"]
    ax.set_title(
        f"Feature Polarity Reversal Matrix\n"
        f"Within-group: {sa['within_group_rate']:.1%} | "
        f"Between-group: {sa['between_group_rate']:.1%}",
        fontsize=11,
    )

    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "polarity_reversal_heatmap.pdf")
        fig.savefig(d / "polarity_reversal_heatmap.png")
    plt.close()
    print("  polarity_reversal_heatmap.pdf")


def fig2_pafc_effect_size(cross_domain: dict) -> None:
    """Bar chart showing PAFC delta per pair, sorted by effect."""
    results = cross_domain["results"]
    pairs = []
    for pair_key, methods in results.items():
        raw_mean = float(np.mean(methods["heuristic_raw"]))
        pafc_mean = float(np.mean(methods["heuristic_pafc"]))
        delta = pafc_mean - raw_mean
        pairs.append({"pair": pair_key, "raw": raw_mean, "pafc": pafc_mean, "delta": delta})

    pairs.sort(key=lambda x: x["delta"])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pairs))
    colors = []
    for p in pairs:
        if p["delta"] > 0.05:
            colors.append("#4CAF50")
        elif p["delta"] < -0.05:
            colors.append("#F44336")
        else:
            colors.append("#9E9E9E")

    bars = ax.bar(x, [p["delta"] for p in pairs], color=colors, alpha=0.85, edgecolor="white")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=0.05, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=-0.05, color="red", linewidth=0.5, linestyle="--", alpha=0.5)

    pair_labels = [p["pair"].replace("->", "\n→ ") for p in pairs]
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=90, fontsize=6, ha="center")
    ax.set_ylabel("ΔAUROC (PAFC - Raw)")
    ax.set_title("PAFC Correction Effect Across Domain Pairs")

    n_helps = sum(1 for p in pairs if p["delta"] > 0.05)
    n_hurts = sum(1 for p in pairs if p["delta"] < -0.05)
    n_neutral = len(pairs) - n_helps - n_hurts
    ax.text(
        0.02, 0.98,
        f"Helps: {n_helps} | Neutral: {n_neutral} | Hurts: {n_hurts}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / "pafc_effect_size.pdf")
        fig.savefig(d / "pafc_effect_size.png")
    plt.close()
    print("  pafc_effect_size.pdf")


def fig3_cross_domain_auroc_matrix(cross_domain: dict) -> None:
    """Side-by-side AUROC matrices: raw vs PAFC-corrected."""
    results = cross_domain["results"]
    n = len(DOMAIN_ORDER)

    raw_matrix = np.full((n, n), np.nan)
    pafc_matrix = np.full((n, n), np.nan)

    for i, src in enumerate(DOMAIN_ORDER):
        for j, tgt in enumerate(DOMAIN_ORDER):
            key = f"{src}->{tgt}"
            if key in results:
                raw_matrix[i, j] = float(np.mean(results[key]["heuristic_raw"]))
                pafc_matrix[i, j] = float(np.mean(results[key]["heuristic_pafc"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, matrix, title in [
        (ax1, raw_matrix, "Heuristic (Raw)"),
        (ax2, pafc_matrix, "Heuristic (PAFC)"),
    ]:
        cmap = plt.cm.RdYlGn
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val < 0.4 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        labels = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        ax.set_title(title, fontsize=11)

    fig.colorbar(im, ax=[ax1, ax2], fraction=0.02, pad=0.04, label="AUROC")
    fig.suptitle("Cross-Domain Transfer AUROC", fontsize=13, y=1.02)
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / "cross_domain_auroc_matrix.pdf")
        fig.savefig(d / "cross_domain_auroc_matrix.png")
    plt.close()
    print("  cross_domain_auroc_matrix.pdf")


def fig4_directional_asymmetry(cross_domain: dict) -> None:
    """Scatter: AUROC(A->B) vs AUROC(B->A) showing 1-x asymmetry."""
    results = cross_domain["results"]
    pairs_seen = set()
    x_vals, y_vals, labels_list = [], [], []

    for pair_key, methods in results.items():
        src, tgt = pair_key.split("->")
        rev_key = f"{tgt}->{src}"
        unordered = tuple(sorted([src, tgt]))
        if unordered in pairs_seen:
            continue
        if rev_key not in results:
            continue
        pairs_seen.add(unordered)

        ab = float(np.mean(methods["heuristic_raw"]))
        ba = float(np.mean(results[rev_key]["heuristic_raw"]))
        x_vals.append(ab)
        y_vals.append(ba)
        labels_list.append(f"{DOMAIN_LABELS.get(src, src)[:3]}→{DOMAIN_LABELS.get(tgt, tgt)[:3]}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_vals, y_vals, s=40, alpha=0.7, edgecolors="white", linewidth=0.5, c="#2196F3")

    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, 1 - x_line, "r--", alpha=0.5, linewidth=1, label="y = 1 - x (anti-prediction)")
    ax.plot(x_line, x_line, "gray", alpha=0.3, linewidth=0.8, linestyle=":", label="y = x (symmetric)")

    for x, y, label in zip(x_vals, y_vals, labels_list):
        ax.annotate(label, (x, y), fontsize=5.5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("AUROC (A → B)")
    ax.set_ylabel("AUROC (B → A)")
    ax.set_title("Directional Asymmetry in Cross-Domain Detection")
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.1, 1.0)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / "directional_asymmetry.pdf")
        fig.savefig(d / "directional_asymmetry.png")
    plt.close()
    print("  directional_asymmetry.pdf")


def fig5_lstm_near_random(cross_domain: dict) -> None:
    """Box plot showing LSTM is near-random on all cross-domain pairs."""
    results = cross_domain["results"]

    lstm_aurocs = []
    pair_names = []
    for pair_key, methods in results.items():
        if "lstm" in methods:
            lstm_aurocs.append(methods["lstm"])
            pair_names.append(pair_key)

    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(
        lstm_aurocs, vert=True, patch_artist=True,
        boxprops=dict(facecolor="#E3F2FD", edgecolor="#1565C0"),
        medianprops=dict(color="#D32F2F", linewidth=1.5),
        whiskerprops=dict(color="#1565C0"),
        capprops=dict(color="#1565C0"),
        flierprops=dict(marker="o", markerfacecolor="#FF9800", markersize=3),
    )

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Random (0.50)")

    short_labels = []
    for p in pair_names:
        src, tgt = p.split("->")
        short_labels.append(f"{src[:3]}→{tgt[:3]}")

    ax.set_xticks(range(1, len(pair_names) + 1))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("AUROC")
    ax.set_title("LSTM Cross-Domain AUROC (10 seeds per pair)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    all_medians = [np.median(vals) for vals in lstm_aurocs]
    grand_median = np.median(all_medians)
    ax.text(
        0.98, 0.98,
        f"Grand median: {grand_median:.3f}",
        transform=ax.transAxes, fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / "lstm_near_random.pdf")
        fig.savefig(d / "lstm_near_random.png")
    plt.close()
    print("  lstm_near_random.pdf")


def fig6_cohens_d_profiles(polarity: dict) -> None:
    """Grouped bar chart showing Cohen's d per domain per feature."""
    table = polarity["polarity_table"]
    features = ["trace_length", "low_diversity", "repeat_rate"]
    n_features = len(features)
    n_domains = len(DOMAIN_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(n_domains)

    for fi, feat in enumerate(features):
        vals = []
        for domain in DOMAIN_ORDER:
            d_mean = table[domain]["d_mean"][feat]
            vals.append(d_mean)
        bars = ax.bar(
            x + fi * width, vals, width,
            label=feat.replace("_", " ").title(),
            alpha=0.85, edgecolor="white",
        )

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=0.1, color="gray", linewidth=0.3, linestyle=":")
    ax.axhline(y=-0.1, color="gray", linewidth=0.3, linestyle=":")

    labels = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's d (anomalous - normal)")
    ax.set_title("Feature-Label Effect Sizes Across Domains")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for d in [FIG_DIR, PAPER_FIG_DIR]:
        fig.savefig(d / "cohens_d_profiles.pdf")
        fig.savefig(d / "cohens_d_profiles.png")
    plt.close()
    print("  cohens_d_profiles.pdf")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Paper Figure Generation: Polarity Reversal")
    print("=" * 60)

    print("\nLoading data...")
    polarity = load_polarity_data()
    cross_domain = load_cross_domain_data()
    print(f"  Polarity: {polarity['n_domains']} domains")
    print(f"  Cross-domain: {len(cross_domain['domain_pairs'])} pairs")

    print("\nGenerating figures...")
    fig1_polarity_reversal_heatmap(polarity)
    fig2_pafc_effect_size(cross_domain)
    fig3_cross_domain_auroc_matrix(cross_domain)
    fig4_directional_asymmetry(cross_domain)
    fig5_lstm_near_random(cross_domain)
    fig6_cohens_d_profiles(polarity)

    print(f"\nFigures saved to:")
    print(f"  {FIG_DIR}")
    print(f"  {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
