"""Paper figures for the TMLR v50 submission.

Palette: the dataviz reference categorical order (blue, orange, aqua, yellow), validated for CVD
separation on a light surface. Three slots fall below 3:1 contrast, so every series carries a
direct label or hatching as well as color.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52514e",
        "axes.labelcolor": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
    }
)

T = Path("output/tables")
FIG = Path("output/figures")
SUB = Path("output/paper/figures")
RUNS = Path("/tmp/agentdojo_data/runs")
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK2 = "#52514e"


def _save(fig: plt.Figure, name: str) -> None:
    for d in (FIG, SUB):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def hero_data() -> dict[str, dict[str, list[float]]]:
    """Per-run tool-call counts by class for gpt-4o with and without the PI detector."""
    out: dict[str, dict[str, list[float]]] = {}
    for run in ("gpt-4o-2024-05-13", "gpt-4o-2024-05-13-transformers_pi_detector"):
        by: dict[str, list[float]] = {"succeeded": [], "resisted": []}
        for f in sorted((RUNS / run).glob("*/*/important_instructions/*.json")):
            rec = json.loads(f.read_text())
            if rec.get("security") is None:
                continue
            n = sum(
                len(m.get("tool_calls") or [])
                for m in rec["messages"]
                if m.get("role") == "assistant"
            )
            by["succeeded" if rec["security"] else "resisted"].append(float(n))
        out[run] = by
    summary = {
        r: {
            c: {
                "n": len(v),
                "mean": float(np.mean(v)),
                "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
            }
            for c, v in by.items()
        }
        for r, by in out.items()
    }
    (T / "v50_hero_calls.json").write_text(json.dumps(summary, indent=2))
    return summary


def fig_hero() -> None:
    s = hero_data()
    runs = [
        ("No defense", "gpt-4o-2024-05-13"),
        ("PI detector", "gpt-4o-2024-05-13-transformers_pi_detector"),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    w = 0.36
    for i, (label, run) in enumerate(runs):
        for j, (cls, color, hatch) in enumerate(
            (("succeeded", BLUE, ""), ("resisted", ORANGE, "//"))
        ):
            v = s[run][cls]
            x = i + (j - 0.5) * (w + 0.02)
            ax.bar(
                x,
                v["mean"],
                w,
                yerr=1.96 * v["se"],
                color=color,
                hatch=hatch,
                edgecolor="white",
                linewidth=0.5,
                capsize=2,
                error_kw={"lw": 0.8, "ecolor": INK2},
            )
            ax.text(
                x,
                v["mean"] + 1.96 * v["se"] + 0.22,
                f"n={v['n']}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK2,
            )
    ax.set_xticks([0, 1], [r[0] for r in runs])
    ax.set_ylabel("Mean tool calls per run")
    ax.set_ylim(0, 6.6)
    handles = [
        Patch(facecolor=BLUE, label="Attack succeeded (label 1)"),
        Patch(facecolor=ORANGE, hatch="//", edgecolor="white", label="Attack resisted (label 0)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    _save(fig, "fig_hero")


def fig_null_hierarchy() -> None:
    r = json.loads((T / "v50_cross_agent_thresholds.json").read_text())
    taus = ("0.1", "0.2", "0.3")
    series = (
        ("Same configuration, split halves", "within", AQUA, ".."),
        ("Permutation null (agents shuffled)", "perm_null_mean", YELLOW, "//"),
        ("Different configurations", "cross", BLUE, ""),
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    w = 0.26
    for j, (label, key, color, hatch) in enumerate(series):
        vals = [100 * r[f"tau={t}"][key] for t in taus]
        xs = np.arange(3) + (j - 1) * (w + 0.02)
        ax.bar(xs, vals, w, color=color, hatch=hatch, edgecolor="white", linewidth=0.5, label=label)
        for x, v in zip(xs, vals):
            ax.text(
                x,
                v + 0.5,
                f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                ha="center",
                fontsize=6.5,
                color=INK2,
            )
    ax.set_xticks(range(3), [f"$|d|>{t}$" for t in taus])
    ax.set_ylabel("Feature cells that flip sign (%)")
    ax.set_ylim(0, 27)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    _save(fig, "fig_null_hierarchy")


def fig_crossover() -> None:
    r = json.loads((T / "v50_reuse_agentdojo.json").read_text())
    ks = (10, 20, 50, 100)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.3))
    for det, color, name in (
        ("rf", BLUE, "Random forest"),
        ("knn", ORANGE, "kNN"),
        ("lr", AQUA, "Logistic regression"),
    ):
        ys = [100 * r[f"all/{det}/k={k}"]["reuse_wins_share"] for k in ks]
        a.plot(ks, ys, color=color, lw=2, marker="o", ms=4, label=name)
    a.axhline(50, color=INK2, lw=0.8, ls="--")
    a.text(10.5, 47.5, "even split", ha="left", va="top", fontsize=7, color=INK2)
    a.set_xscale("log")
    a.set_xticks(ks, [str(k) for k in ks])
    a.set_xlabel("Labeled traces from the new configuration ($k$)")
    a.set_ylabel("Pairs where reuse beats\nretraining (%)")
    a.legend(frameon=False, loc="lower left")
    pol = (
        ("always_reuse", "Always reuse", BLUE, "-"),
        ("always_retrain", "Always retrain", ORANGE, "-"),
        ("sign_gate", "Sign check", AQUA, "--"),
        ("direct_gate", "Reuse-AUROC check", YELLOW, ":"),
        ("divergence_gate", "Distance check (no labels)", MAGENTA, "-."),
    )
    for key, name, color, ls in pol:
        ys = [r[f"all/rf/k={k}"]["regret"][key] for k in ks]
        b.plot(ks, ys, color=color, lw=2, ls=ls, marker="o", ms=4, label=name)
    b.set_xscale("log")
    b.set_xticks(ks, [str(k) for k in ks])
    b.set_xlabel("Labeled target traces ($k$)")
    b.set_ylabel("Regret (AUROC), RF")
    b.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_crossover")


def main() -> None:
    fig_hero()
    fig_null_hierarchy()
    fig_crossover()
    print("figures written to", FIG, "and", SUB)


if __name__ == "__main__":
    main()
