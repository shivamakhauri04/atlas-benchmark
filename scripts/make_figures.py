"""Generate paper figures and tables.

Reads results from output/tables/ and produces publication-quality
matplotlib figures in output/figures/.

Usage:
    python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


def fig_baselines_comparison():
    """Bar chart: independent scoring baselines comparison (Table 1 only).

    Uses only independent anomaly scorers for apples-to-apples AUROC
    comparison. Calibration variants share encoder scores (identical
    AUROC) and are shown separately.
    """
    path = TABLE_DIR / "baselines.json"
    if not path.exists():
        print("  Skipping baselines comparison (no data)")
        return

    raw = json.loads(path.read_text())

    # Support both structured (table1_scoring) and legacy flat format
    if isinstance(raw, dict) and "table1_scoring" in raw:
        data = raw["table1_scoring"]
    elif isinstance(raw, list):
        data = raw
    else:
        print("  Skipping baselines comparison (unrecognized format)")
        return

    names = [d["baseline"] for d in data]
    aurocs = [d["auroc"] for d in data]
    f1s = [d["f1"] for d in data]
    coverages = [d["coverage"] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = np.arange(len(names))
    w = 0.6

    for ax, values, title, color in [
        (axes[0], aurocs, "AUROC", "#2196F3"),
        (axes[1], f1s, "F1 Score", "#4CAF50"),
        (axes[2], coverages, "Coverage", "#FF9800"),
    ]:
        ax.bar(x, values, w, color=color, alpha=0.8, edgecolor="white")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label="Target")
        if title == "Coverage":
            ax.legend(loc="lower right")

    plt.suptitle("Table 1: Independent Anomaly Scorers", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "baselines_comparison.pdf")
    fig.savefig(FIG_DIR / "baselines_comparison.png")
    plt.close()
    print("  baselines_comparison.pdf")


def fig_conformal_coverage():
    """Alpha sweep: coverage vs safety tradeoff."""
    path = TABLE_DIR / "conformal.json"
    if not path.exists():
        print("  Skipping conformal coverage (no data)")
        return

    data = json.loads(path.read_text())
    sweep = data.get("alpha_sweep", [])
    if not sweep:
        print("  Skipping alpha sweep (no data)")
        return

    alphas = [s["alpha"] for s in sweep]
    coverages = [s["coverage"] for s in sweep]
    recalls = [s["recall"] for s in sweep]
    f1s = [s["f1"] for s in sweep]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, coverages, "o-", label="Coverage (safe actions)", color="#2196F3")
    ax.plot(alphas, recalls, "s-", label="Recall (detects unsafe)", color="#F44336")
    ax.plot(alphas, f1s, "^-", label="F1", color="#4CAF50")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel(r"$\alpha$ (target miscoverage rate)")
    ax.set_ylabel("Metric value")
    ax.set_title("Coverage vs Safety Tradeoff")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.savefig(FIG_DIR / "alpha_sweep.pdf")
    fig.savefig(FIG_DIR / "alpha_sweep.png")
    plt.close()
    print("  alpha_sweep.pdf")


def fig_tau_adaptation():
    """ACI and FA-CP threshold adaptation over time."""
    path = TABLE_DIR / "tau_histories.npz"
    if not path.exists():
        print("  Skipping tau adaptation (no data)")
        return

    data = np.load(path)
    aci_tau = data["aci_tau"]
    facp_tau = data["facp_tau"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    steps = np.arange(len(aci_tau))
    ax.plot(steps, aci_tau, label="ACI (Theorem 2)", alpha=0.8, linewidth=1)
    ax.plot(steps[: len(facp_tau)], facp_tau, label="FA-CP (Theorem 3)", alpha=0.8, linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Threshold $\tau_t$")
    ax.set_title("Online Threshold Adaptation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(FIG_DIR / "tau_adaptation.pdf")
    fig.savefig(FIG_DIR / "tau_adaptation.png")
    plt.close()
    print("  tau_adaptation.pdf")


def fig_shift_detection():
    """Feedback-aware CP: detected distribution shift over time."""
    path = TABLE_DIR / "tau_histories.npz"
    if not path.exists():
        print("  Skipping shift detection (no data)")
        return

    data = np.load(path)
    facp_shift = data["facp_shift"]

    fig, ax = plt.subplots(figsize=(8, 3))
    steps = np.arange(len(facp_shift))
    ax.fill_between(steps, facp_shift, alpha=0.3, color="#FF9800")
    ax.plot(steps, facp_shift, color="#FF9800", linewidth=1)
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"Estimated $\Delta$ (shift)")
    ax.set_title("Feedback-Induced Distribution Shift Detection (Theorem 3)")
    ax.grid(True, alpha=0.3)

    fig.savefig(FIG_DIR / "shift_detection.pdf")
    fig.savefig(FIG_DIR / "shift_detection.png")
    plt.close()
    print("  shift_detection.pdf")


def table_main_results():
    """Generate LaTeX table of main results."""
    baselines_path = TABLE_DIR / "baselines.json"
    conformal_path = TABLE_DIR / "conformal.json"
    if not baselines_path.exists() or not conformal_path.exists():
        print("  Skipping main results table (missing data)")
        return

    baselines_raw = json.loads(baselines_path.read_text())
    conformal = json.loads(conformal_path.read_text())

    # Support both structured and legacy flat format
    if isinstance(baselines_raw, dict) and "table1_scoring" in baselines_raw:
        baselines = baselines_raw["table1_scoring"]
    elif isinstance(baselines_raw, list):
        baselines = baselines_raw
    else:
        print("  Skipping main results table (unrecognized baselines format)")
        return

    lines = [
        r"\begin{tabular}{lccccr}",
        r"\toprule",
        r"Method & AUROC & F1 & Coverage & Unsafe\% & Deferral\% \\",
        r"\midrule",
    ]

    for bl in baselines:
        name = bl["baseline"].replace("_", r"\_")
        lines.append(
            f"{name} & {bl['auroc']:.3f} & {bl['f1']:.3f} & "
            f"{bl['coverage']:.3f} & {bl['unsafe_rate']:.3f} & "
            f"{bl['deferral_rate']:.3f} \\\\"
        )

    lines.append(r"\midrule")

    cp = conformal["split_cp"]
    lines.append(
        f"TrustLens (Split CP) & {cp['auroc']:.3f} & {cp['f1']:.3f} & "
        f"{cp['coverage']:.3f} & {1 - cp['recall']:.3f} & -- \\\\"
    )

    aci = conformal["aci"]
    lines.append(f"TrustLens (ACI) & -- & {aci['f1']:.3f} & {aci['coverage']:.3f} & -- & -- \\\\")

    facp = conformal["feedback_aware_cp"]
    lines.append(
        f"TrustLens (FA-CP) & -- & {facp['f1']:.3f} & {facp['coverage']:.3f} & -- & -- \\\\"
    )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    latex = "\n".join(lines)
    (TABLE_DIR / "main_results.tex").write_text(latex)
    print("  main_results.tex")


def fig_ablation_comparison():
    """Ablation comparison: grouped bar chart."""
    path = TABLE_DIR / "ablations.json"
    if not path.exists():
        print("  Skipping ablation comparison (no data)")
        return

    raw = json.loads(path.read_text())
    # Handle both old format (flat) and new format (with aggregate wrapper)
    data = raw.get("aggregate", raw) if isinstance(raw, dict) and "aggregate" in raw else raw
    multi_seed = raw.get("multi_seed", False) if isinstance(raw, dict) else False

    names = {
        "full": "Full\n(Trans+Args)",
        "no_transformer": "No Transformer\n(MLP only)",
        "no_args": "Tool-Only\n(no args)",
        "consequence_no_args": "Consequence\n(no args)",
        "consequence_full": "Consequence\n(+args)",
    }
    # Show all available configs from the data
    all_configs = ["full", "no_transformer", "no_args", "consequence_no_args", "consequence_full"]
    configs = [c for c in all_configs if c in data]
    labels = [names.get(c, c) for c in configs]

    if multi_seed:
        aurocs = [data[c]["auroc_mean"] for c in configs]
        auroc_err = [data[c]["auroc_std"] for c in configs]
        f1_cp = [data[c]["f1_cp_mean"] for c in configs]
        f1_cp_err = [data[c]["f1_cp_std"] for c in configs]
        f1_nocp = [data[c]["f1_nocp_mean"] for c in configs]
        f1_nocp_err = [data[c]["f1_nocp_std"] for c in configs]
    else:
        aurocs = [data[c]["auroc"] for c in configs]
        auroc_err = [0] * len(configs)
        f1_cp = [data[c]["f1_cp"] for c in configs]
        f1_cp_err = [0] * len(configs)
        f1_nocp = [data[c]["f1_nocp"] for c in configs]
        f1_nocp_err = [0] * len(configs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(configs))
    w = 0.35

    # AUROC
    axes[0].bar(x, aurocs, w, yerr=auroc_err, color="#2196F3", alpha=0.8, capsize=4)
    axes[0].set_title("AUROC by Configuration")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0.4, 0.8)
    axes[0].set_ylabel("AUROC")
    axes[0].grid(True, alpha=0.3, axis="y")

    # F1 (with and without CP)
    axes[1].bar(
        x - w / 2, f1_cp, w, yerr=f1_cp_err, label="With CP", color="#4CAF50", alpha=0.8, capsize=4
    )
    axes[1].bar(
        x + w / 2,
        f1_nocp,
        w,
        yerr=f1_nocp_err,
        label="Without CP",
        color="#FF9800",
        alpha=0.8,
        capsize=4,
    )
    axes[1].set_title("F1 Score: Conformal vs Raw Threshold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylim(0, 0.25)
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "ablation_comparison.pdf")
    fig.savefig(FIG_DIR / "ablation_comparison.png")
    plt.close()
    print("  ablation_comparison.pdf")


def fig_multiseed_summary():
    """Multi-seed bar chart with error bars."""
    path = TABLE_DIR / "multiseed.json"
    if not path.exists():
        print("  Skipping multi-seed summary (no data)")
        return

    data = json.loads(path.read_text())
    agg = data["aggregate"]
    metrics = ["auroc", "f1", "coverage", "recall"]
    labels = ["AUROC", "F1", "Coverage", "Recall"]

    means = [agg[m]["mean"] for m in metrics]
    stds = [agg[m]["std"] for m in metrics]
    ci_lo = [agg[m]["ci_lower"] for m in metrics]
    ci_hi = [agg[m]["ci_upper"] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="white")

    # CI annotations
    for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        ax.annotate(
            f"[{lo:.3f}, {hi:.3f}]",
            xy=(x[i], means[i] + stds[i] + 0.02),
            ha="center",
            fontsize=8,
            color="gray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Metric Value")
    ax.set_title(f"TrustLens (Split CP) — {len(data['seeds'])} Seeds, Mean $\\pm$ Std")
    ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.4, label="Target (0.95)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "multiseed_summary.pdf")
    fig.savefig(FIG_DIR / "multiseed_summary.png")
    plt.close()
    print("  multiseed_summary.pdf")


def fig_lipschitz_bound():
    """Lipschitz bound analysis: miscoverage bound vs n for various delta shifts."""
    path = TABLE_DIR / "conformal.json"
    if not path.exists():
        print("  Skipping Lipschitz bound (no data)")
        return

    data = json.loads(path.read_text())
    L = data.get("lipschitz_L", 120.0)
    alpha = 0.05
    conf_delta = 0.05  # confidence parameter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: bound vs n for various shift magnitudes
    ns = np.logspace(2, 5, 100)  # 100 to 100000
    shifts = [0.01, 0.05, 0.10, 0.20]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for shift, color in zip(shifts, colors):
        bound = alpha + L * shift / ns + np.sqrt(np.log(1.0 / conf_delta) / (2 * ns))
        axes[0].plot(ns, bound, label=f"$\\Delta$={shift}", color=color, linewidth=1.5)

    axes[0].axhline(y=1.0, color="black", linestyle=":", alpha=0.3, label="Vacuous (1.0)")
    axes[0].axhline(y=alpha, color="gray", linestyle="--", alpha=0.4, label=f"$\\alpha$={alpha}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Calibration set size $n$")
    axes[0].set_ylabel("Miscoverage bound")
    axes[0].set_title(f"Theorem 3 Bound (L={L:.1f})")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)

    # Right: minimum n for non-vacuous bound vs L
    Ls = np.linspace(1, 200, 100)
    shift_fixed = 0.10
    # Solve: alpha + L*delta/n + sqrt(log(1/d)/(2n)) < 1
    # Approximate: n_min ≈ L*delta / (1 - alpha - epsilon)
    n_mins = []
    for l_val in Ls:
        # Find smallest n where bound < 1
        for n_test in range(10, 100001, 10):
            b = alpha + l_val * shift_fixed / n_test
            b += np.sqrt(np.log(1.0 / conf_delta) / (2 * n_test))
            if b < 1.0:
                n_mins.append(n_test)
                break
        else:
            n_mins.append(100000)

    axes[1].plot(Ls, n_mins, color="#2196F3", linewidth=1.5)
    axes[1].axvline(x=L, color="red", linestyle="--", alpha=0.5, label=f"Our L={L:.0f}")
    axes[1].set_xlabel("Lipschitz constant $L$")
    axes[1].set_ylabel("Min $n$ for non-vacuous bound")
    axes[1].set_title(f"Required $n$ vs $L$ ($\\Delta$={shift_fixed})")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "lipschitz_bound.pdf")
    fig.savefig(FIG_DIR / "lipschitz_bound.png")
    plt.close()
    print("  lipschitz_bound.pdf")


def fig_cal_sweep():
    """Cal set size sweep: coverage and bound slack vs cal set size."""
    path = TABLE_DIR / "cal_sweep.json"
    if not path.exists():
        print("  Skipping cal sweep (no data)")
        return

    data = json.loads(path.read_text())
    cal_sizes = [d["cal_size"] for d in data]
    cov_means = [d["coverage_mean"] for d in data]
    cov_stds = [d["coverage_std"] for d in data]
    bound_slacks = [d["bound_slack"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: Coverage vs cal size
    axes[0].errorbar(
        cal_sizes, cov_means, yerr=cov_stds, fmt="o-", color="#2196F3", capsize=4, markersize=6
    )
    axes[0].axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label=r"Target $1-\alpha=0.95$")
    axes[0].set_xlabel("Calibration set size $n$")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Coverage Guarantee Holds at All Sizes")
    axes[0].legend()
    axes[0].set_ylim(0.92, 1.01)
    axes[0].grid(True, alpha=0.3)

    # Right: Bound slack (tightness) vs cal size
    axes[1].bar(range(len(cal_sizes)), bound_slacks, color="#4CAF50", alpha=0.8, edgecolor="white")
    axes[1].set_xticks(range(len(cal_sizes)))
    axes[1].set_xticklabels([str(n) for n in cal_sizes])
    axes[1].set_xlabel("Calibration set size $n$")
    axes[1].set_ylabel("Bound slack (coverage $- (1-\\alpha)$)")
    axes[1].set_title("Bound Tightens with More Calibration Data")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cal_sweep.pdf")
    fig.savefig(FIG_DIR / "cal_sweep.png")
    plt.close()
    print("  cal_sweep.pdf")


def fig_cp_comparison():
    """Scorer-agnostic CP comparison: coverage holds for multiple scorers (multi-seed)."""
    path = TABLE_DIR / "cp_comparison.json"
    if not path.exists():
        print("  Skipping CP comparison (no data)")
        return

    raw = json.loads(path.read_text())

    # Handle both v1 (flat list) and v2 (aggregate+per_seed) formats
    if isinstance(raw, list):
        # v1 format
        names = [d["scorer"] for d in raw]
        aurocs = [d["auroc"] for d in raw]
        auroc_errs = [0.0] * len(raw)
        f1s = [d["f1"] for d in raw]
        f1_errs = [0.0] * len(raw)
        coverages = [d["coverage"] for d in raw]
        cov_errs = [0.0] * len(raw)
        bound_fracs = [1.0 if d["bound_holds"] else 0.0 for d in raw]
    else:
        # v2 format (multi-seed aggregate)
        agg = raw["aggregate"]
        names = sorted(agg.keys())
        aurocs = [agg[n]["auroc_mean"] for n in names]
        auroc_errs = [agg[n]["auroc_std"] for n in names]
        f1s = [agg[n]["f1_mean"] for n in names]
        f1_errs = [agg[n]["f1_std"] for n in names]
        coverages = [agg[n]["coverage_mean"] for n in names]
        cov_errs = [agg[n]["coverage_std"] for n in names]
        bound_fracs = [agg[n]["bound_holds_fraction"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(names))
    w = 0.55

    # Separate primary (mixed-cal) and normal-only ablation scorers
    has_ablation = any("(normal-only cal)" in n for n in names)

    # Primary figure: mixed-cal scorers (default), with normal-only ablation rows
    if has_ablation:
        ablation_names = [n for n in names if "(normal-only cal)" in n]
        primary_names = [n for n in names if "(normal-only cal)" not in n]
        # Lead with primary (mixed-cal), then ablation (normal-only)
        display_names = primary_names + ablation_names
        display_aurocs, display_auroc_errs = [], []
        display_f1s, display_f1_errs = [], []
        display_coverages, display_cov_errs = [], []
        display_bound_fracs = []
        for n in display_names:
            idx = names.index(n)
            display_aurocs.append(aurocs[idx])
            display_auroc_errs.append(auroc_errs[idx])
            display_f1s.append(f1s[idx])
            display_f1_errs.append(f1_errs[idx])
            display_coverages.append(coverages[idx])
            display_cov_errs.append(cov_errs[idx])
            display_bound_fracs.append(bound_fracs[idx])
        names = display_names
        aurocs = display_aurocs
        auroc_errs = display_auroc_errs
        f1s = display_f1s
        f1_errs = display_f1_errs
        coverages = display_coverages
        cov_errs = display_cov_errs
        bound_fracs = display_bound_fracs
        cal_label = "mixed-cal primary + normal-only ablation"
    else:
        cal_label = "mixed-cal"

    x = np.arange(len(names))

    # Colors: green if bound holds >=80%, yellow if partial, red if fails
    colors_cov = []
    for bf in bound_fracs:
        if bf >= 0.8:
            colors_cov.append("#4CAF50")
        elif bf >= 0.4:
            colors_cov.append("#FF9800")
        else:
            colors_cov.append("#F44336")

    # AUROC
    axes[0].bar(
        x, aurocs, w, yerr=auroc_errs, capsize=4, color="#2196F3", alpha=0.8, edgecolor="white"
    )
    axes[0].set_title("AUROC (Scorer Quality)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("AUROC")
    axes[0].grid(True, alpha=0.3, axis="y")

    # F1 under CP
    axes[1].bar(x, f1s, w, yerr=f1_errs, capsize=4, color="#4CAF50", alpha=0.8, edgecolor="white")
    axes[1].set_title("F1 Score (under Split CP)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylim(0, 0.4)
    axes[1].set_ylabel("F1")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Coverage with bound indicator
    axes[2].bar(
        x, coverages, w, yerr=cov_errs, capsize=4, color=colors_cov, alpha=0.8, edgecolor="white"
    )
    axes[2].axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label=r"Target $1-\alpha=0.95$")
    cov_title = "Coverage (Mixed Cal — Marginal Guarantee)" if has_ablation else "Coverage"
    axes[2].set_title(cov_title)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[2].set_ylim(0.9, 1.0)
    axes[2].set_ylabel("Coverage")
    axes[2].legend(loc="lower left", fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, bf in enumerate(bound_fracs):
        pct = f"{bf * 100:.0f}%"
        axes[2].annotate(
            pct,
            xy=(x[i], coverages[i] + 0.003),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#2E7D32" if bf >= 0.8 else "#C62828",
        )

    if has_ablation:
        axes[2].annotate(
            "Mixed cal (primary) provides marginal guarantee (Theorem 1);\n"
            "normal-only cal shown as ablation.",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7,
            color="gray",
            style="italic",
        )

    n_seeds = raw.get("config", {}).get("n_seeds", "?") if isinstance(raw, dict) else 1
    plt.suptitle(
        f"Scorer-Agnostic CP ({n_seeds} seeds, {cal_label} calibration)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cp_comparison.pdf")
    fig.savefig(FIG_DIR / "cp_comparison.png")
    plt.close()
    print("  cp_comparison.pdf")


def fig_shift_experiment():
    """Distribution shift experiment: windowed coverage for Split CP / ACI / FA-CP."""
    path = TABLE_DIR / "shift_experiment.json"
    if not path.exists():
        print("  Skipping shift experiment (no data)")
        return

    data = json.loads(path.read_text())
    config = data["config"]
    shift_point = config["shift_point"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: windowed coverage comparison
    for method_key, label, color in [
        ("split_cp", "Split CP", "#2196F3"),
        ("aci", "ACI", "#4CAF50"),
        ("facp", "FA-CP", "#FF9800"),
    ]:
        windows = data[method_key]["windows"]
        centers = [(w["start"] + w["end"]) / 2 for w in windows]
        covs = [w["coverage"] for w in windows]
        axes[0].plot(centers, covs, "o-", label=label, color=color, markersize=4, linewidth=1.2)

    axes[0].axvline(
        x=shift_point, color="red", linestyle="--", alpha=0.6, label=f"Shift at {shift_point}"
    )
    axes[0].axhline(y=0.95, color="gray", linestyle=":", alpha=0.5, label=r"Target $1-\alpha=0.95$")
    axes[0].set_xlabel("Trace index")
    axes[0].set_ylabel("Windowed coverage")
    axes[0].set_title(
        f"Coverage Under Distribution Shift "
        f"({config['base_rate']:.0%}→{config['shifted_rate']:.0%})"
    )
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Right: tau adaptation
    tau_path = TABLE_DIR / "shift_tau_histories.npz"
    if tau_path.exists():
        hist = np.load(tau_path)
        aci_tau = hist["aci_tau"]
        facp_tau = hist["facp_tau"]
        steps = np.arange(len(aci_tau))
        axes[1].plot(steps, aci_tau, label="ACI", color="#4CAF50", alpha=0.8, linewidth=1)
        axes[1].plot(
            steps[: len(facp_tau)], facp_tau, label="FA-CP", color="#FF9800", alpha=0.8, linewidth=1
        )
        axes[1].axvline(x=shift_point, color="red", linestyle="--", alpha=0.6)
        axes[1].set_xlabel("Trace index")
        axes[1].set_ylabel(r"Threshold $\tau_t$")
        axes[1].set_title("Threshold Adaptation Under Shift")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "shift_experiment.pdf")
    fig.savefig(FIG_DIR / "shift_experiment.png")
    plt.close()
    print("  shift_experiment.pdf")


def fig_held_out_attacks():
    """Per-pattern AUROC for seen vs unseen attack patterns."""
    path = TABLE_DIR / "held_out_attacks.json"
    if not path.exists():
        print("  Skipping held-out attacks (no data)")
        return

    data = json.loads(path.read_text())
    patterns = data["per_pattern"]

    names = [p["pattern_name"] for p in patterns]
    aurocs = [p["auroc"] for p in patterns]
    types = [p["type"] for p in patterns]
    colors = ["#2196F3" if t == "seen" else "#F44336" for t in types]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, aurocs, 0.6, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, label="Random (0.5)")

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2196F3", alpha=0.8, label="Seen (train)"),
        Patch(facecolor="#F44336", alpha=0.8, label="Unseen (held-out)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Pattern Detection: Seen vs Held-Out Attack Patterns")
    ax.set_ylim(0.4, 0.9)
    ax.grid(True, alpha=0.3, axis="y")

    summary = data["summary"]
    ax.axhline(
        y=summary["seen_auroc_mean"], color="#2196F3", linestyle="--", alpha=0.5, linewidth=0.8
    )
    ax.axhline(
        y=summary["unseen_auroc_mean"], color="#F44336", linestyle="--", alpha=0.5, linewidth=0.8
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "held_out_attacks.pdf")
    fig.savefig(FIG_DIR / "held_out_attacks.png")
    plt.close()
    print("  held_out_attacks.pdf")


def fig_weight_sweep():
    """Consequence-aware encoder: AUROC / F1 / coverage vs outcome_weight."""
    path = TABLE_DIR / "weight_sweep.json"
    if not path.exists():
        print("  Skipping weight sweep (no data)")
        return

    raw = json.loads(path.read_text())
    agg = raw["aggregate"]
    weights_cfg = raw["config"]["outcome_weights"]

    names = sorted(
        [k for k in agg if "(mixed cal)" not in k],
        key=lambda k: agg[k].get("auroc_mean", 0),
    )

    # Sort by weight value for x-axis
    def _extract_weight(name: str) -> float:
        if "w=" in name:
            return float(name.split("w=")[1].rstrip(")"))
        return 0.5

    names = sorted(names, key=_extract_weight)
    x_labels = [f"w={_extract_weight(n)}" for n in names]

    aurocs = [agg[n]["auroc_mean"] for n in names]
    auroc_errs = [agg[n]["auroc_std"] for n in names]
    f1s = [agg[n]["f1_mean"] for n in names]
    f1_errs = [agg[n]["f1_std"] for n in names]
    coverages = [agg[n]["coverage_mean"] for n in names]
    cov_errs = [agg[n]["coverage_std"] for n in names]
    bound_fracs = [agg[n]["bound_holds_fraction"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(names))
    w = 0.55

    axes[0].bar(
        x, aurocs, w, yerr=auroc_errs, capsize=4, color="#2196F3", alpha=0.8, edgecolor="white"
    )
    axes[0].set_title("AUROC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, f1s, w, yerr=f1_errs, capsize=4, color="#4CAF50", alpha=0.8, edgecolor="white")
    axes[1].set_title("F1 Score (under CP)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_ylabel("F1")
    axes[1].set_ylim(0, 0.7)
    axes[1].grid(True, alpha=0.3, axis="y")

    colors_cov = [
        "#4CAF50" if bf >= 0.8 else "#FF9800" if bf >= 0.4 else "#F44336" for bf in bound_fracs
    ]
    axes[2].bar(
        x, coverages, w, yerr=cov_errs, capsize=4, color=colors_cov, alpha=0.8, edgecolor="white"
    )
    axes[2].axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label=r"Target $1-\alpha=0.95$")
    axes[2].set_title("Coverage (mixed cal)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels)
    axes[2].set_ylabel("Coverage")
    axes[2].set_ylim(0.9, 1.0)
    axes[2].legend(loc="lower left", fontsize=8)
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, bf in enumerate(bound_fracs):
        pct = f"{bf * 100:.0f}%"
        axes[2].annotate(
            pct,
            xy=(x[i], coverages[i] + 0.003),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#2E7D32" if bf >= 0.8 else "#C62828",
        )

    plt.suptitle(
        "Consequence-Aware Encoder: Outcome Weight Sweep",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "weight_sweep.pdf")
    fig.savefig(FIG_DIR / "weight_sweep.png")
    plt.close()
    print("  weight_sweep.pdf")


def fig_cal_sweep_consequence():
    """Cal set size sweep for consequence-aware encoder."""
    path = TABLE_DIR / "cal_sweep_consequence.json"
    if not path.exists():
        print("  Skipping cal sweep consequence (no data)")
        return

    raw = json.loads(path.read_text())
    data = raw.get("results", raw)
    if isinstance(data, dict):
        data = list(data.values())

    cal_sizes = [d["cal_size"] for d in data]
    cov_means = [d["coverage_mean"] for d in data]
    cov_stds = [d["coverage_std"] for d in data]
    bound_slacks = [d["bound_slack"] for d in data]
    bound_holds = [d.get("bound_holds_frac", 1.0 if d["bound_holds"] else 0.0) for d in data]

    config = raw.get("config", {})
    scorer = config.get("scorer", "consequence-aware")
    cal_mode = config.get("calibration", "normal-only")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: Coverage vs cal size
    axes[0].errorbar(
        cal_sizes, cov_means, yerr=cov_stds, fmt="o-", color="#9C27B0", capsize=4, markersize=6
    )
    axes[0].axhline(y=0.95, color="red", linestyle="--", alpha=0.5, label=r"Target $1-\alpha=0.95$")
    axes[0].set_xlabel("Calibration set size $n$")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title(f"Coverage vs Cal Size ({scorer}, {cal_mode})")
    axes[0].legend()
    axes[0].set_ylim(0.90, 1.01)
    axes[0].grid(True, alpha=0.3)

    # Right: Bound holds fraction vs cal size
    colors = [
        "#4CAF50" if bh >= 0.8 else "#FF9800" if bh >= 0.4 else "#F44336" for bh in bound_holds
    ]
    axes[1].bar(
        range(len(cal_sizes)),
        [bh * 100 for bh in bound_holds],
        color=colors,
        alpha=0.8,
        edgecolor="white",
    )
    axes[1].set_xticks(range(len(cal_sizes)))
    axes[1].set_xticklabels([str(n) for n in cal_sizes])
    axes[1].set_xlabel("Calibration set size $n$")
    axes[1].set_ylabel("Bound holds (% of trials)")
    axes[1].set_title("Bound Satisfaction vs Cal Size")
    axes[1].axhline(y=80, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylim(0, 110)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "cal_sweep_consequence.pdf")
    fig.savefig(FIG_DIR / "cal_sweep_consequence.png")
    plt.close()
    print("  cal_sweep_consequence.pdf")


def fig_consequence_comparison():
    """Compare sequence-only vs consequence-aware encoder under CP."""
    path = TABLE_DIR / "cp_comparison.json"
    if not path.exists():
        print("  Skipping consequence comparison (no data)")
        return

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "aggregate" not in raw:
        print("  Skipping consequence comparison (old format)")
        return

    agg = raw["aggregate"]
    if "Consequence-Aware Encoder" not in agg:
        print("  Skipping consequence comparison (no consequence scorer)")
        return

    scorers = ["Transformer Encoder", "Consequence-Aware Encoder", "LSTM Next-Token"]
    present = [s for s in scorers if s in agg]
    if len(present) < 2:
        print("  Skipping consequence comparison (insufficient scorers)")
        return

    names = present
    aurocs = [agg[n]["auroc_mean"] for n in names]
    auroc_errs = [agg[n]["auroc_std"] for n in names]
    f1s = [agg[n]["f1_mean"] for n in names]
    f1_errs = [agg[n]["f1_std"] for n in names]
    bound_fracs = [agg[n]["bound_holds_fraction"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(names))
    w = 0.5
    colors = ["#2196F3", "#9C27B0", "#4CAF50"][: len(names)]

    axes[0].bar(
        x, aurocs, w, yerr=auroc_errs, capsize=4, color=colors, alpha=0.8, edgecolor="white"
    )
    axes[0].set_title("AUROC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylim(0.5, 0.85)
    axes[0].set_ylabel("AUROC")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, f1s, w, yerr=f1_errs, capsize=4, color=colors, alpha=0.8, edgecolor="white")
    axes[1].set_title("F1 Score (under Split CP)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylim(0, 0.4)
    axes[1].set_ylabel("F1")
    axes[1].grid(True, alpha=0.3, axis="y")

    colors_bound = []
    for bf in bound_fracs:
        if bf >= 0.8:
            colors_bound.append("#4CAF50")
        elif bf >= 0.4:
            colors_bound.append("#FF9800")
        else:
            colors_bound.append("#F44336")

    axes[2].bar(
        x, [bf * 100 for bf in bound_fracs], w, color=colors_bound, alpha=0.8, edgecolor="white"
    )
    axes[2].set_title("Coverage Bound Holds (%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[2].set_ylim(0, 110)
    axes[2].set_ylabel("% of Seeds")
    axes[2].axhline(y=80, color="gray", linestyle="--", alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Sequence-Only vs Consequence-Aware Encoder",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "consequence_comparison.pdf")
    fig.savefig(FIG_DIR / "consequence_comparison.png")
    plt.close()
    print("  consequence_comparison.pdf")


def fig_trail_transfer():
    """TRAIL transfer evaluation: grouped bar chart by mode and scorer."""
    path = TABLE_DIR / "trail_transfer.json"
    if not path.exists():
        print("  Skipping TRAIL transfer (no data)")
        return

    raw = json.loads(path.read_text())
    agg = raw["aggregate"]
    config = raw["config"]

    modes = config["modes"]
    mode_labels = {
        "zero_shot": "Zero-Shot",
        "fine_tuned": "Fine-Tuned",
        "native": "Native",
    }
    scorer_colors = {
        "Transformer": "#2196F3",
        "Consequence-Aware": "#9C27B0",
        "LSTM": "#4CAF50",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    all_scorers = sorted(set(v["scorer"] for v in agg.values()))
    n_modes = len(modes)
    n_scorers = len(all_scorers)
    bar_width = 0.8 / max(n_scorers, 1)

    for j, scorer in enumerate(all_scorers):
        x_positions = []
        heights = []
        errors = []
        for i, mode in enumerate(modes):
            key = f"{mode}/{scorer}"
            if key in agg:
                x_positions.append(i + j * bar_width - 0.4 + bar_width / 2)
                heights.append(agg[key]["auroc_mean"])
                errors.append(agg[key]["auroc_std"])

        color = scorer_colors.get(scorer, "#757575")
        ax.bar(
            x_positions,
            heights,
            bar_width,
            yerr=errors,
            capsize=3,
            color=color,
            alpha=0.85,
            edgecolor="white",
            label=scorer,
        )

    ax.set_xticks(range(n_modes))
    ax.set_xticklabels([mode_labels.get(m, m) for m in modes])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(
        f"TRAIL Transfer ({config['n_trail_traces']} real traces, {len(config['seeds'])} seeds)",
    )
    ax.legend(loc="upper right")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4, label="Random")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "trail_transfer.pdf")
    fig.savefig(FIG_DIR / "trail_transfer.png")
    plt.close()
    print("  trail_transfer.pdf")


def fig_toolbench_transfer():
    """ToolBench transfer: 3-bar chart with leakage check (tool-only, CQ, CQ-shuffled)."""
    path = TABLE_DIR / "toolbench_transfer.json"
    if not path.exists():
        print("  Skipping ToolBench transfer (no data)")
        return

    raw = json.loads(path.read_text())
    per_seed = raw["per_seed"]
    agg = raw["aggregate"]

    has_shuffled = "cq_shuffled_auroc" in agg

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: per-seed AUROC comparison
    ax = axes[0]
    seeds = [r["seed"] for r in per_seed]
    tool_aurocs = [r["tool_only"]["auroc"] for r in per_seed]
    cq_aurocs = [r["consequence_aware"]["auroc"] for r in per_seed]

    x = np.arange(len(seeds))
    if has_shuffled:
        shuf_aurocs = [r["cq_shuffled"]["auroc"] for r in per_seed]
        w = 0.25
        ax.bar(x - w, tool_aurocs, w, label="Tool-Only", color="#2196F3", alpha=0.85)
        ax.bar(x, cq_aurocs, w, label="CQ", color="#9C27B0", alpha=0.85)
        ax.bar(x + w, shuf_aurocs, w, label="CQ Shuffled", color="#FF9800", alpha=0.85)
    else:
        w = 0.35
        ax.bar(x - w / 2, tool_aurocs, w, label="Tool-Only", color="#2196F3", alpha=0.85)
        ax.bar(x + w / 2, cq_aurocs, w, label="CQ", color="#9C27B0", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=9)
    ax.set_xlabel("Seed")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Per-Seed AUROC on ToolBench")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: aggregate with CIs
    ax2 = axes[1]
    tool_ci = agg["tool_only_auroc"]
    cq_ci = agg["consequence_auroc"]

    if has_shuffled:
        shuf_ci = agg["cq_shuffled_auroc"]
        means = [tool_ci["mean"], cq_ci["mean"], shuf_ci["mean"]]
        ci_lo = [m - agg[k]["ci_lower"] for m, k in zip(
            means, ["tool_only_auroc", "consequence_auroc", "cq_shuffled_auroc"]
        )]
        ci_hi = [agg[k]["ci_upper"] - m for m, k in zip(
            means, ["tool_only_auroc", "consequence_auroc", "cq_shuffled_auroc"]
        )]
        colors = ["#2196F3", "#9C27B0", "#FF9800"]
        labels = ["Tool-Only", "CQ", "CQ Shuffled"]
        positions = [0, 1, 2]
    else:
        means = [tool_ci["mean"], cq_ci["mean"]]
        ci_lo = [tool_ci["mean"] - tool_ci["ci_lower"],
                 cq_ci["mean"] - cq_ci["ci_lower"]]
        ci_hi = [tool_ci["ci_upper"] - tool_ci["mean"],
                 cq_ci["ci_upper"] - cq_ci["mean"]]
        colors = ["#2196F3", "#9C27B0"]
        labels = ["Tool-Only", "CQ"]
        positions = [0, 1]

    ax2.bar(
        positions, means,
        yerr=[ci_lo, ci_hi],
        capsize=5, color=colors, alpha=0.85, edgecolor="white",
    )
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("AUROC (95% CI)")
    ax2.set_ylim(0.4, 1.05)

    delta = agg["auroc_paired_test"]["difference_mean"]
    sig = agg["auroc_paired_test"].get("significant_by_ci", False)
    sig_marker = "*" if sig else ""
    ax2.set_title(
        f"Aggregate AUROC ({len(per_seed)} seeds, "
        f"$\\Delta$={delta:+.3f}{sig_marker})"
    )
    ax2.grid(True, alpha=0.3, axis="y")

    if has_shuffled and "leakage_fraction" in agg:
        lf = agg["leakage_fraction"]
        ax2.annotate(
            f"Leakage: {lf['mean']:.0%} "
            f"[{lf['ci_lower']:.0%}, {lf['ci_upper']:.0%}]\n"
            f"CQ advantage is outcome leakage",
            xy=(0.02, 0.02), xycoords="axes fraction",
            fontsize=8, color="#C62828", style="italic",
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "toolbench_transfer.pdf")
    fig.savefig(FIG_DIR / "toolbench_transfer.png")
    plt.close()
    print("  toolbench_transfer.pdf")


def fig_agentharm_eval():
    """AgentHarm evaluation: paired bar chart with real adversarial labels."""
    path = TABLE_DIR / "agentharm_eval.json"
    if not path.exists():
        print("  Skipping AgentHarm eval (no data)")
        return

    raw = json.loads(path.read_text())
    per_seed = raw["per_seed"]
    agg = raw["aggregate"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: per-seed AUROC
    ax = axes[0]
    seeds = [r["seed"] for r in per_seed]
    tool_aurocs = [r["tool_only"]["auroc"] for r in per_seed]
    cq_aurocs = [r["consequence_aware"]["auroc"] for r in per_seed]

    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w / 2, tool_aurocs, w, label="Tool-Only", color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, cq_aurocs, w, label="Consequence-Aware", color="#9C27B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=9)
    ax.set_xlabel("Seed")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Per-Seed AUROC on AgentHarm (Real Labels)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: aggregate with CIs
    ax2 = axes[1]
    tool_ci = agg["tool_only_auroc"]
    cq_ci = agg["consequence_auroc"]

    bars_agg = ax2.bar(
        [0, 1],
        [tool_ci["mean"], cq_ci["mean"]],
        yerr=[
            [tool_ci["mean"] - tool_ci["ci_lower"], cq_ci["mean"] - cq_ci["ci_lower"]],
            [tool_ci["ci_upper"] - tool_ci["mean"], cq_ci["ci_upper"] - cq_ci["mean"]],
        ],
        capsize=5,
        color=["#2196F3", "#9C27B0"],
        alpha=0.85,
        edgecolor="white",
    )
    for bar, ci in zip(bars_agg, [tool_ci, cq_ci]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{ci['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Tool-Only", "Consequence-Aware"])
    ax2.set_ylabel("AUROC (95% CI)")
    ax2.set_ylim(0.0, 1.15)

    delta = agg["auroc_paired_test"]["difference_mean"]
    sig = agg["auroc_paired_test"].get("significant_by_ci", False)
    sig_marker = "*" if sig else ""
    ax2.set_title(
        f"AgentHarm Aggregate ({raw['n_traces']} traces, "
        f"$\\Delta$={delta:+.3f}{sig_marker})"
    )
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "agentharm_eval.pdf")
    fig.savefig(FIG_DIR / "agentharm_eval.png")
    plt.close()
    print("  agentharm_eval.pdf")


def fig_coverage_contamination():
    """Theoretical coverage bounds vs contamination rate (Theorem 4)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    alpha = 0.05

    # Left: Coverage bounds vs contamination rate
    epsilons = np.linspace(0, 0.5, 100)
    cov_lower = (1.0 - alpha) * (1.0 - epsilons)
    cov_upper = (1.0 - alpha) * (1.0 - epsilons) + epsilons
    cov_no_detect = np.full_like(epsilons, 1.0 - alpha)

    axes[0].fill_between(
        epsilons,
        cov_lower,
        cov_upper,
        alpha=0.2,
        color="#2196F3",
        label="Achievable range",
    )
    axes[0].plot(
        epsilons,
        cov_lower,
        color="#2196F3",
        linewidth=1.5,
        label="Perfect detector (lower)",
    )
    axes[0].plot(
        epsilons,
        cov_upper,
        color="#2196F3",
        linewidth=1.5,
        linestyle="--",
        label="No detection (upper)",
    )
    axes[0].plot(
        epsilons,
        cov_no_detect,
        color="gray",
        linewidth=1,
        linestyle=":",
        label=f"Standard CP (1-$\\alpha$={1 - alpha})",
    )

    for auroc, color, ls in [(0.9, "#F44336", "-"), (0.75, "#FF9800", "-"), (0.6, "#4CAF50", "--")]:
        cov = (1.0 - alpha) * (1.0 - epsilons) + epsilons * (1.0 - auroc)
        axes[0].plot(
            epsilons,
            cov,
            color=color,
            linewidth=1.2,
            linestyle=ls,
            label=f"AUROC={auroc}",
        )

    axes[0].set_xlabel(r"Contamination rate $\varepsilon$")
    axes[0].set_ylabel("Marginal coverage")
    axes[0].set_title(
        f"Theorem 4: Coverage Under Contamination ($\\alpha$={alpha})",
    )
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].grid(True, alpha=0.3)

    # Right: Coverage-AUROC tradeoff at fixed contamination rates
    aurocs = np.linspace(0.5, 1.0, 100)
    eps_values = [0.05, 0.10, 0.20, 0.30]
    colors_eps = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    for eps, color in zip(eps_values, colors_eps):
        cov = (1.0 - alpha) * (1.0 - eps) + eps * (1.0 - aurocs)
        axes[1].plot(
            aurocs,
            cov,
            color=color,
            linewidth=1.5,
            label=f"$\\varepsilon$={eps}",
        )

    axes[1].axhline(
        y=1.0 - alpha,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label=f"1-$\\alpha$={1 - alpha}",
    )
    axes[1].set_xlabel("Detector AUROC")
    axes[1].set_ylabel("Expected marginal coverage")
    axes[1].set_title("Coverage-AUROC Tradeoff (Corollary 4.1)")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "coverage_contamination.pdf")
    fig.savefig(FIG_DIR / "coverage_contamination.png")
    plt.close()
    print("  coverage_contamination.pdf")


def fig_rac_comparison():
    """RAC two-threshold vs standard CP: VaR cost and decision zone analysis."""
    path = TABLE_DIR / "rac_comparison.json"
    if not path.exists():
        print("  Skipping rac_comparison (no data)")
        return

    data = json.loads(path.read_text())
    agg = data["aggregate"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: VaR cost comparison (bar chart with CIs)
    rac = agg["rac_default"]
    labels = ["Binary CP", "RAC (3-way)"]
    means = [rac["binary_var_cost"]["mean"], rac["var_cost"]["mean"]]
    ci_lo = [
        means[0] - rac["binary_var_cost"]["ci_lower"],
        means[1] - rac["var_cost"]["ci_lower"],
    ]
    ci_hi = [
        rac["binary_var_cost"]["ci_upper"] - means[0],
        rac["var_cost"]["ci_upper"] - means[1],
    ]
    colors = ["#d62728", "#2ca02c"]
    bars = axes[0].bar(labels, means, color=colors, alpha=0.8, width=0.5)
    axes[0].errorbar(
        labels, means, yerr=[ci_lo, ci_hi], fmt="none", color="black", capsize=5
    )
    for bar, m in zip(bars, means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_ylabel("VaR Cost (lower is better)")
    axes[0].set_title("VaR Cost: Binary vs RAC")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: Error rates (FNR, FPR, Defer)
    metrics = ["FNR", "FPR", "Defer Rate"]
    vals = [rac["fnr"]["mean"], rac["fpr"]["mean"], rac["defer_rate"]["mean"]]
    stds = [rac["fnr"]["std"], rac["fpr"]["std"], rac["defer_rate"]["std"]]
    bar_colors = ["#ff7f0e", "#1f77b4", "#9467bd"]
    bars = axes[1].bar(metrics, vals, color=bar_colors, alpha=0.8, width=0.5)
    axes[1].errorbar(metrics, vals, yerr=stds, fmt="none", color="black", capsize=5)
    for bar, v in zip(bars, vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_ylabel("Rate")
    axes[1].set_title("RAC Error Rates")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel 3: Per-seed VaR improvement
    per_seed = data.get("per_seed", [])
    if per_seed:
        seeds = [r["seed"] for r in per_seed]
        rac_vars = []
        bin_vars = []
        for r in per_seed:
            best = r.get("best_rac", r.get("rac_sweep", [{}])[0])
            rac_vars.append(best.get("var_cost", 0))
            bin_vars.append(best.get("binary_var", 0))
        x = np.arange(len(seeds))
        w = 0.35
        axes[2].bar(x - w / 2, bin_vars, w, label="Binary", color="#d62728", alpha=0.8)
        axes[2].bar(x + w / 2, rac_vars, w, label="RAC", color="#2ca02c", alpha=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels([str(s) for s in seeds], fontsize=8)
        axes[2].set_xlabel("Seed")
        axes[2].set_ylabel("VaR Cost")
        axes[2].set_title("Per-Seed VaR Comparison")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "rac_comparison.pdf")
    fig.savefig(FIG_DIR / "rac_comparison.png")
    plt.close()
    print("  rac_comparison.pdf")


def fig_per_tool_coverage():
    """Per-tool conditional coverage across tool categories."""
    path = TABLE_DIR / "per_tool_coverage.json"
    if not path.exists():
        print("  Skipping per_tool_coverage (no data)")
        return

    data = json.loads(path.read_text())
    agg = data["aggregate"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Per-category coverage (localized vs global)
    cats = agg.get("per_category", {})
    if cats:
        cat_names = sorted(cats.keys())
        # Handle both flat (_mean suffix) and nested (dict with "mean" key) formats
        first_val = cats[cat_names[0]]
        if "localized_coverage_mean" in first_val:
            loc_covs = [cats[c]["localized_coverage_mean"] for c in cat_names]
            glob_covs = [cats[c]["global_coverage_mean"] for c in cat_names]
            counts = [cats[c]["mean_n_traces"] for c in cat_names]
        else:
            loc_covs = [cats[c]["localized_coverage"]["mean"] for c in cat_names]
            glob_covs = [cats[c]["global_coverage"]["mean"] for c in cat_names]
            counts = [cats[c]["count"]["mean"] for c in cat_names]
    else:
        per_seed = data.get("per_seed", [])
        if per_seed and "per_category" in per_seed[0]:
            cat_names = sorted(per_seed[0]["per_category"].keys())
            loc_covs = []
            glob_covs = []
            counts = []
            for cat in cat_names:
                locs = [
                    s["per_category"][cat]["localized_coverage"]
                    for s in per_seed
                    if cat in s.get("per_category", {})
                ]
                globs = [
                    s["per_category"][cat]["global_coverage"]
                    for s in per_seed
                    if cat in s.get("per_category", {})
                ]
                ns = [
                    s["per_category"][cat]["count"]
                    for s in per_seed
                    if cat in s.get("per_category", {})
                ]
                loc_covs.append(np.mean(locs) if locs else 0)
                glob_covs.append(np.mean(globs) if globs else 0)
                counts.append(np.mean(ns) if ns else 0)
        else:
            cat_names, loc_covs, glob_covs, counts = [], [], [], []

    if cat_names:
        x = np.arange(len(cat_names))
        w = 0.35
        axes[0].bar(
            x - w / 2,
            loc_covs,
            w,
            label="Localized (per-tool)",
            color="#2ca02c",
            alpha=0.8,
        )
        axes[0].bar(
            x + w / 2,
            glob_covs,
            w,
            label="Global (marginal)",
            color="#1f77b4",
            alpha=0.8,
        )
        axes[0].axhline(y=0.95, color="red", linestyle="--", alpha=0.7, label="95% target")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cat_names, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Coverage")
        axes[0].set_title("Per-Tool Coverage: Localized vs Global")
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(0.85, 1.02)
        axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: Coverage improvement (localized - global) vs sample count
    if cat_names:
        diffs = [l - g for l, g in zip(loc_covs, glob_covs)]
        sizes = [max(c, 1) for c in counts]
        colors_map = ["#2ca02c" if d >= 0 else "#d62728" for d in diffs]
        axes[1].scatter(sizes, diffs, c=colors_map, s=80, alpha=0.8, edgecolors="black")
        for i, name in enumerate(cat_names):
            axes[1].annotate(
                name,
                (sizes[i], diffs[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Category Sample Count")
        axes[1].set_ylabel("Coverage Improvement (Loc - Glob)")
        axes[1].set_title("Localized Coverage Improvement vs Category Size")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "per_tool_coverage.pdf")
    fig.savefig(FIG_DIR / "per_tool_coverage.png")
    plt.close()
    print("  per_tool_coverage.pdf")


def fig_speculative_regret():
    """Speculative vs conservative execution: regret, latency, rate sweep, alpha sweep.

    v12: dual-threshold design. Speculative uses permissive tau_spec,
    conservative uses strict tau_cons.
    """
    path = TABLE_DIR / "speculative_execution.json"
    if not path.exists():
        print("  Skipping speculative_regret (no data)")
        return

    data = json.loads(path.read_text())
    agg = data["aggregate"]
    dc = agg["default_config"]
    lt = agg["latency"]
    config = data.get("config", {})

    # v12: 2x3 layout to include alpha sweep panel
    alpha_sweep = agg.get("alpha_sweep", [])
    n_rows = 2
    n_cols = 3 if alpha_sweep else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    if n_cols == 2:
        # Ensure consistent indexing
        axes = axes.reshape(n_rows, n_cols)

    # Panel (a): Regret comparison (error-only, commit_reward=0)
    labels = ["Speculative", "Conservative"]
    means = [dc["speculative_regret"]["mean"], dc["conservative_regret"]["mean"]]
    stds = [dc["speculative_regret"]["std"], dc["conservative_regret"]["std"]]
    colors = ["#2ca02c", "#d62728"]
    bars = axes[0, 0].bar(labels, means, color=colors, alpha=0.8, width=0.5)
    axes[0, 0].errorbar(labels, means, yerr=stds, fmt="none", color="black", capsize=5)
    for bar, m in zip(bars, means):
        y_pos = bar.get_height()
        va = "bottom" if m >= 0 else "top"
        offset = max(abs(m) * 0.05, 5) if m >= 0 else -max(abs(m) * 0.05, 5)
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            y_pos + offset,
            f"{m:.0f}",
            ha="center",
            va=va,
            fontsize=10,
        )
    axes[0, 0].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[0, 0].set_ylabel("Total Regret")
    alpha_spec = config.get("alpha_spec", "?")
    alpha_cons = config.get("alpha_cons", "?")
    axes[0, 0].set_title(
        f"(a) Error-Only Regret\n"
        f"($\\alpha_{{spec}}$={alpha_spec}, $\\alpha_{{cons}}$={alpha_cons})"
    )
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Panel (b): Latency comparison
    lat_labels = ["Speculative", "Conservative"]
    lat_means = [lt["speculative_mean_ms"]["mean"], lt["conservative_mean_ms"]["mean"]]
    lat_stds = [lt["speculative_mean_ms"]["std"], lt["conservative_mean_ms"]["std"]]
    bars = axes[0, 1].bar(lat_labels, lat_means, color=colors, alpha=0.8, width=0.5)
    axes[0, 1].errorbar(
        lat_labels, lat_means, yerr=lat_stds, fmt="none", color="black", capsize=5
    )
    for bar, m in zip(bars, lat_means):
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{m:.0f}ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0, 1].set_ylabel("Mean Latency (ms)")
    axes[0, 1].set_title(f"(b) Latency ({lt['speedup']['mean']:.1f}x speedup)")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Panel (c): Alpha-spec sweep — regret crossover
    if alpha_sweep and n_cols == 3:
        ax_sweep = axes[0, 2]
        sweep_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(alpha_sweep)))
        for i, entry in enumerate(alpha_sweep):
            a_spec = entry["alpha_spec"]
            rate_data = entry["rate_sweep"]
            rates = [pt["target_rate"] for pt in rate_data]
            regret_diffs = [
                pt["speculative_regret_mean"] - pt["conservative_regret_mean"]
                for pt in rate_data
            ]
            ax_sweep.plot(
                rates, regret_diffs, "o-",
                color=sweep_colors[i],
                label=f"$\\alpha_{{spec}}$={a_spec}",
                linewidth=1.5, markersize=4,
            )
        ax_sweep.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax_sweep.set_xlabel("Anomaly Rate")
        ax_sweep.set_ylabel("Regret Difference (Spec - Cons)")
        ax_sweep.set_title("(c) Crossover: Spec vs Cons by $\\alpha_{spec}$")
        ax_sweep.legend(fontsize=7, loc="upper left")
        ax_sweep.grid(True, alpha=0.3)
        ax_sweep.annotate(
            "Below 0 = speculative wins",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", fontsize=7, color="gray", style="italic",
        )

    # Panel (d): Rate sweep — error-only (solid) and economic (dashed)
    rate_sweep = agg.get("rate_sweep", [])
    rate_sweep_econ = agg.get("rate_sweep_economic", [])
    panel_d_col = 0
    if rate_sweep:
        rates = [pt["target_rate"] for pt in rate_sweep]
        spec_regs = [pt["speculative_regret_mean"] for pt in rate_sweep]
        cons_regs = [pt["conservative_regret_mean"] for pt in rate_sweep]

        axes[1, panel_d_col].plot(
            rates, spec_regs, "o-", color="#2ca02c",
            label="Spec (error-only)", linewidth=2,
        )
        axes[1, panel_d_col].plot(
            rates, cons_regs, "s-", color="#d62728",
            label="Cons (error-only)", linewidth=2,
        )

        if rate_sweep_econ:
            econ_rates = [pt["target_rate"] for pt in rate_sweep_econ]
            econ_spec = [pt["speculative_regret_mean"] for pt in rate_sweep_econ]
            axes[1, panel_d_col].plot(
                econ_rates, econ_spec, "o--", color="#2ca02c",
                alpha=0.5, label="Spec (w/ reward)", linewidth=1,
            )

        axes[1, panel_d_col].axvline(
            x=0.667, color="gray", linestyle="--", alpha=0.7,
            label=r"$\theta^*$=0.667",
        )
        axes[1, panel_d_col].axhline(y=0, color="gray", linestyle="-", alpha=0.2)
        axes[1, panel_d_col].set_xlabel("Anomaly Rate")
        axes[1, panel_d_col].set_ylabel("Total Regret")
        axes[1, panel_d_col].set_title(r"(d) Regret vs Anomaly Rate (dual threshold)")
        axes[1, panel_d_col].legend(fontsize=7, loc="upper left")
        axes[1, panel_d_col].grid(True, alpha=0.3)
    else:
        axes[1, panel_d_col].text(0.5, 0.5, "No rate sweep data", ha="center", va="center")
        axes[1, panel_d_col].set_title("(d) Rate Sweep")

    # Panel (e): Per-seed rollback rate + conservative blocks
    per_seed = data.get("per_seed", [])
    panel_e_col = 1
    if per_seed:
        seeds = [r["seed"] for r in per_seed]
        rollbacks = [
            r["comparisons"][0]["speculative"]["rollback_rate"] for r in per_seed
        ]
        cons_blocked = [
            r["comparisons"][0]["conservative"]["n_blocked"] for r in per_seed
        ]

        ax2 = axes[1, panel_e_col]
        x = np.arange(len(seeds))
        w = 0.35
        ax2.bar(
            x - w / 2, rollbacks, w,
            color="#ff7f0e", alpha=0.8, label="Spec Rollback Rate",
        )

        ax2_twin = ax2.twinx()
        ax2_twin.bar(
            x + w / 2, cons_blocked, w,
            color="#d62728", alpha=0.5, label="Cons Blocked",
        )
        ax2_twin.set_ylabel("Conservative Traces Blocked", color="#d62728")

        ax2.set_ylabel("Speculative Rollback Rate", color="#ff7f0e")
        ax2.set_xlabel("Seed")
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(s) for s in seeds], fontsize=8)
        ax2.set_title("(e) Per-Seed: Rollback vs Blocking")
        ax2.grid(True, alpha=0.3, axis="y")

    # Hide unused panel if 2x3 layout
    if n_cols == 3:
        axes[1, 2].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "speculative_regret.pdf")
    fig.savefig(FIG_DIR / "speculative_regret.png")
    plt.close()
    print("  speculative_regret.pdf")


def fig_three_way_decision():
    """Three-way decision error analysis from RAC sensitivity sweep."""
    path = TABLE_DIR / "rac_comparison.json"
    if not path.exists():
        print("  Skipping three_way_decision (no data)")
        return

    data = json.loads(path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Sensitivity sweep — VaR cost vs cost_ratio
    per_seed = data.get("per_seed", [])
    if per_seed and "rac_sweep" in per_seed[0]:
        sweep = per_seed[0]["rac_sweep"]
        cost_ratios = sorted(set(r.get("cost_ratio", 0) for r in sweep))

        # Aggregate across seeds
        for cr in cost_ratios:
            vars_at_cr = []
            for s in per_seed:
                for r in s.get("rac_sweep", []):
                    if r.get("cost_ratio") == cr:
                        vars_at_cr.append(r.get("var_cost", 0))
            if vars_at_cr:
                axes[0].plot(
                    [cr],
                    [np.mean(vars_at_cr)],
                    "o",
                    markersize=8,
                    color="#1f77b4",
                )

        axes[0].set_xlabel("Cost Ratio ($c_{FN} / c_{FP}$)")
        axes[0].set_ylabel("VaR Cost")
        axes[0].set_title("VaR Cost vs Cost Ratio")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No sweep data",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    # Panel 2: Decision zone proportions (approve/defer/reject)
    agg = data["aggregate"]
    rac = agg["rac_default"]
    defer = rac["defer_rate"]["mean"]
    coverage = rac["coverage"]["mean"]
    reject_rate = 1.0 - coverage - defer

    labels = ["Approve", "Defer", "Reject"]
    sizes = [coverage, defer, max(reject_rate, 0)]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    explode = (0, 0.05, 0)

    wedges, texts, autotexts = axes[1].pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[1].set_title("Three-Way Decision Distribution")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "three_way_decision.pdf")
    fig.savefig(FIG_DIR / "three_way_decision.png")
    plt.close()
    print("  three_way_decision.pdf")


def fig_leakage_decomposition():
    """v13 E17: 2-panel leakage decomposition figure.

    Panel A: Bar chart — Seq-Only, CQ, CQ-Shuffled, Outcome-Only (5-seed, CIs)
    Panel B: Line plot — CQ advantage vs noise_p [0.0–1.0] (3-seed, shaded CI)
    """
    leakage_path = TABLE_DIR / "leakage_analysis.json"
    noisy_path = TABLE_DIR / "noisy_outcome_sweep.json"

    has_leakage = leakage_path.exists()
    has_noisy = noisy_path.exists()

    if not has_leakage and not has_noisy:
        print("  Skipping leakage decomposition (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel A: Leakage bar chart ---
    if has_leakage:
        data = json.loads(leakage_path.read_text())
        agg = data["aggregate"]
        mi = data["mutual_information"]

        scorers = [
            ("Seq-Only", "seq_only_auroc", "#2196F3"),
            ("CQ", "cq_auroc", "#9C27B0"),
            ("CQ (shuffled)", "shuffled_cq_auroc", "#FF9800"),
            ("Outcome-Only", "outcome_only_auroc", "#F44336"),
        ]

        x = np.arange(len(scorers))
        means = [agg[key]["mean"] for _, key, _ in scorers]
        stds = [agg[key]["std"] for _, key, _ in scorers]
        colors = [c for _, _, c in scorers]
        labels = [n for n, _, _ in scorers]

        axes[0].bar(
            x, means, 0.55, yerr=stds, capsize=4,
            color=colors, alpha=0.85, edgecolor="white",
        )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=9)
        axes[0].set_ylabel("AUROC")
        axes[0].set_ylim(0.4, 1.05)
        axes[0].set_title(
            f"(a) Leakage Decomposition "
            f"(MI={mi['mi_bits']:.2f} bits)"
        )
        axes[0].grid(True, alpha=0.3, axis="y")

        if agg.get("leakage_fraction"):
            lf = agg["leakage_fraction"]
            axes[0].annotate(
                f"Leakage: {lf['mean']:.0%} "
                f"[{lf['ci_lower']:.0%}, {lf['ci_upper']:.0%}]",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=8, color="gray", style="italic",
            )
    else:
        axes[0].text(
            0.5, 0.5, "No leakage data", ha="center", va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("(a) Leakage Decomposition")

    # --- Panel B: Noise degradation curve ---
    if has_noisy:
        ndata = json.loads(noisy_path.read_text())
        results = ndata["results"]

        noise_ps = [r["noise_p"] for r in results]
        cq_means = [r["cq_advantage"]["mean"] for r in results]
        cq_lo = [r["cq_advantage"]["ci_lower"] for r in results]
        cq_hi = [r["cq_advantage"]["ci_upper"] for r in results]

        axes[1].plot(
            noise_ps, cq_means, "o-", color="#9C27B0",
            linewidth=2, markersize=6, label="CQ advantage",
        )
        axes[1].fill_between(
            noise_ps, cq_lo, cq_hi,
            alpha=0.2, color="#9C27B0",
        )
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Outcome noise probability")
        axes[1].set_ylabel("CQ AUROC advantage over Seq-Only")
        axes[1].set_title("(b) CQ Advantage vs Outcome Noise")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        if len(noise_ps) > 1:
            axes[1].annotate(
                "noise_p=0: deterministic (max leakage)\n"
                "noise_p=1: no outcome signal",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=7,
                color="gray", style="italic",
            )
    else:
        axes[1].text(
            0.5, 0.5, "No noisy sweep data", ha="center", va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_title("(b) CQ Advantage vs Outcome Noise")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "leakage_decomposition.pdf")
    fig.savefig(FIG_DIR / "leakage_decomposition.png")
    plt.close()
    print("  leakage_decomposition.pdf")


def fig_lstm_outcome_comparison():
    """v13 E19: Bar chart comparing Seq-Only, LSTM, LSTM+Outcome, CQ.

    Reads from cp_comparison.json (which now includes LSTM+Outcome
    as a scorer).
    """
    path = TABLE_DIR / "cp_comparison.json"
    if not path.exists():
        print("  Skipping LSTM outcome comparison (no data)")
        return

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "aggregate" not in raw:
        print("  Skipping LSTM outcome comparison (old format)")
        return

    agg = raw["aggregate"]

    target_scorers = [
        ("Transformer Encoder", "#2196F3"),
        ("LSTM Next-Token", "#4CAF50"),
        ("LSTM+Outcome", "#FF9800"),
        ("Consequence-Aware Encoder", "#9C27B0"),
    ]

    present = [(s, c) for s, c in target_scorers if s in agg]
    if len(present) < 3:
        print(
            f"  Skipping LSTM outcome comparison "
            f"(need >=3 scorers, have {len(present)})"
        )
        return

    names = [s for s, _ in present]
    colors = [c for _, c in present]
    aurocs = [agg[s]["auroc_mean"] for s in names]
    auroc_errs = [agg[s]["auroc_std"] for s in names]
    f1s = [agg[s]["f1_mean"] for s in names]
    f1_errs = [agg[s]["f1_std"] for s in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    x = np.arange(len(names))
    w = 0.5

    short_names = [
        n.replace("Transformer Encoder", "Seq-Only\nTransformer")
        .replace("LSTM Next-Token", "LSTM")
        .replace("LSTM+Outcome", "LSTM\n+Outcome")
        .replace("Consequence-Aware Encoder", "CQ\nTransformer")
        for n in names
    ]

    axes[0].bar(
        x, aurocs, w, yerr=auroc_errs, capsize=4,
        color=colors, alpha=0.85, edgecolor="white",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=9)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_title("AUROC: Architecture vs Outcome Ablation")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(
        x, f1s, w, yerr=f1_errs, capsize=4,
        color=colors, alpha=0.85, edgecolor="white",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=9)
    axes[1].set_ylabel("F1 (under Split CP)")
    axes[1].set_ylim(0, 0.5)
    axes[1].set_title("F1: Architecture vs Outcome Ablation")
    axes[1].grid(True, alpha=0.3, axis="y")

    n_seeds = raw.get("config", {}).get("n_seeds", "?")
    plt.suptitle(
        f"Architecture vs Outcome Information ({n_seeds} seeds)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "lstm_outcome_comparison.pdf")
    fig.savefig(FIG_DIR / "lstm_outcome_comparison.png")
    plt.close()
    print("  lstm_outcome_comparison.pdf")


def fig_coverage_vs_auroc_scatter():
    """v13-reframe: Coverage slack vs AUROC scatter — scorer-agnostic CP narrative.

    Key reframing figure: demonstrates that CP coverage is scorer-agnostic
    (guarantee holds for all) but practical tightness depends on scorer quality.
    """
    path = TABLE_DIR / "cp_comparison.json"
    if not path.exists():
        print("  Skipping coverage-vs-AUROC scatter (no data)")
        return

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "aggregate" not in raw:
        print("  Skipping coverage-vs-AUROC scatter (old format)")
        return

    agg = raw["aggregate"]
    alpha = raw.get("config", {}).get("alpha", 0.05)
    target_coverage = 1.0 - alpha

    fig, ax = plt.subplots(figsize=(8, 6))

    short_labels = {
        "Consequence-Aware Encoder": "CQ",
        "LSTM+Outcome": "LSTM+Out",
        "Transformer Encoder": "Transformer",
        "LSTM Next-Token": "LSTM",
        "Isolation Forest": "IF",
        "Large Seq-Only (deep)": "Large-deep",
        "Large Seq-Only (wide)": "Large-wide",
        "N-gram LM": "N-gram",
    }

    for name, stats in agg.items():
        auroc = stats["auroc_mean"]
        cov_slack = stats["coverage_mean"] - target_coverage
        bf = stats["bound_holds_fraction"]
        n_eff = stats.get("n_effective", stats.get("n_seeds", 1))

        if bf >= 0.8:
            color, marker = "#4CAF50", "o"
        elif bf >= 0.4:
            color, marker = "#FF9800", "s"
        else:
            color, marker = "#F44336", "X"

        size = 120 if n_eff > 1 else 60
        ax.scatter(
            auroc, cov_slack * 100, c=color, marker=marker,
            s=size, edgecolors="white", linewidth=0.5, zorder=3,
        )
        label = short_labels.get(name, name)
        offset_x = 0.01
        offset_y = 0.15 if cov_slack >= 0 else -0.25
        ax.annotate(
            label, (auroc, cov_slack * 100),
            textcoords="offset points", xytext=(5, 5),
            fontsize=8, ha="left",
        )

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
               label=f"Target coverage (1-α={target_coverage:.2f})")
    ax.axhspan(-1, 0, alpha=0.05, color="red")
    ax.axhspan(0, 3.5, alpha=0.05, color="green")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=10, label="Bound holds ≥80%"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF9800",
               markersize=10, label="Bound holds 40-80%"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#F44336",
               markersize=10, label="Bound holds <40%"),
        Line2D([0], [0], color="red", linestyle="--", alpha=0.6,
               label=f"Target 1-α={target_coverage:.2f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_xlabel("Scorer AUROC", fontsize=11)
    ax.set_ylabel("Coverage Slack (pp above target)", fontsize=11)
    ax.set_title(
        "Scorer Quality vs CP Coverage Tightness\n"
        "(CP guarantee is scorer-agnostic; tightness is not)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-1.5, 3.5)

    n_seeds = raw.get("config", {}).get("n_seeds", "?")
    ax.annotate(
        f"{n_seeds} seeds, mixed-cal calibration",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7, color="gray", style="italic",
    )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "coverage_vs_auroc_scatter.pdf")
    fig.savefig(FIG_DIR / "coverage_vs_auroc_scatter.png")
    plt.close()
    print("  coverage_vs_auroc_scatter.pdf")


def fig_atbench_eval():
    """ATBench CP evaluation: AUROC + coverage across scorers and risk categories."""
    path = TABLE_DIR / "atbench_eval.json"
    if not path.exists():
        print("  Skipping ATBench eval (no data)")
        return

    raw = json.loads(path.read_text())
    agg = raw.get("aggregate", {})
    baselines = raw.get("baselines", {})
    risk_cats = raw.get("per_risk_category", {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: AUROC comparison (LSTM vs baselines)
    ax = axes[0]
    names = ["LSTM"]
    aurocs = [agg.get("auroc", {}).get("mean", 0)]
    stds = [agg.get("auroc", {}).get("std", 0)]
    for bn in sorted(baselines.keys()):
        names.append(bn.replace("_", "\n"))
        aurocs.append(baselines[bn].get("auroc_mean", 0))
        aurocs_std = baselines[bn].get("auroc_std", 0)
        stds.append(aurocs_std)

    x = np.arange(len(names))
    colors = ["#2196F3"] + ["#FF9800"] * (len(names) - 1)
    ax.bar(x, aurocs, 0.6, yerr=stds, color=colors, alpha=0.8,
           edgecolor="white", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Scorer Comparison")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    # Panel 2: Coverage + miscoverage ratio
    ax = axes[1]
    names2 = ["LSTM"]
    covs = [agg.get("coverage", {}).get("mean", 0)]
    mrs = [agg.get("miscoverage_ratio", {}).get("mean", 0)]
    for bn in sorted(baselines.keys()):
        names2.append(bn.replace("_", "\n"))
        covs.append(baselines[bn].get("coverage_mean", 0))
        mrs.append(baselines[bn].get("miscoverage_ratio_mean", 0))

    x2 = np.arange(len(names2))
    ax.bar(x2 - 0.15, covs, 0.3, color="#4CAF50", alpha=0.8, label="Coverage")
    ax.bar(x2 + 0.15, mrs, 0.3, color="#F44336", alpha=0.8, label="Miscov Ratio")
    ax.set_xticks(x2)
    ax.set_xticklabels(names2, fontsize=9)
    ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Coverage & Miscoverage Ratio")
    ax.legend(fontsize=8, loc="lower right")

    # Panel 3: Per-risk-category detection (LSTM)
    ax = axes[2]
    if risk_cats:
        rnames = sorted(risk_cats.keys())
        r_covs = [risk_cats[r].get("coverage_mean", 0) for r in rnames]
        r_dets = [risk_cats[r].get("detection_mean", 0) or 0 for r in rnames]
        short_rnames = [r.replace("_", "\n")[:20] for r in rnames]

        x3 = np.arange(len(rnames))
        ax.barh(x3, r_dets, 0.4, color="#E91E63", alpha=0.8)
        ax.set_yticks(x3)
        ax.set_yticklabels(short_rnames, fontsize=7)
        ax.set_xlabel("Detection Rate")
        ax.set_title("Per-Risk Detection (LSTM)")
        ax.set_xlim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No per-risk data", ha="center", va="center",
                transform=ax.transAxes)

    plt.suptitle("ATBench Evaluation (Real Safety Labels)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "atbench_eval.pdf")
    fig.savefig(FIG_DIR / "atbench_eval.png")
    plt.close()
    print("  atbench_eval.pdf")


def fig_rac_swe_agent():
    """RAC evaluation on SWE-agent: VaR improvement across scorers."""
    path = TABLE_DIR / "rac_swe_eval.json"
    if not path.exists():
        print("  Skipping RAC SWE-agent (no data)")
        return

    raw = json.loads(path.read_text())
    lstm_agg = raw.get("aggregate", {}).get("lstm", {})
    baselines = raw.get("baselines", {})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel 1: VaR improvement comparison
    ax = axes[0]
    names = ["LSTM"]
    improvements = [lstm_agg.get("var_improvement", {}).get("mean", 0)]
    imp_stds = [lstm_agg.get("var_improvement", {}).get("std", 0)]

    for bn in sorted(baselines.keys()):
        names.append(bn.replace("_", "\n"))
        improvements.append(baselines[bn].get("var_improvement_mean", 0))
        imp_ci = baselines[bn].get("var_improvement_ci", {})
        imp_stds.append(imp_ci.get("std", 0))

    x = np.arange(len(names))
    colors = ["#2196F3"] + ["#FF9800"] * (len(names) - 1)
    ax.bar(x, improvements, 0.6, yerr=imp_stds, color=colors, alpha=0.8,
           edgecolor="white", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("VaR Improvement (binary - RAC)")
    ax.set_title("RAC VaR Improvement")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel 2: Defer rates
    ax = axes[1]
    defer_names = ["LSTM"]
    defer_rates = [lstm_agg.get("defer_rate_mean", 0)]

    for bn in sorted(baselines.keys()):
        defer_names.append(bn.replace("_", "\n"))
        defer_rates.append(baselines[bn].get("defer_rate_mean", 0))

    x2 = np.arange(len(defer_names))
    ax.bar(x2, defer_rates, 0.6, color="#9C27B0", alpha=0.8, edgecolor="white")
    ax.set_xticks(x2)
    ax.set_xticklabels(defer_names, fontsize=9)
    ax.set_ylabel("Defer Rate")
    ax.set_title("Human Referral Rate")
    ax.set_ylim(0, 0.35)

    plt.suptitle("RAC on SWE-agent (Real-World)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "rac_swe_agent.pdf")
    fig.savefig(FIG_DIR / "rac_swe_agent.png")
    plt.close()
    print("  rac_swe_agent.pdf")


def fig_hybrid_comparison():
    """v22: Bar chart comparing scorers across datasets, with ablation inset."""
    path = TABLE_DIR / "hybrid_eval_v22.json"
    if not path.exists():
        print("  Skipping hybrid comparison (no data)")
        return

    raw = json.loads(path.read_text())

    ds_keys = [("swe_agent", "SWE-agent"), ("atbench", "ATBench"),
                ("synthetic", "Synthetic")]
    datasets = []
    ablation_data = []
    labels = []
    for key, label in ds_keys:
        if key in raw and "aggregate" in raw[key]:
            datasets.append(raw[key]["aggregate"])
            ablation_data.append(raw[key].get("ablation_aggregate", {}))
            labels.append(label)

    if not datasets:
        print("  Skipping hybrid comparison (no aggregate data)")
        return

    scorer_display = [
        ("lstm", "LSTM", "#2196F3"),
        ("trace_length", "Trace Length", "#FF9800"),
        ("low_diversity", "Low Diversity", "#4CAF50"),
        ("hybrid_equal", "Hybrid (equal)", "#9C27B0"),
        ("hybrid_fitted", "Hybrid (LR)", "#E91E63"),
        ("hybrid_rf", "Hybrid (RF)", "#795548"),
        ("hybrid_gbt", "Hybrid (GBT)", "#607D8B"),
    ]

    has_ablation = any(ab for ab in ablation_data)
    n_rows = 2 if has_ablation else 1
    fig, all_axes = plt.subplots(
        n_rows, len(datasets),
        figsize=(6 * len(datasets), 5 * n_rows),
    )
    if len(datasets) == 1 and n_rows == 1:
        all_axes = np.array([[all_axes]])
    elif len(datasets) == 1:
        all_axes = all_axes.reshape(n_rows, 1)
    elif n_rows == 1:
        all_axes = all_axes.reshape(1, -1)

    # Row 0: Main scorer comparison
    for col, (agg, ds_label) in enumerate(zip(datasets, labels)):
        ax = all_axes[0, col]
        present = [(k, lbl, c) for k, lbl, c in scorer_display if k in agg]
        names = [lbl for _, lbl, _ in present]
        aurocs = [agg[k]["auroc_mean"] for k, _, _ in present]
        stds = [agg[k]["auroc_std"] for k, _, _ in present]
        colors = [c for _, _, c in present]

        x = np.arange(len(names))
        ax.bar(x, aurocs, 0.6, yerr=stds, color=colors, alpha=0.85,
               edgecolor="white", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("AUROC")
        ax.set_title(ds_label)
        ax.set_ylim(0.3, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
        ax.grid(True, alpha=0.3, axis="y")
        for i, (a, s) in enumerate(zip(aurocs, stds)):
            ax.text(x[i], a + s + 0.01, f"{a:.3f}", ha="center", fontsize=8)

    # Row 1: Ablation (leave-one-feature-out) for hybrid_equal and hybrid_rf
    if has_ablation:
        features = ["lstm_nll", "trace_length", "low_diversity", "repeat_rate"]
        feat_labels = ["LSTM", "Length", "Diversity", "Repeat"]
        for col, (ab_agg, ds_label) in enumerate(zip(ablation_data, labels)):
            ax = all_axes[1, col]
            if not ab_agg:
                ax.set_visible(False)
                continue
            x = np.arange(len(features))
            w = 0.35
            eq_aurocs = [
                ab_agg.get(f"minus_{f}_equal", {}).get("auroc_mean", 0)
                for f in features
            ]
            rf_aurocs = [
                ab_agg.get(f"minus_{f}_rf", {}).get("auroc_mean", 0)
                for f in features
            ]
            ax.bar(x - w / 2, eq_aurocs, w, label="Hybrid-Eq (minus)",
                   color="#9C27B0", alpha=0.7)
            ax.bar(x + w / 2, rf_aurocs, w, label="Hybrid-RF (minus)",
                   color="#795548", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"-{fl}" for fl in feat_labels], fontsize=9)
            ax.set_ylabel("AUROC")
            ax.set_title(f"{ds_label} — Leave-One-Out Ablation")
            ax.set_ylim(0.3, 1.0)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

    n_seeds = raw.get("config", {}).get("n_seeds", "?")
    plt.suptitle(f"Hybrid Scorer Comparison ({n_seeds} seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hybrid_comparison.pdf")
    fig.savefig(FIG_DIR / "hybrid_comparison.png")
    plt.close()
    print("  hybrid_comparison.pdf")


def fig_hybrid_cp():
    """v22: Coverage and miscoverage ratio for hybrid vs individual scorers."""
    path = TABLE_DIR / "hybrid_eval_v22.json"
    if not path.exists():
        print("  Skipping hybrid CP (no data)")
        return

    raw = json.loads(path.read_text())
    alpha = raw.get("config", {}).get("alpha", 0.05)

    datasets = []
    labels = []
    for key, label in [("swe_agent", "SWE-agent"), ("atbench", "ATBench"),
                        ("synthetic", "Synthetic")]:
        if key in raw and "aggregate" in raw[key]:
            datasets.append(raw[key]["aggregate"])
            labels.append(label)

    if not datasets:
        print("  Skipping hybrid CP (no aggregate data)")
        return

    scorers = [
        "lstm", "trace_length", "low_diversity",
        "hybrid_equal", "hybrid_fitted", "hybrid_rf", "hybrid_gbt",
    ]
    scorer_labels = [
        "LSTM", "Length", "Diversity",
        "Hybrid-Eq", "Hybrid-LR", "Hybrid-RF", "Hybrid-GBT",
    ]

    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 8))
    if len(datasets) == 1:
        axes = axes.reshape(2, 1)

    for col, (agg, ds_label) in enumerate(zip(datasets, labels)):
        present_idx = [i for i, s in enumerate(scorers) if s in agg]
        names = [scorer_labels[i] for i in present_idx]
        keys = [scorers[i] for i in present_idx]

        covs = [agg[k]["coverage_mean"] for k in keys]
        cov_stds = [agg[k]["coverage_std"] for k in keys]
        misc_ratios = [agg[k].get("mean_miscoverage_ratio", 1.0) for k in keys]

        x = np.arange(len(names))
        colors_cov = [
            "#4CAF50" if mr <= 1.0 else "#FF9800" if mr <= 1.5 else "#F44336"
            for mr in misc_ratios
        ]

        # Top: Coverage
        axes[0, col].bar(x, covs, 0.6, yerr=cov_stds, color=colors_cov,
                         alpha=0.8, edgecolor="white", capsize=3)
        axes[0, col].axhline(y=1 - alpha, color="red", linestyle="--",
                             alpha=0.6, label=f"1-α={1 - alpha}")
        axes[0, col].set_xticks(x)
        axes[0, col].set_xticklabels(names, fontsize=9)
        axes[0, col].set_ylabel("Coverage")
        axes[0, col].set_title(f"{ds_label} — Coverage")
        axes[0, col].set_ylim(0.85, 1.02)
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3, axis="y")

        # Bottom: Miscoverage ratio (< 1.0 means target exceeded)
        colors_mr = [
            "#4CAF50" if mr <= 1.0 else "#FF9800" if mr <= 1.5 else "#F44336"
            for mr in misc_ratios
        ]
        axes[1, col].bar(x, misc_ratios, 0.6, color=colors_mr,
                         alpha=0.8, edgecolor="white")
        axes[1, col].axhline(y=1.0, color="red", linestyle="--", alpha=0.6,
                             label="Ratio = 1.0 (target)")
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(names, fontsize=9)
        axes[1, col].set_ylabel("Mean Miscoverage Ratio")
        axes[1, col].set_title(f"{ds_label} — Miscoverage Ratio")
        axes[1, col].set_ylim(0, max(max(misc_ratios) * 1.2, 1.5))
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3, axis="y")
        for i, mr in enumerate(misc_ratios):
            axes[1, col].text(
                x[i], mr + 0.03, f"{mr:.2f}", ha="center", fontsize=8,
                color="#2E7D32" if mr <= 1.0 else "#C62828",
            )

    plt.suptitle("CP Coverage: Hybrid vs Individual Scorers",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "hybrid_cp.pdf")
    fig.savefig(FIG_DIR / "hybrid_cp.png")
    plt.close()
    print("  hybrid_cp.pdf")


def fig_transfer_learning():
    """v20: Transfer learning AUROC comparison (from-scratch vs zero-shot)."""
    path = TABLE_DIR / "hybrid_eval_v22.json"
    if not path.exists():
        print("  Skipping transfer learning (no data)")
        return

    raw = json.loads(path.read_text())
    swe = raw.get("swe_agent", {})
    per_seed = swe.get("per_seed", [])

    has_transfer = any("transfer" in r for r in per_seed)
    if not has_transfer:
        print("  Skipping transfer learning (no transfer data)")
        return

    scratch_aurocs = [r["lstm"]["auroc"] for r in per_seed]
    transfer_aurocs = [
        r["transfer"]["zero_shot"]["auroc"]
        for r in per_seed if "transfer" in r
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(scratch_aurocs))
    w = 0.35

    ax.bar(x - w / 2, scratch_aurocs, w, label="From-Scratch (SWE-agent)",
           color="#2196F3", alpha=0.85)
    if len(transfer_aurocs) == len(scratch_aurocs):
        ax.bar(x + w / 2, transfer_aurocs, w, label="Transfer (Synthetic→SWE)",
               color="#FF9800", alpha=0.85)

    seeds = [r["seed"] for r in per_seed]
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=9)
    ax.set_xlabel("Seed")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("LSTM Scoring: From-Scratch vs Transfer Learning")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    scratch_arr = np.array(scratch_aurocs)
    scratch_mean = float(scratch_arr.mean())
    scratch_std = float(scratch_arr.std())
    ax.annotate(
        f"From-scratch: {scratch_mean:.3f} +/- {scratch_std:.3f}",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, color="gray",
    )
    if transfer_aurocs:
        transfer_arr = np.array(transfer_aurocs)
        transfer_mean = float(transfer_arr.mean())
        transfer_std = float(transfer_arr.std())
        ax.annotate(
            f"Transfer: {transfer_mean:.3f} +/- {transfer_std:.3f}",
            xy=(0.02, 0.08), xycoords="axes fraction",
            fontsize=8, color="gray",
        )

    plt.tight_layout()
    fig.savefig(FIG_DIR / "transfer_learning.pdf")
    fig.savefig(FIG_DIR / "transfer_learning.png")
    plt.close()
    print("  transfer_learning.pdf")


def fig_domain_inversion():
    """v22: Cohen's d heatmap showing feature sign-inversion across datasets."""
    path = TABLE_DIR / "hybrid_eval_v22.json"
    if not path.exists():
        print("  Skipping domain inversion (no data)")
        return

    raw = json.loads(path.read_text())

    ds_keys = [("swe_agent", "SWE-agent"), ("atbench", "ATBench"),
                ("synthetic", "Synthetic")]
    features = ["trace_length", "low_diversity", "repeat_rate", "lstm"]
    feat_labels = ["Trace Length", "Low Diversity", "Repeat Rate", "LSTM NLL"]

    # Compute Cohen's d from per-seed AUROC: d = (AUROC - 0.5) / pooled_std_approx
    # For anomaly scorers, AUROC > 0.5 means the feature separates normal/anomalous
    # in the expected direction; AUROC < 0.5 means inversion.
    # We use a signed effect: d > 0 = feature works, d < 0 = inverted.
    d_matrix = np.zeros((len(features), len(ds_keys)))
    ds_present = []

    for col, (key, label) in enumerate(ds_keys):
        if key not in raw or "per_seed" not in raw[key]:
            continue
        ds_present.append(label)
        per_seed = raw[key]["per_seed"]
        for row, feat in enumerate(features):
            aurocs = [r[feat]["auroc"] for r in per_seed if feat in r]
            if not aurocs:
                continue
            mean_auroc = np.mean(aurocs)
            std_auroc = max(np.std(aurocs), 0.01)
            # Signed Cohen's d: positive = discriminative, negative = anti-predictive
            d_matrix[row, col] = (mean_auroc - 0.5) / std_auroc

    if not ds_present:
        print("  Skipping domain inversion (no per-seed data)")
        return

    d_matrix = d_matrix[:, :len(ds_present)]

    fig, ax = plt.subplots(figsize=(max(5, 2 * len(ds_present)), 4))
    im = ax.imshow(d_matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-max(abs(d_matrix.min()), abs(d_matrix.max())),
                   vmax=max(abs(d_matrix.min()), abs(d_matrix.max())))

    ax.set_xticks(range(len(ds_present)))
    ax.set_xticklabels(ds_present, fontsize=10)
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=10)

    # Annotate cells
    for i in range(len(feat_labels)):
        for j in range(len(ds_present)):
            val = d_matrix[i, j]
            color = "white" if abs(val) > 2 else "black"
            direction = "+" if val > 0 else "-" if val < 0 else "0"
            ax.text(j, i, f"{val:.1f}\n({direction})", ha="center",
                    va="center", fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Signed Cohen's d (+ = discriminative, - = inverted)")
    ax.set_title("Domain Inversion: Feature Discriminability Across Datasets",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "domain_inversion.pdf")
    fig.savefig(FIG_DIR / "domain_inversion.png")
    plt.close()
    print("  domain_inversion.pdf")


# ---------------------------------------------------------------------------
# v26: FSI + PA-CP + Adversarial figures
# ---------------------------------------------------------------------------


def fig_fsi_heatmap():
    """NxN domain pair x feature FSI heatmap. Red=reversal, blue=stable."""
    path = TABLE_DIR / "fsi_analysis.json"
    if not path.exists():
        print("  Skipping FSI heatmap (no data)")
        return

    data = json.loads(path.read_text())
    fsi_summary = data.get("fsi_summary", {})
    features = data.get("features", [])
    if not fsi_summary or not features:
        print("  Skipping FSI heatmap (empty data)")
        return

    pairs = sorted(fsi_summary.keys())
    n_pairs = len(pairs)
    n_features = len(features)

    matrix = np.zeros((n_pairs, n_features))
    for i, pair in enumerate(pairs):
        for j, feat in enumerate(features):
            entry = fsi_summary[pair].get(feat, {})
            matrix[i, j] = entry.get("mean", 0.0)

    fig, ax = plt.subplots(figsize=(max(8, n_features * 1.5), max(6, n_pairs * 0.4)))
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.1)
    im = ax.imshow(matrix, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_pairs))
    pair_labels = [p.replace("_vs_", " vs ") for p in pairs]
    ax.set_yticklabels(pair_labels, fontsize=7)

    for i in range(n_pairs):
        for j in range(n_features):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="FSI (negative = polarity reversal)")
    ax.set_title("Feature Stability Index Across Domain Pairs", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fsi_heatmap.pdf")
    fig.savefig(FIG_DIR / "fsi_heatmap.png")
    plt.close()
    print("  fsi_heatmap.pdf")


def fig_pacp_coverage():
    """Bar chart: standard CP vs PA-CP coverage + AUROC across domain pairs."""
    path = TABLE_DIR / "pacp_eval.json"
    if not path.exists():
        print("  Skipping PA-CP coverage (no data)")
        return

    data = json.loads(path.read_text())
    aggregate = data.get("aggregate", {})
    if not aggregate:
        print("  Skipping PA-CP coverage (empty data)")
        return

    pairs = sorted(aggregate.keys())
    std_means = [aggregate[p]["standard_cp_coverage"]["mean"] for p in pairs]
    pacp_means = [aggregate[p]["pacp_coverage"]["mean"] for p in pairs]

    has_auroc = all("standard_cp_auroc" in aggregate[p] for p in pairs)
    n_panels = 2 if has_auroc else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(max(10, len(pairs) * 0.8) * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(pairs))
    w = 0.35
    pair_labels = [p.replace("_to_", "\nto\n") for p in pairs]

    ax_cov = axes[0]
    ax_cov.bar(x - w / 2, std_means, w, label="Standard CP", color="#F44336", alpha=0.8)
    ax_cov.bar(x + w / 2, pacp_means, w, label="PA-CP", color="#2196F3", alpha=0.8)
    ax_cov.axhline(y=0.95, color="green", linestyle="--", alpha=0.5, label="Target (1-alpha)")
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(pair_labels, fontsize=7, ha="center")
    ax_cov.set_ylabel("Coverage on normal samples")
    ax_cov.set_title("Coverage: Standard CP vs PA-CP", fontweight="bold")
    ax_cov.legend(loc="lower left")
    ax_cov.set_ylim(0, 1.1)
    ax_cov.grid(True, alpha=0.3, axis="y")

    if has_auroc:
        std_aurocs = [aggregate[p]["standard_cp_auroc"]["mean"] for p in pairs]
        pacp_aurocs = [aggregate[p]["pacp_auroc"]["mean"] for p in pairs]

        ax_aur = axes[1]
        ax_aur.bar(x - w / 2, std_aurocs, w, label="All features", color="#F44336", alpha=0.8)
        ax_aur.bar(x + w / 2, pacp_aurocs, w, label="Stable features", color="#2196F3", alpha=0.8)
        ax_aur.set_xticks(x)
        ax_aur.set_xticklabels(pair_labels, fontsize=7, ha="center")
        ax_aur.set_ylabel("AUROC")
        ax_aur.set_title("AUROC: All Features vs Stable Features", fontweight="bold")
        ax_aur.legend(loc="lower left")
        ax_aur.set_ylim(0, 1.1)
        ax_aur.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "pacp_coverage.pdf")
    fig.savefig(FIG_DIR / "pacp_coverage.png")
    plt.close()
    print("  pacp_coverage.pdf")


def fig_adversarial_scatter():
    """Dual-panel scatter: |FSI| vs AUROC (left) and |FSI| vs evasion (right)."""
    path = TABLE_DIR / "adversarial_domain.json"
    if not path.exists():
        print("  Skipping adversarial scatter (no data)")
        return

    data = json.loads(path.read_text())
    pair_aggregate = data.get("pair_aggregate", {})
    if not pair_aggregate:
        print("  Skipping adversarial scatter (empty data)")
        return

    # Detect v26.1 (dual rho) vs v26 (single rho) format
    rho_auroc_summary = data.get("spearman_rho_auroc_summary", {})
    rho_evasion_summary = data.get(
        "spearman_rho_evasion_summary",
        data.get("spearman_rho_summary", {}),
    )
    has_dual = bool(rho_auroc_summary)

    pairs = sorted(pair_aggregate.keys())
    abs_fsi = [pair_aggregate[p]["abs_fsi"]["mean"] for p in pairs]
    evasion = [pair_aggregate[p]["evasion_rate"]["mean"] for p in pairs]
    auroc = [pair_aggregate[p]["auroc"]["mean"] for p in pairs]

    def _scatter_panel(ax, x_vals, y_vals, ylabel, rho_summary, title_prefix):
        ax.scatter(
            x_vals, y_vals, s=60, alpha=0.7, c="#2196F3",
            edgecolors="white", linewidth=0.5,
        )
        for i, pair in enumerate(pairs):
            label = pair.replace("_to_", "->").replace("_", "")
            ax.annotate(
                label, (x_vals[i], y_vals[i]), fontsize=5,
                xytext=(4, 4), textcoords="offset points", alpha=0.7,
            )
        if len(x_vals) > 2:
            z = np.polyfit(x_vals, y_vals, 1)
            poly = np.poly1d(z)
            x_fit = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(x_fit, poly(x_fit), "--", color="red", alpha=0.6, linewidth=1.5)
        rho_mean = rho_summary.get("mean", 0.0)
        rho_lo = rho_summary.get("ci_lower", 0.0)
        rho_hi = rho_summary.get("ci_upper", 0.0)
        ax.set_xlabel("|Aggregate FSI|")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{title_prefix}\nrho={rho_mean:.3f} [{rho_lo:.3f}, {rho_hi:.3f}]",
            fontweight="bold", fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    if has_dual:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        _scatter_panel(ax1, abs_fsi, auroc, "AUROC", rho_auroc_summary, "|FSI| vs AUROC")
        _scatter_panel(ax2, abs_fsi, evasion, "Evasion Rate", rho_evasion_summary,
                       "|FSI| vs Evasion")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        _scatter_panel(ax, abs_fsi, evasion, "Evasion Rate (FNR on anomalies)",
                       rho_evasion_summary, "Adversarial Domain Selection")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "adversarial_scatter.pdf")
    fig.savefig(FIG_DIR / "adversarial_scatter.png")
    plt.close()
    print("  adversarial_scatter.pdf")


def fig_fsi_threshold_sweep():
    """Line: PUF threshold vs (coverage, AUROC) tradeoff."""
    path = TABLE_DIR / "pacp_eval.json"
    if not path.exists():
        print("  Skipping FSI threshold sweep (no data)")
        return

    data = json.loads(path.read_text())
    thresholds = data.get("puf_thresholds", [])
    per_seed = data.get("per_seed_results", [])
    if not per_seed or not thresholds:
        print("  Skipping FSI threshold sweep (empty data)")
        return

    coverage_by_thresh: dict[float, list[float]] = {t: [] for t in thresholds}
    auroc_by_thresh: dict[float, list[float]] = {t: [] for t in thresholds}

    for result in per_seed:
        for tr in result.get("pacp_by_threshold", []):
            t = tr["threshold"]
            if t in coverage_by_thresh:
                coverage_by_thresh[t].append(tr["coverage"])
                auroc_by_thresh[t].append(tr["auroc"])

    thresh_vals = sorted(thresholds)
    cov_means = [np.mean(coverage_by_thresh[t]) for t in thresh_vals]
    cov_stds = [np.std(coverage_by_thresh[t]) for t in thresh_vals]
    auroc_means = [np.mean(auroc_by_thresh[t]) for t in thresh_vals]
    auroc_stds = [np.std(auroc_by_thresh[t]) for t in thresh_vals]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color1 = "#2196F3"
    color2 = "#F44336"

    ax1.errorbar(thresh_vals, cov_means, yerr=cov_stds, fmt="o-",
                 color=color1, capsize=4, label="Coverage")
    ax1.set_xlabel("PUF Threshold (FSI < threshold = unstable)")
    ax1.set_ylabel("Coverage", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(y=0.95, color="green", linestyle="--", alpha=0.4, label="Target")

    ax2 = ax1.twinx()
    ax2.errorbar(thresh_vals, auroc_means, yerr=auroc_stds, fmt="s--",
                 color=color2, capsize=4, label="AUROC")
    ax2.set_ylabel("AUROC", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")

    ax1.set_title("PA-CP: PUF Threshold vs Coverage-AUROC Tradeoff", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fsi_threshold_sweep.pdf")
    fig.savefig(FIG_DIR / "fsi_threshold_sweep.png")
    plt.close()
    print("  fsi_threshold_sweep.pdf")


def fig_polarity_violin():
    """2xN panel: feature distributions (normal vs anomalous) across domains."""
    path = TABLE_DIR / "fsi_analysis.json"
    if not path.exists():
        print("  Skipping polarity violin (no data)")
        return

    data = json.loads(path.read_text())
    per_seed = data.get("per_seed_results", [])
    if not per_seed:
        print("  Skipping polarity violin (empty data)")
        return

    seed0 = per_seed[0]
    domains = sorted(seed0.get("per_domain_correlations", {}).keys())
    features = data.get("features", [])
    if not domains or not features:
        print("  Skipping polarity violin (no domain/feature data)")
        return

    n_domains = len(domains)
    n_features = len(features)

    fig, axes = plt.subplots(n_features, 1, figsize=(max(8, n_domains * 1.8), n_features * 3))
    if n_features == 1:
        axes = [axes]

    corr_data = seed0["per_domain_correlations"]
    for fi, feat in enumerate(features):
        ax = axes[fi]
        rhos = [corr_data.get(d, {}).get(feat, 0.0) for d in domains]
        colors = ["#F44336" if r < 0 else "#2196F3" for r in rhos]

        ax.bar(range(n_domains), rhos, color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xticks(range(n_domains))
        ax.set_xticklabels(domains, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"Feature: {feat}", fontweight="bold", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(-1.0, 1.0)

    plt.suptitle("Per-Domain Feature-Label Correlations (red = negative = potential reversal)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "polarity_violin.pdf")
    fig.savefig(FIG_DIR / "polarity_violin.png")
    plt.close()
    print("  polarity_violin.pdf")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")
    fig_baselines_comparison()
    fig_conformal_coverage()
    fig_tau_adaptation()
    fig_shift_detection()
    fig_ablation_comparison()
    fig_multiseed_summary()
    fig_lipschitz_bound()
    fig_cal_sweep()
    fig_cp_comparison()
    fig_shift_experiment()
    fig_held_out_attacks()
    fig_consequence_comparison()
    fig_weight_sweep()
    fig_cal_sweep_consequence()
    fig_trail_transfer()
    fig_toolbench_transfer()
    fig_agentharm_eval()
    fig_coverage_contamination()

    # v11: Novelty elevation figures
    fig_rac_comparison()
    fig_per_tool_coverage()
    fig_speculative_regret()
    fig_three_way_decision()

    # v13: Leakage analysis + architecture ablation
    fig_leakage_decomposition()
    fig_lstm_outcome_comparison()

    # v13-reframe: Scorer-agnostic CP narrative
    fig_coverage_vs_auroc_scatter()

    # v18: ATBench + RAC SWE-agent
    fig_atbench_eval()
    fig_rac_swe_agent()

    # v22: Hybrid scorer evaluation + domain inversion
    fig_hybrid_comparison()
    fig_hybrid_cp()
    fig_transfer_learning()
    fig_domain_inversion()

    # v26: FSI + PA-CP + Adversarial domain
    fig_fsi_heatmap()
    fig_pacp_coverage()
    fig_adversarial_scatter()
    fig_fsi_threshold_sweep()
    fig_polarity_violin()

    print("\nGenerating tables...")
    table_main_results()

    print(f"\nDone. Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
