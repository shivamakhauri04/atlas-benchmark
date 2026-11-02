"""Write LaTeX table bodies for the TMLR appendix directly from the v50 result JSON files."""

import json
from pathlib import Path

T = Path("output/tables")
OUT = Path("output/paper/tables")
POLICIES = ("always_reuse", "always_retrain", "sign_gate", "direct_gate", "divergence_gate")


def e2b_rows() -> str:
    r = json.loads((T / "v50_reuse_agentdojo.json").read_text())
    lines = []
    for fs, fs_name in (("all", "all six"), ("length_free", "length-free")):
        for det, det_name in (("rf", "RF"), ("knn", "kNN"), ("lr", "LR")):
            for k in (10, 20, 50, 100):
                v = r[f"{fs}/{det}/k={k}"]
                regs = " & ".join(f"{v['regret'][p]:.3f}" for p in POLICIES)
                lines.append(f"{fs_name} & {det_name} & {k} & {regs} & "
                             f"{100 * v['reuse_wins_share']:.0f} \\\\")
    return "\n".join(lines) + "\n"


def config_rows() -> str:
    included = json.loads((T / "v50_cross_agent.json").read_text())["backbones"]
    return "\n".join(f"\\texttt{{{c.replace('_', chr(92) + '_')}}} \\\\" for c in included) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Reason: \\input inside a tabular breaks booktabs rules, so each file holds a whole tabular.
    (OUT / "e2b_table.tex").write_text(
        "\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n"
        "Features & Detector & $k$ & Reuse & Retrain & Sign & AUROC & Distance & Reuse wins (\\%) \\\\\n"
        "\\midrule\n" + e2b_rows() + "\\bottomrule\n\\end{tabular}\n")
    (OUT / "config_table.tex").write_text(
        "\\begin{tabular}{@{}l@{}}\n\\toprule\n" + config_rows() + "\\bottomrule\n\\end{tabular}\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
