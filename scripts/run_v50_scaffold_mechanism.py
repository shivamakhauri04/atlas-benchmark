"""Why do defense scaffolds flip feature polarity? Per-class run statistics on AgentDojo.

For each base model and its defense variants (attack `important_instructions`), report the attack
success rate, the mean tool calls in successful and resisted runs, Cohen's d for trace length, and
the share of tool outputs the defense replaced (the PI detector writes
"<Data omitted because a prompt injection was detected>").
"""

import json
from pathlib import Path

import numpy as np

from traceshift.v50 import cohens_d_anomalous

RUNS = Path("/tmp/agentdojo_data/runs")
OUT = Path("output/tables/v50_scaffold_mechanism.json")
FAMILIES = {
    "gpt-4o": ["gpt-4o-2024-05-13", "gpt-4o-2024-05-13-repeat_user_prompt",
               "gpt-4o-2024-05-13-spotlighting_with_delimiting", "gpt-4o-2024-05-13-tool_filter",
               "gpt-4o-2024-05-13-transformers_pi_detector"],
    "llama-3.3-70b": ["meta-llama_Llama-3.3-70B-Instruct",
                      "meta-llama_Llama-3.3-70B-Instruct-repeat_user_prompt"],
    "meta-secalign-70b": ["Meta-SecAlign-70B", "Meta-SecAlign-70B-repeat_user_prompt"],
}
OMITTED = "prompt injection was detected"


def _stats(run: str) -> dict[str, float]:
    calls: list[int] = []
    labels: list[int] = []
    tool_msgs = omitted = 0
    for f in sorted((RUNS / run).glob("*/*/important_instructions/*.json")):
        rec = json.loads(f.read_text())
        if rec.get("security") is None:
            continue
        msgs = rec.get("messages", [])
        calls.append(sum(len(m.get("tool_calls") or []) for m in msgs if m.get("role") == "assistant"))
        labels.append(int(bool(rec["security"])))
        for m in msgs:
            if m.get("role") == "tool":
                tool_msgs += 1
                omitted += OMITTED in str(m.get("content", ""))
    x, y = np.array(calls, float), np.array(labels)
    return {
        "n_runs": int(len(y)), "attack_success_rate": float(y.mean()),
        "mean_calls_success": float(x[y == 1].mean()) if y.sum() else float("nan"),
        "mean_calls_resisted": float(x[y == 0].mean()),
        "d_trace_length": float(cohens_d_anomalous(x, y)),
        "share_tool_outputs_omitted": float(omitted / tool_msgs) if tool_msgs else 0.0,
    }


def main() -> None:
    result = {fam: {run: _stats(run) for run in runs if (RUNS / run).exists()}
              for fam, runs in FAMILIES.items()}
    OUT.write_text(json.dumps(result, indent=2))
    for fam, runs in result.items():
        for run, s in runs.items():
            print(f"{run:50s} n={s['n_runs']:4d} ASR={s['attack_success_rate']:.3f} "
                  f"calls succ={s['mean_calls_success']:.2f} res={s['mean_calls_resisted']:.2f} "
                  f"d_len={s['d_trace_length']:+.2f} omitted={s['share_tool_outputs_omitted']:.3f}")


if __name__ == "__main__":
    main()
