"""Measure provenance rule R2 for each injected dataset, without using labels.

R2 marks a feature construction-tied when the dataset's anomaly injector, applied to clean traces,
shifts the feature mean by at least 0.2 pooled SD. Each clean trace is compared with its own
injected copy, so no anomaly label is read.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from data.domain_loader import _inject_api_misuse, compute_arg_features, compute_trace_heuristics
from data.schema import Action, Trace
from data.trace_loader import (
    _inject_anomaly,
    generate_synthetic_traces,
    load_apibank,
    load_toolace,
    load_toolbench,
)

OUT = Path("output/tables/v50_r2_shift.json")
R2_THRESHOLD = 0.2
MIN_TRACES = 1000
SEED = 0

Injector = Callable[[list[Action], random.Random], list[Action]]


def _features(traces: list[list[Action]]) -> dict[str, np.ndarray]:
    feats = compute_trace_heuristics([[a.tool for a in t] for t in traces])
    feats.update(compute_arg_features([[a.args for a in t] for t in traces]))
    return {k: np.asarray(v, dtype=float) for k, v in feats.items()}


def r2_shift(clean: list[Trace], inject: Injector) -> dict[str, dict[str, float]]:
    """Return the standardized mean shift per feature from injecting every clean trace."""
    rng = random.Random(SEED)
    before = [list(t.actions) for t in clean]
    after = [inject(list(acts), rng) for acts in before]
    fb, fa = _features(before), _features(after)
    out: dict[str, dict[str, float]] = {}
    for name in fb:
        sd = float(np.sqrt((fb[name].var(ddof=1) + fa[name].var(ddof=1)) / 2)) or 1e-12
        shift = float((fa[name] - fb[name]).mean() / sd)
        out[name] = {
            "std_shift": shift,
            "tied_r2": abs(shift) >= R2_THRESHOLD,
            "predicted_sign": int(np.sign(shift)) if abs(shift) >= R2_THRESHOLD else 0,
        }
    return out


def _api_misuse(clean: list[Trace]) -> Injector:
    tools = sorted({a.tool for t in clean for a in t.actions})
    return lambda acts, rng: _inject_api_misuse(acts, tools, rng)


def main() -> None:
    clean_sets: dict[str, list[Trace]] = {
        "synthetic": generate_synthetic_traces(n_traces=2000, anomaly_rate=0.0, seed=42),
        "toolbench": load_toolbench(Path("data"), max_traces=5000),
        "toolace": load_toolace(max_traces=5000),
        "apibank": load_apibank(max_traces=5000),
    }
    result: dict[str, object] = {"threshold": R2_THRESHOLD, "seed": SEED, "datasets": {}}
    for name, clean in clean_sets.items():
        inject = _api_misuse(clean) if name in ("toolace", "apibank") else _inject_anomaly
        result["datasets"][name] = {  # type: ignore[index]
            "n_clean": len(clean),
            "meets_min_traces": len(clean) >= MIN_TRACES,
            "lengths": {
                "median": float(np.median([len(t.actions) for t in clean])),
                "share_below_10": float(np.mean([len(t.actions) < 10 for t in clean])),
            },
            "features": r2_shift(clean, inject),
        }
        print(name, len(clean))
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
