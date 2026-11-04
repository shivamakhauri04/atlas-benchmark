"""Domain-level feature extraction for cross-domain analysis.

Loads each dataset and extracts heuristic features (trace_length,
low_diversity, repeat_rate) as numpy arrays alongside binary labels.
Used by v26 experiment scripts (FSI, PA-CP, adversarial).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

MIN_TRACE_LEN = 3
MAX_TRACE_LEN = 200


def compute_trace_heuristics(
    tools_list: list[list[str]],
) -> dict[str, np.ndarray]:
    """Compute heuristic features from tool sequences."""
    n = len(tools_list)
    lengths = np.zeros(n)
    low_diversity = np.zeros(n)
    repeat_rate = np.zeros(n)

    for i, tools in enumerate(tools_list):
        n_actions = len(tools)
        lengths[i] = float(n_actions)
        unique_tools = len(set(tools))
        low_diversity[i] = 1.0 - (unique_tools / max(n_actions, 1))
        if n_actions > 1:
            repeats = sum(
                1 for j in range(1, n_actions) if tools[j] == tools[j - 1]
            )
            repeat_rate[i] = repeats / (n_actions - 1)

    return {
        "trace_length": lengths,
        "low_diversity": low_diversity,
        "repeat_rate": repeat_rate,
    }


def compute_arg_features(
    args_list: list[list[str]],
) -> dict[str, np.ndarray]:
    """Compute argument-based features from action args strings.

    Extracts 3 features that capture argument complexity and structure:
    - mean_arg_length: average character length of args per trace
    - arg_diversity: ratio of unique arg strings to total actions
    - arg_empty_rate: fraction of actions with empty/minimal args
    """
    import math

    n = len(args_list)
    mean_arg_length = np.zeros(n)
    arg_diversity = np.zeros(n)
    arg_empty_rate = np.zeros(n)

    for i, args in enumerate(args_list):
        n_actions = len(args)
        if n_actions == 0:
            continue
        lengths = [len(a) for a in args]
        mean_arg_length[i] = sum(lengths) / n_actions
        unique_args = len(set(args))
        arg_diversity[i] = unique_args / n_actions
        n_empty = sum(1 for a in args if len(a.strip()) <= 2)
        arg_empty_rate[i] = n_empty / n_actions

    return {
        "mean_arg_length": mean_arg_length,
        "arg_diversity": arg_diversity,
        "arg_empty_rate": arg_empty_rate,
    }


DomainData = tuple[dict[str, np.ndarray], np.ndarray]


def _extract_all_features(traces: list) -> dict[str, np.ndarray]:
    """Extract both structural and argument features from traces."""
    tools_list = [[a.tool for a in t.actions] for t in traces]
    args_list = [[a.args for a in t.actions] for t in traces]
    features = compute_trace_heuristics(tools_list)
    features.update(compute_arg_features(args_list))
    return features


def load_synthetic_domain(seed: int = 42) -> DomainData:
    """Load synthetic traces and extract features."""
    from data.trace_loader import generate_synthetic_traces

    traces = generate_synthetic_traces(
        n_traces=2000, anomaly_rate=0.15, seed=seed,
    )
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    return _extract_all_features(traces), labels


def load_swe_agent_domain(max_traces: int = 5000) -> DomainData:
    """Load SWE-agent traces and extract features."""
    from datasets import load_dataset

    print("  Loading SWE-agent traces (streaming)...")
    ds = load_dataset(
        "nebius/SWE-agent-trajectories", split="train", streaming=True,
    )

    traces = []
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions = []
        action_args = []
        for turn in example["trajectory"]:
            if turn.get("role") == "ai" and turn.get("text"):
                blocks = re.findall(
                    r"```\s*\n?(.*?)\n?```", turn["text"], re.DOTALL,
                )
                for block in blocks:
                    block = block.strip()
                    if not block:
                        continue
                    first_line = block.split("\n")[0].strip()
                    cmd = first_line.split()[0] if first_line.split() else ""
                    cmd = cmd.lstrip("./!$")
                    if cmd:
                        actions.append(cmd)
                        action_args.append(block)
        if MIN_TRACE_LEN <= len(actions) <= MAX_TRACE_LEN:
            traces.append({
                "actions": actions,
                "args": action_args,
                "solved": bool(example["target"]),
            })

    tools_list = [t["actions"] for t in traces]
    args_list = [t["args"] for t in traces]
    labels = np.array([0 if t["solved"] else 1 for t in traces])
    features = compute_trace_heuristics(tools_list)
    features.update(compute_arg_features(args_list))
    print(f"  SWE-agent: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_atbench_domain() -> DomainData:
    """Load ATBench traces and extract features."""
    from data.trace_loader import load_atbench

    print("  Loading ATBench...")
    traces = load_atbench()
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    features = _extract_all_features(traces)
    print(f"  ATBench: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_agentharm_domain() -> DomainData:
    """Load AgentHarm traces and extract features."""
    from data.trace_loader import load_agentharm

    print("  Loading AgentHarm...")
    traces = load_agentharm()
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    features = _extract_all_features(traces)
    print(f"  AgentHarm: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_rjudge_domain(
    max_traces: int | None = None,
) -> DomainData:
    """Load R-Judge traces and extract features."""
    from data.trace_loader import load_rjudge

    print("  Loading R-Judge...")
    traces = load_rjudge(Path("data"), max_traces=max_traces)
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    features = _extract_all_features(traces)
    print(f"  R-Judge: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_injecagent_domain(
    max_traces: int | None = None,
) -> DomainData:
    """Load InjecAgent traces and extract features."""
    from data.trace_loader import load_injecagent

    print("  Loading InjecAgent...")
    traces = load_injecagent(Path("data"), max_traces=max_traces)
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    features = _extract_all_features(traces)
    print(f"  InjecAgent: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_toolbench_domain(
    max_traces: int = 5000,
    anomaly_rate: float = 0.10,
    seed: int = 42,
) -> DomainData:
    """Load ToolBench traces with injected anomalies and extract features.

    ToolBench traces are all normal (real API tool-use). We inject synthetic
    anomalies at a controlled rate to create labeled data for AD evaluation.
    """
    import random

    from data.trace_loader import _inject_anomaly, load_toolbench

    print("  Loading ToolBench...")
    traces = load_toolbench(Path("data"), max_traces=max_traces)
    if not traces:
        raise ValueError("No ToolBench traces loaded")

    rng = random.Random(seed)
    tools_list: list[list[str]] = []
    args_list: list[list[str]] = []
    label_list: list[int] = []

    for trace in traces:
        if rng.random() < anomaly_rate:
            new_actions = _inject_anomaly(list(trace.actions), rng)
            tools_list.append([a.tool for a in new_actions])
            args_list.append([a.args for a in new_actions])
            label_list.append(1)
        else:
            tools_list.append([a.tool for a in trace.actions])
            args_list.append([a.args for a in trace.actions])
            label_list.append(0)

    labels = np.array(label_list)
    features = compute_trace_heuristics(tools_list)
    features.update(compute_arg_features(args_list))
    print(f"  ToolBench: {len(traces)} traces, {labels.sum()} anomalous "
          f"(injected at {anomaly_rate:.0%})")
    return features, labels


def load_agentdojo_domain(
    data_dir: Path | None = None,
    max_traces: int | None = None,
) -> DomainData:
    """Load AgentDojo traces and extract features."""
    from data.trace_loader import load_agentdojo

    if data_dir is None:
        data_dir = Path("/tmp/agentdojo_data/runs/gpt-4o-2024-05-13")

    print("  Loading AgentDojo...")
    traces = load_agentdojo(data_dir, max_traces=max_traces, min_tool_calls=1)
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    features = _extract_all_features(traces)
    print(f"  AgentDojo: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_trail_domain(
    trail_dir: Path | None = None,
    auto_download: bool = True,
    error_threshold: float = 0.3,
) -> DomainData:
    """Load TRAIL benchmark traces and extract features.

    Uses a threshold-based anomaly definition: a trace is anomalous if
    the fraction of error-labeled actions exceeds ``error_threshold``.
    Raw TRAIL traces have ~97% anomaly rate at trace level (any-error),
    but only ~19% at action level. The threshold gives a balanced split.
    """
    from data.trace_loader import download_trail, load_trail

    if trail_dir is None:
        trail_dir = Path("data/trail/TRAIL-Benchmark")

    if not trail_dir.exists() and auto_download:
        print("  Downloading TRAIL benchmark...")
        trail_dir = download_trail(trail_dir.parent)

    print("  Loading TRAIL...")
    traces = load_trail(trail_dir)
    if not traces:
        raise ValueError("No TRAIL traces loaded")

    label_list = []
    for t in traces:
        n_actions = len(t.actions)
        n_errors = sum(1 for a in t.actions if a.label == 1)
        error_frac = n_errors / max(n_actions, 1)
        label_list.append(1 if error_frac > error_threshold else 0)
    labels = np.array(label_list)
    features = _extract_all_features(traces)
    print(f"  TRAIL: {len(traces)} traces, {labels.sum()} anomalous "
          f"(threshold: >{error_threshold:.0%} actions errored)")
    return features, labels


def _inject_api_misuse(
    actions: list,
    all_tools: list[str],
    rng: "random.Random",
) -> list:
    """Inject API-misuse anomalies that preserve trace length.

    Unlike the generic synthetic injector (which adds/removes actions),
    this creates anomalies by corrupting tool calls in place:
    - Wrong function: replace tool with a random other tool (50%)
    - Arg scramble: swap args between two actions (25%)
    - Arg corruption: empty or truncate args (25%)

    This produces anomalies where trace_length is unchanged but
    low_diversity and repeat_rate patterns shift — matching the
    API-misuse semantic group's expected failure mode.
    """
    from data.schema import Action

    if len(actions) < 2:
        return actions

    new_actions = list(actions)
    n_corrupt = max(1, rng.randint(1, len(new_actions) // 2 + 1))
    positions = rng.sample(range(len(new_actions)), min(n_corrupt, len(new_actions)))

    for pos in positions:
        a = new_actions[pos]
        strategy = rng.random()

        if strategy < 0.50:
            # Wrong function: replace tool name
            wrong_tool = rng.choice(all_tools)
            new_actions[pos] = Action(
                tool=wrong_tool,
                args=a.args,
                context=a.context,
                timestamp=a.timestamp,
                label=1,
                outcome=a.outcome,
                outcome_label=1,
            )
        elif strategy < 0.75:
            # Arg scramble: use args from a different position
            other = rng.choice(range(len(new_actions)))
            new_actions[pos] = Action(
                tool=a.tool,
                args=new_actions[other].args,
                context=a.context,
                timestamp=a.timestamp,
                label=1,
                outcome=a.outcome,
                outcome_label=1,
            )
        else:
            # Arg corruption: empty or truncate
            corrupted = "{}" if rng.random() < 0.5 else a.args[:5]
            new_actions[pos] = Action(
                tool=a.tool,
                args=corrupted,
                context=a.context,
                timestamp=a.timestamp,
                label=1,
                outcome=a.outcome,
                outcome_label=1,
            )

    return new_actions


def load_apibank_domain(
    max_traces: int = 5000,
    anomaly_rate: float = 0.10,
    seed: int = 42,
) -> DomainData:
    """Load API-Bank traces with API-misuse anomalies and extract features.

    API-Bank traces are all normal (correct API call sequences across
    73 tools and 1,000 domains). We inject API-specific anomalies
    (wrong function, arg scramble, arg corruption) that preserve trace
    length while corrupting the tool-call content. This produces a
    different anomaly signature from ToolBench's generic injection.
    """
    import random

    from data.trace_loader import load_apibank

    print("  Loading API-Bank...")
    traces = load_apibank(max_traces=max_traces)
    if not traces:
        raise ValueError("No API-Bank traces loaded")

    # Build tool vocabulary for wrong-function injection
    all_tools = sorted({a.tool for t in traces for a in t.actions})

    rng = random.Random(seed)
    tools_list: list[list[str]] = []
    args_list: list[list[str]] = []
    label_list: list[int] = []

    for trace in traces:
        if rng.random() < anomaly_rate:
            new_actions = _inject_api_misuse(
                list(trace.actions), all_tools, rng,
            )
            tools_list.append([a.tool for a in new_actions])
            args_list.append([a.args for a in new_actions])
            label_list.append(1)
        else:
            tools_list.append([a.tool for a in trace.actions])
            args_list.append([a.args for a in trace.actions])
            label_list.append(0)

    labels = np.array(label_list)
    features = compute_trace_heuristics(tools_list)
    features.update(compute_arg_features(args_list))
    print(f"  API-Bank: {len(traces)} traces, {labels.sum()} anomalous "
          f"(API-misuse injection at {anomaly_rate:.0%})")
    return features, labels


def load_toolace_domain(
    max_traces: int = 5000,
    anomaly_rate: float = 0.15,
    seed: int = 42,
) -> DomainData:
    """Load ToolACE traces with length-preserving anomalies and extract features.

    ToolACE traces are all normal (correct multi-turn tool usage across
    26,507 APIs). We inject anomalies using in-place corruption (wrong
    function, arg scramble, arg corruption) that preserves trace length.
    This ensures trace_length polarity aligns with Synthetic (both in
    the Injected semantic group) rather than reversing due to action
    insertion.
    """
    import random

    from data.trace_loader import load_toolace

    print("  Loading ToolACE...")
    traces = load_toolace(max_traces=max_traces)
    if not traces:
        raise ValueError("No ToolACE traces loaded")

    all_tools = sorted({a.tool for t in traces for a in t.actions})

    rng = random.Random(seed)
    tools_list: list[list[str]] = []
    args_list: list[list[str]] = []
    label_list: list[int] = []

    for trace in traces:
        if rng.random() < anomaly_rate:
            new_actions = _inject_api_misuse(list(trace.actions), all_tools, rng)
            tools_list.append([a.tool for a in new_actions])
            args_list.append([a.args for a in new_actions])
            label_list.append(1)
        else:
            tools_list.append([a.tool for a in trace.actions])
            args_list.append([a.args for a in trace.actions])
            label_list.append(0)

    labels = np.array(label_list)
    features = compute_trace_heuristics(tools_list)
    features.update(compute_arg_features(args_list))
    print(f"  ToolACE: {len(traces)} traces, {labels.sum()} anomalous "
          f"(injected at {anomaly_rate:.0%})")
    return features, labels


def load_stabletoolbench_domain(
    data_dir: Path | None = None,
    model: str = "chatgpt_cot",
    max_traces: int | None = None,
) -> DomainData:
    """Load StableToolBench traces with organic pass/fail labels.

    Unlike ToolBench and API-Bank (injected anomalies), StableToolBench
    labels come from GPT-4 ToolEval judgments on real API task completion.
    Failed traces represent organic API-misuse/failure patterns.
    """
    from data.trace_loader import load_stabletoolbench

    print("  Loading StableToolBench...")
    traces = load_stabletoolbench(
        data_dir=data_dir, model=model, max_traces=max_traces,
    )
    if not traces:
        raise ValueError("No StableToolBench traces loaded")
    labels = np.array([1 if t.has_anomaly else 0 for t in traces])
    features = _extract_all_features(traces)
    print(f"  StableToolBench: {len(traces)} traces, {labels.sum()} anomalous")
    return features, labels


def load_all_domains(
    *,
    seeds: list[int] | None = None,
    max_swe_traces: int = 5000,
    max_rjudge: int | None = None,
    max_injecagent: int | None = None,
    max_toolbench: int = 5000,
    max_agentdojo: int | None = None,
    max_toolace: int = 5000,
    max_apibank: int = 5000,
    include_toolbench: bool = True,
    include_trail: bool = True,
    include_agentdojo: bool = True,
    include_toolace: bool = True,
    include_apibank: bool = True,
    include_stabletoolbench: bool = True,
    allow_missing: bool = False,
) -> dict[str, DomainData]:
    """Load all domains. Raise if one fails, unless `allow_missing` is set."""
    seed = seeds[0] if seeds else 42
    domain_data: dict[str, DomainData] = {}

    domain_data["synthetic"] = load_synthetic_domain(seed=seed)

    loaders: list[tuple[str, callable]] = [
        ("swe_agent", lambda: load_swe_agent_domain(max_swe_traces)),
        ("atbench", load_atbench_domain),
        ("agentharm", load_agentharm_domain),
        ("rjudge", lambda: load_rjudge_domain(max_rjudge)),
        ("injecagent", lambda: load_injecagent_domain(max_injecagent)),
    ]
    if include_toolbench:
        loaders.append(
            ("toolbench", lambda: load_toolbench_domain(max_toolbench, seed=seed)),
        )
    if include_trail:
        loaders.append(("trail", load_trail_domain))
    if include_agentdojo:
        loaders.append(
            ("agentdojo", lambda: load_agentdojo_domain(max_traces=max_agentdojo)),
        )
    if include_toolace:
        loaders.append(
            ("toolace", lambda: load_toolace_domain(max_toolace, seed=seed)),
        )
    if include_apibank:
        loaders.append(
            ("apibank", lambda: load_apibank_domain(max_apibank, seed=seed)),
        )
    if include_stabletoolbench:
        loaders.append(
            ("stabletoolbench", load_stabletoolbench_domain),
        )

    for name, loader in loaders:
        try:
            domain_data[name] = loader()
        except Exception as e:
            if not allow_missing:
                raise RuntimeError(f"domain {name!r} failed to load") from e
            print(f"  {name} loading failed (allow_missing=True): {e}")

    return domain_data
