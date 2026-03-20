"""Trace loader for agent benchmarks.

Supports loading and normalizing traces from:
- ToolBench (tool-use trajectories, ReAct format)
- TRAIL (PatronusAI OTel span traces)
- AgentHarm (paired benign/harmful tool-use scenarios)
- Synthetic traces (for development and smoke testing)
- Custom MCP trace format (JSONL)

All datasets are converted to the canonical Trace/Action schema.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import orjson
import torch
from torch.utils.data import Dataset

from data.schema import Action, Trace

# ---------------------------------------------------------------------------
# ToolBench parser
# ---------------------------------------------------------------------------

_REACT_ACTION_PREFIX = "Action: "
_REACT_INPUT_PREFIX = "Action Input: "


def _parse_react_turn(text: str) -> tuple[str, str, str] | None:
    """Extract (thought, tool_name, args_json) from a ReAct assistant turn.

    Returns None if the turn is not a valid tool call.
    """
    action_idx = text.find(_REACT_ACTION_PREFIX)
    input_idx = text.find(_REACT_INPUT_PREFIX)
    if action_idx == -1 or input_idx == -1:
        return None

    thought = text[:action_idx].replace("Thought: ", "").strip()
    tool_name = text[action_idx + len(_REACT_ACTION_PREFIX) : input_idx].strip()
    args_str = text[input_idx + len(_REACT_INPUT_PREFIX) :].strip()
    return thought, tool_name, args_str


def _conversation_to_trace(conv: list[dict], trace_id: str) -> Trace | None:
    """Convert a ToolBench conversation to a normalized Trace."""
    actions: list[Action] = []
    task = ""
    t = 0.0

    for msg in conv:
        role = msg.get("from", msg.get("role", ""))
        value = msg.get("value", msg.get("content", ""))

        if role == "user" and not task:
            task = value[:500]  # truncate long prompts

        if role == "assistant":
            parsed = _parse_react_turn(value)
            if parsed is None:
                continue
            thought, tool_name, args_str = parsed
            if tool_name.lower() == "finish":
                continue  # skip terminal action
            actions.append(
                Action(
                    tool=tool_name,
                    args=args_str[:2000],  # cap very long args
                    context=thought[:500],
                    timestamp=t,
                    label=0,  # ToolBench = normal traces
                )
            )
            t += 1.0

    if len(actions) < 2:
        return None  # skip trivially short traces
    return Trace(
        trace_id=trace_id,
        actions=actions,
        source="toolbench",
        task=task,
    )


def load_toolbench(data_dir: Path, *, max_traces: int | None = None) -> list[Trace]:
    """Load and normalize ToolBench traces.

    Expects ``data_dir / toolbench_train.json`` (the flattened training file
    from the ToolBench release).
    """
    train_path = data_dir / "toolbench_train.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"ToolBench training file not found at {train_path}. "
            "Download from https://drive.google.com/drive/folders/"
            "1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL and rename to toolbench_train.json"
        )

    raw = orjson.loads(train_path.read_bytes())
    traces: list[Trace] = []

    items = raw if isinstance(raw, list) else [raw]
    for i, item in enumerate(items):
        if max_traces is not None and len(traces) >= max_traces:
            break
        conv = item.get("conversations", [])
        if not conv:
            continue
        tid = f"toolbench_{i:06d}"
        trace = _conversation_to_trace(conv, tid)
        if trace is not None:
            traces.append(trace)

    return traces


# ---------------------------------------------------------------------------
# Synthetic trace generator (v3: scaled 107-tool registry, 20 workflows)
# ---------------------------------------------------------------------------

from data.tool_registry import (  # noqa: E402, I001
    _DEFAULT_OUTCOMES,
    ALL_TOOLS,
    ANOMALY_PATTERNS as _ANOMALY_PATTERNS,
    TOOL_OUTCOMES as _TOOL_OUTCOMES,
    WORKFLOW_TEMPLATES as _WORKFLOW_TEMPLATES,
)

# Flat list of (tool_name, dummy_args) for auxiliary padding actions.
_ALL_TOOL_ACTIONS = [(name, "{}") for name, _risky in ALL_TOOLS]


def _sample_outcome(
    tool: str,
    is_anomalous: bool,
    rng: random.Random,
    *,
    noise_p: float = 0.0,
) -> tuple[str, int]:
    """Sample an outcome string and label for a tool call.

    Args:
        tool: Tool name.
        is_anomalous: Whether this action is anomalous (label=1).
        rng: Random instance for reproducibility.
        noise_p: Probability that an anomalous action gets a normal-looking
            outcome (simulating realistic noisy environments). 0.0 = deterministic
            outcomes (default), 1.0 = outcomes carry no anomaly signal.

    Returns:
        (outcome_str, outcome_label) where outcome_label=1 means anomalous outcome.
    """
    outcomes = _TOOL_OUTCOMES.get(tool, _DEFAULT_OUTCOMES)
    if is_anomalous and outcomes.get("anomalous"):
        if noise_p > 0.0 and rng.random() < noise_p:
            normal = outcomes.get("normal", _DEFAULT_OUTCOMES["normal"])
            return rng.choice(normal), 0
        return rng.choice(outcomes["anomalous"]), 1
    normal = outcomes.get("normal", _DEFAULT_OUTCOMES["normal"])
    return rng.choice(normal), 0


def _sample_workflow_trace(
    rng: random.Random,
    min_length: int,
    max_length: int,
) -> list[Action]:
    """Build a normal trace from a workflow template with noise.

    When padding beyond the template length, 30% of extra actions come
    from the full tool registry (not just the current workflow) so that
    all 107 tools appear in the training data.
    """
    workflow = rng.choice(_WORKFLOW_TEMPLATES)
    target_len = rng.randint(min_length, max_length)

    template = list(workflow)

    if len(template) > target_len:
        template = template[:target_len]

    while len(template) < target_len:
        if rng.random() < 0.3:
            extra = rng.choice(_ALL_TOOL_ACTIONS)
        else:
            extra = rng.choice(workflow)
        pos = rng.randint(0, len(template))
        template.insert(pos, extra)

    if len(template) > 2 and rng.random() < 0.3:
        pos = rng.randint(0, len(template) - 2)
        template[pos], template[pos + 1] = template[pos + 1], template[pos]

    actions = []
    for t, (tool, args) in enumerate(template):
        outcome, outcome_lbl = _sample_outcome(tool, False, rng)
        actions.append(
            Action(
                tool=tool,
                args=args,
                context=f"step {t} of task",
                timestamp=float(t),
                label=0,
                outcome=outcome,
                outcome_label=outcome_lbl,
            )
        )
    return actions


def _inject_anomaly(
    actions: list[Action],
    rng: random.Random,
    allowed_pattern_indices: list[int] | None = None,
    *,
    noise_p: float = 0.0,
) -> list[Action]:
    """Inject an anomaly pattern into the second half of a trace."""
    if allowed_pattern_indices is not None:
        pool = [_ANOMALY_PATTERNS[i] for i in allowed_pattern_indices]
    else:
        pool = _ANOMALY_PATTERNS
    pattern = rng.choice(pool)

    half = max(1, len(actions) // 2)
    inject_pos = rng.randint(half, max(half, len(actions) - 1))

    new_actions = list(actions[:inject_pos])

    for tool, args in pattern:
        outcome, outcome_lbl = _sample_outcome(tool, True, rng, noise_p=noise_p)
        new_actions.append(
            Action(
                tool=tool,
                args=args,
                context=f"step {len(new_actions)} of task",
                timestamp=float(len(new_actions)),
                label=1,
                outcome=outcome,
                outcome_label=outcome_lbl,
            )
        )

    tail_count = min(rng.randint(0, 2), len(actions) - inject_pos)
    for a in actions[inject_pos : inject_pos + tail_count]:
        new_actions.append(
            Action(
                tool=a.tool,
                args=a.args,
                context=f"step {len(new_actions)} of task",
                timestamp=float(len(new_actions)),
                label=0,
            )
        )

    return new_actions


def generate_synthetic_traces(
    n_traces: int = 1000,
    *,
    anomaly_rate: float = 0.05,
    min_length: int = 5,
    max_length: int = 50,
    seed: int = 42,
    allowed_pattern_indices: list[int] | None = None,
    outcome_noise: float = 0.0,
) -> list[Trace]:
    """Generate synthetic agent traces with non-trivially separable anomalies.

    Uses a scaled registry of 107 tools across 12 categories, 20 workflow
    templates, and 15 anomaly patterns. Risky tools appear in both normal
    and anomalous contexts — separating them requires sequential pattern
    analysis, not tool-identity shortcuts.

    Args:
        allowed_pattern_indices: If set, only use these anomaly pattern
            indices (0-based) for injection. Used to create held-out
            attack generalization experiments.
    """
    rng = random.Random(seed)
    traces: list[Trace] = []

    for i in range(n_traces):
        is_anomalous = rng.random() < anomaly_rate
        actions = _sample_workflow_trace(rng, min_length, max_length)

        if is_anomalous:
            actions = _inject_anomaly(
                actions,
                rng,
                allowed_pattern_indices=allowed_pattern_indices,
                noise_p=outcome_noise,
            )

        traces.append(
            Trace(
                trace_id=f"synthetic_{i:06d}",
                actions=actions,
                source="synthetic",
                task=f"synthetic task {i}",
            )
        )

    return traces


def generate_shifted_traces(
    n_traces: int = 2000,
    *,
    shift_point: int = 1000,
    base_rate: float = 0.05,
    shifted_rate: float = 0.20,
    min_length: int = 5,
    max_length: int = 50,
    seed: int = 42,
) -> list[Trace]:
    """Generate traces with a distribution shift at `shift_point`.

    Before shift_point: anomaly rate = base_rate (stationary).
    After shift_point: anomaly rate = shifted_rate (non-stationary).

    Used to validate ACI (Theorem 2) and FA-CP (Theorem 3) under
    non-stationary conditions where Split CP assumptions break.
    """
    rng = random.Random(seed)
    traces: list[Trace] = []

    for i in range(n_traces):
        rate = base_rate if i < shift_point else shifted_rate
        is_anomalous = rng.random() < rate
        actions = _sample_workflow_trace(rng, min_length, max_length)

        if is_anomalous:
            actions = _inject_anomaly(actions, rng)

        traces.append(
            Trace(
                trace_id=f"shifted_{i:06d}",
                actions=actions,
                source="synthetic_shifted",
                task=f"shifted task {i}",
            )
        )

    return traces


# ---------------------------------------------------------------------------
# TRAIL dataset loader (PatronusAI/TRAIL-Benchmark)
# ---------------------------------------------------------------------------


def _flatten_trail_spans(span: dict, depth: int = 0) -> list[dict]:
    """Recursively flatten nested OTel spans into a flat list."""
    flat = [span]
    for child in span.get("child_spans", []):
        flat.extend(_flatten_trail_spans(child, depth + 1))
    return flat


def _trail_span_to_action(
    span: dict,
    error_span_ids: set[str],
    timestamp_offset: float,
) -> Action | None:
    """Convert a TRAIL OTel span to an Action if it represents a tool call or LLM step."""
    attrs = span.get("span_attributes", {})
    span_kind = attrs.get("openinference.span.kind", "")
    span_name = span.get("span_name", "unknown")
    span_id = span.get("span_id", "")

    tool_name = attrs.get("tool.name", "")
    if not tool_name and span_kind == "TOOL":
        tool_name = span_name
    if not tool_name and span_kind == "LLM":
        model = attrs.get("llm.model_name", "llm_call")
        tool_name = f"llm:{model}"
    if not tool_name:
        if "agent" in span_name.lower() or "run" in span_name.lower():
            tool_name = span_name.replace(" ", "_")
        else:
            return None

    args_str = str(attrs.get("input.value", attrs.get("tool.parameters", "")))[:2000]
    outcome_str = str(attrs.get("output.value", ""))[:2000]
    status = span.get("status_code", "OK")

    is_error_span = span_id in error_span_ids
    is_status_error = status == "ERROR"
    label = 1 if (is_error_span or is_status_error) else 0

    outcome_label = 1 if is_status_error else 0

    timestamp_str = span.get("timestamp", "")
    try:
        ts = float(timestamp_str) if timestamp_str else timestamp_offset
    except (ValueError, TypeError):
        ts = timestamp_offset

    return Action(
        tool=tool_name[:100],
        args=args_str,
        context=span_name[:500],
        timestamp=ts,
        label=label,
        outcome=outcome_str if outcome_str else status,
        outcome_label=outcome_label,
    )


def _load_trail_split(
    data_dir: Path,
    annotations_dir: Path,
    split_name: str,
) -> list[Trace]:
    """Load one TRAIL split (GAIA or SWE Bench)."""
    traces: list[Trace] = []

    for trace_file in sorted(data_dir.glob("*.json")):
        try:
            raw = json.loads(trace_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        trace_id = raw.get("trace_id", trace_file.stem)

        error_span_ids: set[str] = set()
        annotation_file = annotations_dir / trace_file.name
        if annotation_file.exists():
            try:
                ann = json.loads(annotation_file.read_text())
                for err in ann.get("errors", []):
                    loc = err.get("location", "")
                    if loc:
                        error_span_ids.add(loc)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        all_spans: list[dict] = []
        for top_span in raw.get("spans", []):
            all_spans.extend(_flatten_trail_spans(top_span))

        actions: list[Action] = []
        for i, span in enumerate(all_spans):
            action = _trail_span_to_action(span, error_span_ids, float(i))
            if action is not None:
                actions.append(action)

        if len(actions) < 2:
            continue

        traces.append(
            Trace(
                trace_id=f"trail_{split_name}_{trace_id}",
                actions=actions,
                source="trail",
                task=f"TRAIL {split_name}",
            )
        )

    return traces


def load_trail(
    trail_dir: Path,
    *,
    splits: list[str] | None = None,
) -> list[Trace]:
    """Load TRAIL benchmark traces from local checkout.

    Expects the repo structure:
        trail_dir/benchmarking/data/{GAIA,SWE Bench}/*.json
        trail_dir/benchmarking/processed_annotations_{gaia,swe_bench}/*.json

    Args:
        trail_dir: Path to TRAIL-Benchmark repo root.
        splits: Which splits to load. Default: ["GAIA", "SWE Bench"].

    Returns:
        List of Trace objects with error-annotated actions.
    """
    if splits is None:
        splits = ["GAIA", "SWE Bench"]

    split_configs = {
        "GAIA": ("GAIA", "processed_annotations_gaia"),
        "SWE Bench": ("SWE Bench", "processed_annotations_swe_bench"),
    }

    traces: list[Trace] = []
    for split in splits:
        data_name, ann_name = split_configs[split]
        data_dir = trail_dir / "benchmarking" / "data" / data_name
        ann_dir = trail_dir / "benchmarking" / ann_name

        if not data_dir.exists():
            print(f"  TRAIL split {split} not found at {data_dir}, skipping")
            continue

        split_traces = _load_trail_split(data_dir, ann_dir, split.replace(" ", "_").lower())
        traces.extend(split_traces)
        print(f"  Loaded {len(split_traces)} traces from TRAIL/{split}")

    return traces


def download_trail(target_dir: Path) -> Path:
    """Clone the TRAIL-Benchmark repo if not already present.

    Returns the path to the repo root.
    """
    repo_dir = target_dir / "TRAIL-Benchmark"
    if repo_dir.exists() and (repo_dir / "benchmarking").exists():
        return repo_dir

    import subprocess

    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/patronus-ai/TRAIL-Benchmark.git",
            str(repo_dir),
        ],
        check=True,
    )
    return repo_dir


# ---------------------------------------------------------------------------
# AgentHarm dataset (HuggingFace: ai-safety-institute/AgentHarm)
# ---------------------------------------------------------------------------


def load_agentharm(
    *,
    cache_dir: Path | None = None,
    max_traces: int | None = None,
) -> list[Trace]:
    """Load AgentHarm benchmark as paired benign/harmful traces.

    The AgentHarm dataset (ai-safety-institute/AgentHarm) contains 176
    paired scenarios: each harmful prompt has a benign counterpart with
    the same target tool sequence. This makes it ideal for testing
    whether anomaly detection can distinguish harmful intent from
    benign use of the same tools.

    Each row has ``target_functions`` (the intended tool sequence) and
    a ``prompt`` (the task description).  We convert each row into a
    Trace where actions correspond to the target tool sequence.

    Returns:
        Combined list of benign (label=0) and harmful (label=1) traces.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the datasets library: uv add --dev datasets"
        ) from exc

    kwargs: dict = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    benign_ds = load_dataset(
        "ai-safety-institute/AgentHarm",
        "harmless_benign",
        split="test_public",
        **kwargs,
    )
    harmful_ds = load_dataset(
        "ai-safety-institute/AgentHarm",
        "harmful",
        split="test_public",
        **kwargs,
    )

    traces: list[Trace] = []

    for split_label, ds in [("benign", benign_ds), ("harmful", harmful_ds)]:
        is_harmful = split_label == "harmful"
        for i, row in enumerate(ds):
            if max_traces is not None and len(traces) >= max_traces:
                break

            tools = row.get("target_functions") or []
            if not tools:
                continue

            prompt = (row.get("prompt") or "")[:500]
            category = row.get("category", "unknown")
            trace_label = 1 if is_harmful else 0

            actions = [
                Action(
                    tool=tool_name,
                    args=json.dumps({"task": prompt[:200]}),
                    context=f"category={category}",
                    label=trace_label,
                    outcome="completed",
                    outcome_label=trace_label,
                )
                for tool_name in tools
            ]

            traces.append(
                Trace(
                    trace_id=f"agentharm_{split_label}_{i:04d}",
                    actions=actions,
                    source="agentharm",
                    task=prompt,
                )
            )

    return traces


# ---------------------------------------------------------------------------
# ATBench dataset (AI45Research/ATBench — AgentDoG benchmark)
# ---------------------------------------------------------------------------


def _parse_atbench_trajectory(content: list[list[dict]]) -> list[Action]:
    """Extract tool-call Actions from ATBench trajectory groups.

    Each group is a list of turns with roles: user, agent, environment.
    Agent turns have an ``action`` field containing JSON tool calls.
    """
    actions: list[Action] = []
    t = 0.0

    for group in content:
        for turn in group:
            role = turn.get("role", "")
            if role != "agent":
                continue
            action_str = turn.get("action") or ""
            if not action_str:
                continue
            try:
                parsed = json.loads(action_str)
                tool_name = parsed.get("name", "unknown")
                args_raw = parsed.get("arguments", {})
                args_str = json.dumps(args_raw)[:2000]
            except (json.JSONDecodeError, TypeError):
                tool_name = "unknown"
                args_str = action_str[:2000]
            thought = turn.get("thought") or ""
            actions.append(
                Action(
                    tool=tool_name,
                    args=args_str,
                    context=thought[:500],
                    timestamp=t,
                    label=0,  # set at trace level, not action level
                    outcome="",
                    outcome_label=0,
                )
            )
            t += 1.0

    return actions


def load_atbench(
    *,
    cache_dir: Path | None = None,
    max_traces: int | None = None,
) -> list[Trace]:
    """Load ATBench (AgentDoG) benchmark as safe/unsafe agent trajectories.

    The ATBench dataset (AI45Research/ATBench) contains 500 trajectories
    (250 safe, 250 unsafe) with ~8.97 turns per trajectory and 1,575
    unique tools. Each unsafe trajectory is annotated with risk_source,
    failure_mode, and real_world_harm categories.

    Trajectories are converted to Trace objects where each agent tool
    call becomes an Action. The trace-level label (safe=0, unsafe=1)
    is propagated to all actions within the trace.

    Args:
        cache_dir: Optional HuggingFace cache directory.
        max_traces: Maximum number of traces to load.

    Returns:
        List of Trace objects with safety-labeled actions.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the datasets library: uv add --dev datasets"
        ) from exc

    kwargs: dict = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset("AI45Research/ATBench", split="test", **kwargs)

    traces: list[Trace] = []
    for i, row in enumerate(ds):
        if max_traces is not None and len(traces) >= max_traces:
            break

        content = row.get("content") or []
        trace_label = int(row.get("label", 0))
        risk_source = row.get("risk_source", "")
        failure_mode = row.get("failure_mode", "")
        conv_id = row.get("conv_id", f"atbench_{i:04d}")

        actions = _parse_atbench_trajectory(content)
        if len(actions) < 2:
            continue

        # Propagate trace-level label to all actions
        labeled_actions = [
            Action(
                tool=a.tool,
                args=a.args,
                context=a.context,
                timestamp=a.timestamp,
                label=trace_label,
                outcome=a.outcome,
                outcome_label=trace_label,
            )
            for a in actions
        ]

        traces.append(
            Trace(
                trace_id=conv_id,
                actions=labeled_actions,
                source="atbench",
                task=f"risk={risk_source} mode={failure_mode}",
            )
        )

    return traces


# ---------------------------------------------------------------------------
# R-Judge dataset (multi-turn agent safety evaluation)
# ---------------------------------------------------------------------------


def _parse_rjudge_turns(turn_groups: list[list[dict]]) -> list[Action]:
    """Extract actions from R-Judge conversation turn groups.

    R-Judge uses roles: user, agent, environment. Agent turns have
    ``thought`` and ``action`` fields. The action field contains
    tool calls as ``ToolName: {json_args}``.
    """
    actions: list[Action] = []
    t = 0.0

    for group in turn_groups:
        for i, msg in enumerate(group):
            role = msg.get("role", "")
            if role != "agent":
                continue

            action_str = msg.get("action") or ""
            thought = msg.get("thought") or ""
            if not action_str:
                continue

            if ": " in action_str:
                tool_name, args_str = action_str.split(": ", 1)
            else:
                tool_name = action_str
                args_str = "{}"

            outcome = ""
            outcome_label = 0
            if i + 1 < len(group):
                nxt = group[i + 1]
                if nxt.get("role") == "environment":
                    outcome = (nxt.get("content") or "")[:2000]
                    if "error" in outcome.lower():
                        outcome_label = 1

            actions.append(
                Action(
                    tool=tool_name[:100],
                    args=args_str[:2000],
                    context=thought[:500],
                    timestamp=t,
                    label=0,
                    outcome=outcome,
                    outcome_label=outcome_label,
                )
            )
            t += 1.0

    return actions


def load_rjudge(
    data_dir: Path,
    *,
    max_traces: int | None = None,
) -> list[Trace]:
    """Load R-Judge benchmark as safety-labeled agent traces.

    R-Judge (https://github.com/Lordog/R-Judge) contains ~564 multi-turn
    agent interactions with binary safety labels across 27 risk scenarios
    and 5 application categories.

    Expects data_dir / rjudge/ to be the cloned repo with JSON files
    under ``data/<Category>/<scenario>.json``. Each JSON file is a list
    of dicts with keys: id, scenario, contents (turn groups), label
    (0=safe, 1=unsafe), attack_type.

    Args:
        data_dir: Path to data directory containing rjudge/ subdirectory.
        max_traces: Maximum number of traces to load.

    Returns:
        List of Trace objects with safety labels (safe=0, unsafe=1).
    """
    rjudge_root = data_dir / "rjudge"
    rjudge_data = rjudge_root / "data"
    if not rjudge_data.exists():
        raise FileNotFoundError(
            f"R-Judge data not found at {rjudge_data}. "
            "Clone from https://github.com/Lordog/R-Judge"
        )

    traces: list[Trace] = []
    for json_path in sorted(rjudge_data.rglob("*.json")):
        if max_traces is not None and len(traces) >= max_traces:
            break
        try:
            raw = json.loads(json_path.read_text())
        except (json.JSONDecodeError, PermissionError):
            continue
        if not isinstance(raw, list):
            continue

        category = json_path.parent.name
        for idx, item in enumerate(raw):
            if max_traces is not None and len(traces) >= max_traces:
                break

            contents = item.get("contents", [])
            if isinstance(contents, str):
                try:
                    contents = json.loads(contents)
                except json.JSONDecodeError:
                    continue
            if not contents:
                continue

            label_str = str(item.get("label", "0"))
            trace_label = 1 if label_str in ("1", "unsafe") else 0
            scenario = item.get("scenario", json_path.stem)
            attack_type = item.get("attack_type", "")
            item_id = item.get("id", f"{idx}")

            actions = _parse_rjudge_turns(contents)
            if len(actions) < 2:
                continue

            labeled_actions = [
                Action(
                    tool=a.tool,
                    args=a.args,
                    context=a.context,
                    timestamp=a.timestamp,
                    label=trace_label,
                    outcome=a.outcome,
                    outcome_label=a.outcome_label,
                )
                for a in actions
            ]

            traces.append(
                Trace(
                    trace_id=f"rjudge_{category}_{item_id}",
                    actions=labeled_actions,
                    source="rjudge",
                    task=f"scenario={scenario} attack={attack_type}",
                )
            )

    return traces


# ---------------------------------------------------------------------------
# InjecAgent dataset (ACL 2024: injection attack traces)
# ---------------------------------------------------------------------------


def _parse_attacker_tools(raw: str | list) -> list[str]:
    """Parse the Attacker Tools field which may be a string repr of a list."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw.replace("'", '"'))
                return [str(t) for t in parsed]
            except json.JSONDecodeError:
                pass
        return [raw] if raw else []
    return []


def load_injecagent(
    data_dir: Path,
    *,
    max_traces: int | None = None,
) -> list[Trace]:
    """Load InjecAgent benchmark as attack-labeled agent traces.

    InjecAgent (ACL 2024 Findings) contains ~1,054 test cases with
    direct harm (dh) and data stealing (ds) attacks across 17 user
    tools and 62 attacker tools.

    Expects data_dir / injecagent/ to be the cloned repo. Reads
    ``data/test_cases_dh_base.json``, ``data/test_cases_ds_base.json``
    (attack traces, label=1) and ``data/user_cases.jsonl`` (benign
    traces, label=0).

    Each attack trace becomes: [User Tool] → [Attacker Tool(s)].
    Each benign trace becomes: [User Tool] (single action, padded
    with a thought step to reach min length 2).

    Args:
        data_dir: Path to data directory containing injecagent/ subdirectory.
        max_traces: Maximum number of traces to load.

    Returns:
        List of Trace objects with attack labels (benign=0, attack=1).
    """
    inject_root = data_dir / "injecagent"
    inject_data = inject_root / "data"
    if not inject_data.exists():
        raise FileNotFoundError(
            f"InjecAgent data not found at {inject_data}. "
            "Clone from https://github.com/uiuc-kang-lab/InjecAgent"
        )

    traces: list[Trace] = []

    attack_files = sorted(inject_data.glob("test_cases_*.json"))
    for json_path in attack_files:
        if max_traces is not None and len(traces) >= max_traces:
            break
        try:
            raw = json.loads(json_path.read_text())
        except (json.JSONDecodeError, PermissionError):
            continue
        if not isinstance(raw, list):
            continue

        attack_category = json_path.stem.replace("test_cases_", "")
        for idx, item in enumerate(raw):
            if max_traces is not None and len(traces) >= max_traces:
                break

            user_tool = item.get("User Tool", "")
            attacker_tools = _parse_attacker_tools(
                item.get("Attacker Tools", []),
            )
            attack_type = item.get("Attack Type", "")
            user_instr = item.get("User Instruction", "")[:500]
            tool_params = str(item.get("Tool Parameters", ""))[:2000]
            thought = item.get("Thought", "")[:500]

            tools = [user_tool] + attacker_tools
            tools = [t for t in tools if t]
            if len(tools) < 2:
                continue

            actions = []
            for t, tool_name in enumerate(tools):
                is_attacker = t > 0
                actions.append(
                    Action(
                        tool=tool_name[:100],
                        args=tool_params if t == 0 else "{}",
                        context=thought if t == 0 else "injected",
                        timestamp=float(t),
                        label=1,
                        outcome="",
                        outcome_label=1 if is_attacker else 0,
                    )
                )

            traces.append(
                Trace(
                    trace_id=f"injecagent_{attack_category}_{idx:04d}",
                    actions=actions,
                    source="injecagent",
                    task=f"attack={attack_type} {user_instr[:200]}",
                )
            )

    benign_path = inject_data / "user_cases.jsonl"
    if benign_path.exists():
        for line_num, line in enumerate(
            benign_path.read_text().strip().split("\n"),
        ):
            if not line.strip():
                continue
            if max_traces is not None and len(traces) >= max_traces:
                break
            item = json.loads(line)
            user_tool = item.get("User Tool", "")
            if not user_tool:
                continue
            user_instr = item.get("User Instruction", "")[:500]
            tool_params = str(item.get("Tool Parameters", ""))[:2000]
            thought = item.get("Thought", "")[:500]

            actions = [
                Action(
                    tool=user_tool[:100],
                    args=tool_params,
                    context=thought,
                    timestamp=0.0,
                    label=0,
                    outcome="",
                    outcome_label=0,
                ),
                Action(
                    tool=f"{user_tool}_response",
                    args="{}",
                    context="response",
                    timestamp=1.0,
                    label=0,
                    outcome="completed",
                    outcome_label=0,
                ),
            ]

            traces.append(
                Trace(
                    trace_id=f"injecagent_benign_{line_num:04d}",
                    actions=actions,
                    source="injecagent",
                    task=f"benign {user_instr[:200]}",
                )
            )

    return traces


# ---------------------------------------------------------------------------
# SWE-agent traces (HuggingFace streaming)
# ---------------------------------------------------------------------------


def load_swe_agent(*, max_traces: int = 5000) -> list[Trace]:
    """Load SWE-agent traces from HuggingFace as Trace objects."""
    import re

    from datasets import load_dataset

    ds = load_dataset(
        "nebius/SWE-agent-trajectories", split="train", streaming=True,
    )
    traces: list[Trace] = []
    for example in ds:
        if len(traces) >= max_traces:
            break
        actions_raw: list[str] = []
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
                        actions_raw.append(cmd)
        if 3 <= len(actions_raw) <= 200:
            solved = bool(example.get("target"))
            trace_actions = [
                Action(tool=t, args="{}", context="swe", label=0 if solved else 1)
                for t in actions_raw
            ]
            traces.append(Trace(
                trace_id=f"swe_{len(traces):06d}",
                actions=trace_actions,
                source="swe_agent",
            ))
    return traces


# ---------------------------------------------------------------------------
# AgentDojo traces (prompt injection benchmark)
# ---------------------------------------------------------------------------


def load_agentdojo(
    data_dir: Path,
    *,
    max_traces: int | None = None,
    min_tool_calls: int = 3,
) -> list[Trace]:
    """Load AgentDojo benchmark as prompt-injection-labeled agent traces.

    AgentDojo (Debenedetti et al., 2024) evaluates prompt injection
    attacks across 4 domains (banking, slack, travel, workspace). Each
    trace has a ``security`` field: True = agent resisted the injection
    (or no injection present), False = agent was compromised.

    Expects ``data_dir / agentdojo/`` containing suite subdirectories
    with nested JSON files from the AgentDojo runs output.

    Args:
        data_dir: Path containing agentdojo/ subdirectory (or direct path
            to the runs directory with suite folders).
        max_traces: Maximum traces to load.
        min_tool_calls: Minimum tool calls for a trace to be included.

    Returns:
        List of Trace objects (label 0=resisted/benign, 1=compromised).
    """
    # Support both data_dir/agentdojo/ and direct path to suite folders
    agentdojo_root = data_dir / "agentdojo"
    if not agentdojo_root.exists():
        agentdojo_root = data_dir
    suites = ["banking", "slack", "travel", "workspace"]
    found_suites = [s for s in suites if (agentdojo_root / s).exists()]
    if not found_suites:
        raise FileNotFoundError(
            f"AgentDojo data not found at {agentdojo_root}. "
            "Expected subdirectories: banking/, slack/, travel/, workspace/"
        )

    traces: list[Trace] = []
    for suite in found_suites:
        suite_dir = agentdojo_root / suite
        for json_path in sorted(suite_dir.rglob("*.json")):
            if max_traces is not None and len(traces) >= max_traces:
                break
            try:
                raw = json.loads(json_path.read_text())
            except (json.JSONDecodeError, PermissionError):
                continue

            messages = raw.get("messages", [])
            security = raw.get("security")
            if security is None:
                continue

            # Extract tool calls from assistant messages
            actions: list[Action] = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func_name = tc.get("function", "")
                        if isinstance(func_name, dict):
                            func_name = func_name.get("name", "unknown")
                        args_str = tc.get("args", {})
                        if isinstance(args_str, dict):
                            args_str = json.dumps(args_str)
                        actions.append(Action(
                            tool=str(func_name),
                            args=str(args_str)[:2000],
                            context=suite,
                            label=0 if security else 1,
                        ))
                elif msg.get("role") == "tool":
                    # Attach outcome to the last action if available
                    content = str(msg.get("content", ""))[:2000]
                    if actions:
                        actions[-1] = Action(
                            tool=actions[-1].tool,
                            args=actions[-1].args,
                            context=actions[-1].context,
                            outcome=content,
                            label=actions[-1].label,
                        )

            if len(actions) < min_tool_calls:
                continue

            trace_id = (
                f"agentdojo_{suite}_{raw.get('user_task_id', 'unk')}"
                f"_{raw.get('injection_task_id', 'none')}"
                f"_{raw.get('attack_type', 'none')}"
            )
            is_compromised = not security
            traces.append(Trace(
                trace_id=trace_id,
                actions=actions,
                source="agentdojo",
                task=f"{suite} {raw.get('user_task_id', '')}",
                has_anomaly=is_compromised,
            ))

    return traces


# ---------------------------------------------------------------------------
# ToolACE traces (HuggingFace: Team-ACE/ToolACE)
# ---------------------------------------------------------------------------

import re as _re


def _parse_toolace_calls(assistant_value: str) -> list[tuple[str, str]]:
    """Parse ToolACE assistant turn into (tool_name, args_json) pairs.

    ToolACE format: [FuncName(k="v", k2=123)] or [f1(a="x"), f2(b="y")]
    """
    results: list[tuple[str, str]] = []
    pattern = _re.compile(r"([\w][\w\s]*?)\(([^)]*)\)")
    for match in pattern.finditer(assistant_value):
        func_name = match.group(1).strip()
        args_raw = match.group(2).strip()
        results.append((func_name, args_raw if args_raw else "{}"))
    return results


def load_toolace(
    *,
    max_traces: int | None = None,
    min_tool_calls: int = 3,
    cache_dir: str | None = None,
) -> list[Trace]:
    """Load ToolACE dataset from HuggingFace as normalized traces.

    ToolACE (Team-ACE/ToolACE) contains 11.3k multi-turn tool-calling
    conversations across 26,507 APIs. Each conversation has user queries,
    assistant tool calls (string format), and tool responses (JSON).

    All traces are normal (correct tool usage). Anomaly injection is
    handled separately in the domain loader.

    Args:
        max_traces: Maximum traces to load.
        min_tool_calls: Minimum tool calls for a trace to be included.
        cache_dir: HuggingFace cache directory.

    Returns:
        List of Trace objects (all label=0, normal tool usage).
    """
    from datasets import load_dataset

    print("  Downloading ToolACE from HuggingFace...")
    ds = load_dataset(
        "Team-ACE/ToolACE",
        split="train",
        cache_dir=cache_dir,
    )

    traces: list[Trace] = []
    for i, row in enumerate(ds):
        if max_traces is not None and len(traces) >= max_traces:
            break

        conversations = row.get("conversations", [])
        if not conversations:
            continue

        actions: list[Action] = []
        pending_calls: list[Action] = []

        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "assistant" and value.strip().startswith("["):
                calls = _parse_toolace_calls(value)
                pending_calls = []
                for func_name, args_str in calls:
                    action = Action(
                        tool=func_name,
                        args=args_str[:2000],
                        context="toolace",
                        timestamp=float(len(actions) + len(pending_calls)),
                        label=0,
                    )
                    pending_calls.append(action)

            elif role == "tool" and pending_calls:
                try:
                    tool_results = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    tool_results = [{"results": value}]

                if not isinstance(tool_results, list):
                    tool_results = [tool_results]

                for j, pc in enumerate(pending_calls):
                    outcome = ""
                    if j < len(tool_results):
                        res = tool_results[j]
                        if isinstance(res, dict):
                            outcome = json.dumps(
                                res.get("results", res)
                            )[:2000]
                        else:
                            outcome = str(res)[:2000]
                    actions.append(Action(
                        tool=pc.tool,
                        args=pc.args,
                        context=pc.context,
                        timestamp=float(len(actions)),
                        label=0,
                        outcome=outcome,
                    ))
                pending_calls = []

        # Flush any pending calls without tool responses
        for pc in pending_calls:
            actions.append(Action(
                tool=pc.tool,
                args=pc.args,
                context=pc.context,
                timestamp=float(len(actions)),
                label=0,
            ))

        if len(actions) < min_tool_calls:
            continue

        traces.append(Trace(
            trace_id=f"toolace_{i:06d}",
            actions=actions,
            source="toolace",
            task=conversations[0].get("value", "")[:200] if conversations else "",
        ))

    print(f"  ToolACE: {len(traces)} traces loaded (all normal)")
    return traces


# ---------------------------------------------------------------------------
# API-Bank traces (HuggingFace: liminghao1630/API-Bank)
# ---------------------------------------------------------------------------


_APIBANK_CALL_RE = _re.compile(
    r"\[([A-Za-z_]\w*)\(([^)]*)\)\](?:->(.+?))?(?=\nUser:|\nAI:|\nAPI-Request:|\Z)",
    _re.DOTALL,
)


def _parse_apibank_call(call_str: str) -> tuple[str, str]:
    """Parse 'func_name(k1=v1, k2=v2)' into (func_name, args_string)."""
    paren = call_str.find("(")
    if paren == -1:
        return call_str.strip(), "{}"
    func = call_str[:paren].strip()
    args = call_str[paren + 1 :].rstrip(")")
    return func, args if args else "{}"


def load_apibank(
    *,
    max_traces: int | None = None,
    min_tool_calls: int = 3,
    cache_dir: str | None = None,
) -> list[Trace]:
    """Load API-Bank dataset from HuggingFace as normalized traces.

    API-Bank (Li et al., EMNLP 2023) contains 33k training rows of
    multi-turn dialogues with API calls across 73 tools and 1,000 domains.
    Each row contains the conversation history (with previous API calls
    and responses) plus the next expected API call.

    We reconstruct full conversations by detecting conversation boundaries
    (when the number of previous calls drops, a new conversation starts)
    and taking the final step of each conversation as the complete trace.

    All traces are normal (correct API usage). Anomaly injection is
    handled separately in the domain loader.

    Args:
        max_traces: Maximum traces to load.
        min_tool_calls: Minimum tool calls for a trace to be included.
        cache_dir: HuggingFace cache directory.

    Returns:
        List of Trace objects (all label=0, correct API usage).
    """
    from datasets import load_dataset

    print("  Downloading API-Bank from HuggingFace...")
    ds = load_dataset(
        "liminghao1630/API-Bank",
        data_files="training-data/*.json",
        split="train",
        cache_dir=cache_dir,
    )

    traces: list[Trace] = []
    prev_call_count = 0
    current_actions: list[Action] = []
    current_task = ""
    trace_idx = 0

    def _flush_trace() -> None:
        nonlocal trace_idx
        if len(current_actions) >= min_tool_calls:
            traces.append(Trace(
                trace_id=f"apibank_{trace_idx:06d}",
                actions=list(current_actions),
                source="apibank",
                task=current_task[:200],
            ))
            trace_idx += 1

    for row in ds:
        if max_traces is not None and len(traces) >= max_traces:
            break

        inp = row.get("input", "")
        output = row.get("output", "")
        n_prev = inp.count("API-Request:")

        if n_prev < prev_call_count and current_actions:
            _flush_trace()
            current_actions = []
            current_task = ""

        # Rebuild actions from input history + output
        current_actions = []
        # Extract user task from first User: line
        user_match = _re.search(r"User:\s*(.+?)(?:\n|$)", inp)
        if user_match and not current_task:
            current_task = user_match.group(1).strip()

        # Extract all API calls from input
        for match in _re.finditer(
            r"API-Request: \[([^\]]+)\](?:->(.+?))?(?=\n|User:|AI:|API-Request:|$)",
            inp,
            _re.DOTALL,
        ):
            call_str = match.group(1)
            resp_str = (match.group(2) or "").strip()
            func, args = _parse_apibank_call(call_str)
            current_actions.append(Action(
                tool=func,
                args=args[:2000],
                context="apibank",
                timestamp=float(len(current_actions)),
                label=0,
                outcome=resp_str[:2000],
            ))

        # Extract the output call (no response available)
        out_match = _re.search(
            r"API-Request: \[([^\]]+)\]", output,
        )
        if out_match:
            func, args = _parse_apibank_call(out_match.group(1))
            current_actions.append(Action(
                tool=func,
                args=args[:2000],
                context="apibank",
                timestamp=float(len(current_actions)),
                label=0,
            ))

        prev_call_count = n_prev

    # Flush last conversation
    if current_actions:
        _flush_trace()

    print(f"  API-Bank: {len(traces)} traces loaded (all normal)")
    return traces


# ---------------------------------------------------------------------------
# Custom MCP trace format (JSONL)
# ---------------------------------------------------------------------------


def load_mcp_traces(data_dir: Path) -> list[Trace]:
    """Load custom MCP-format traces from JSONL files.

    Expected format: one JSON object per line with fields:
        trace_id, actions: [{tool, args, context, timestamp, label}], ...
    """
    traces: list[Trace] = []
    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        for line in jsonl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            obj = orjson.loads(line)
            trace = Trace.model_validate(obj)
            traces.append(trace)
    return traces


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_traces(
    traces: list[Trace],
    *,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    """Split traces into train / calibration / test sets.

    The calibration set is essential for conformal prediction.
    """
    rng = random.Random(seed)
    indices = list(range(len(traces)))
    rng.shuffle(indices)

    n_train = int(len(traces) * train_frac)
    n_cal = int(len(traces) * cal_frac)

    train = [traces[i] for i in indices[:n_train]]
    cal = [traces[i] for i in indices[n_train : n_train + n_cal]]
    test = [traces[i] for i in indices[n_train + n_cal :]]
    return train, cal, test


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_traces(traces: list[Trace], path: Path) -> None:
    """Save traces to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for trace in traces:
            f.write(orjson.dumps(trace.model_dump()) + b"\n")


def load_traces(path: Path) -> list[Trace]:
    """Load traces from a JSONL file."""
    traces: list[Trace] = []
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        traces.append(Trace.model_validate(orjson.loads(line)))
    return traces


# ---------------------------------------------------------------------------
# StableToolBench parser (organic API failure labels from ToolEval)
# ---------------------------------------------------------------------------


def load_stabletoolbench(
    data_dir: Path | None = None,
    model: str = "chatgpt_cot",
    max_traces: int | None = None,
) -> list[Trace]:
    """Load StableToolBench traces with organic pass/fail labels.

    Each JSON file contains a full tool-call chain and a ``win`` field
    indicating whether the agent successfully completed the task (judged
    by GPT-4 via ToolEval's Solvable Pass Rate metric). Failed traces
    (win=False) are labeled anomalous — these are *organic* failures,
    not injected anomalies.

    The chain alternates Action (tool name) → Action Input (args + observation)
    → Thought nodes. We extract Action/Action Input pairs as canonical Actions.
    """
    if data_dir is None:
        data_dir = Path("/tmp/stabletoolbench_baselines/data_baselines") / model

    if not data_dir.exists():
        raise FileNotFoundError(
            f"StableToolBench data not found at {data_dir}. "
            "Download from https://github.com/THUNLP-MT/StableToolBench"
        )

    traces: list[Trace] = []
    group_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir()
    )

    for group_dir in group_dirs:
        for json_path in sorted(group_dir.glob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            win = data.get("win", False)
            trys = data.get("trys", [])
            if not trys:
                continue

            chain = trys[0].get("chain", [])
            if not chain:
                continue

            # Trace-level label: win=False → anomalous (all actions labeled 1)
            action_label = 0 if win else 1

            actions: list[Action] = []
            i = 0
            while i < len(chain):
                node = chain[i]
                if node.get("node_type") == "Action":
                    tool_name = node.get("description", "unknown")
                    args_str = ""
                    observation = ""
                    if i + 1 < len(chain) and chain[i + 1].get("node_type") == "Action Input":
                        inp_node = chain[i + 1]
                        args_str = inp_node.get("description", "")
                        observation = inp_node.get("observation", "")
                        i += 2
                    else:
                        i += 1
                    actions.append(Action(
                        tool=tool_name,
                        args=args_str,
                        context="",
                        timestamp=float(len(actions)),
                        label=action_label,
                        outcome=observation,
                        outcome_label=action_label,
                    ))
                else:
                    i += 1

            if len(actions) < 2:
                continue

            query = data.get("forward_args", {}).get("query", "")
            trace_id = f"stabletoolbench_{json_path.stem}"

            traces.append(Trace(
                trace_id=trace_id,
                actions=actions,
                source="stabletoolbench",
                agent=model,
                task=query,
            ))

            if max_traces and len(traces) >= max_traces:
                break
        if max_traces and len(traces) >= max_traces:
            break

    n_anom = sum(1 for t in traces if t.has_anomaly)
    print(f"Loaded {len(traces)} StableToolBench traces "
          f"({n_anom} anomalous / {len(traces) - n_anom} normal)")
    return traces


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class TraceVocab:
    """Maps tool names and tokens to integer IDs for the encoder."""

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    _SPECIAL_COUNT = 4

    def __init__(self) -> None:
        self.tool2id: dict[str, int] = {}
        self._next_id = self._SPECIAL_COUNT

    def fit(self, traces: list[Trace]) -> "TraceVocab":
        """Build vocabulary from a list of traces."""
        tools: set[str] = set()
        for trace in traces:
            for action in trace.actions:
                tools.add(action.tool)
        for tool in sorted(tools):
            self.tool2id[tool] = self._next_id
            self._next_id += 1
        return self

    def encode_tool(self, tool: str) -> int:
        return self.tool2id.get(tool, self.UNK)

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def save(self, path: Path) -> None:
        path.write_bytes(orjson.dumps(self.tool2id))

    @classmethod
    def load(cls, path: Path) -> "TraceVocab":
        vocab = cls()
        vocab.tool2id = orjson.loads(path.read_bytes())
        vocab._next_id = max(vocab.tool2id.values()) + 1 if vocab.tool2id else cls._SPECIAL_COUNT
        return vocab


def _hash_args(args: str) -> int:
    """Deterministic hash of argument string to a 32-bit int."""
    return int(hashlib.md5(args.encode(), usedforsecurity=False).hexdigest()[:8], 16)


class TraceDataset(Dataset):
    """PyTorch Dataset wrapping a list of Traces.

    Each item returns a dict with:
        tool_ids: LongTensor of shape (seq_len,)
        args_hashes: LongTensor of shape (seq_len,)
        labels: LongTensor of shape (seq_len,) — per-action labels
        length: int
    """

    def __init__(self, traces: list[Trace], vocab: TraceVocab, max_len: int = 200) -> None:
        self.traces = traces
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> dict:
        trace = self.traces[idx]
        actions = trace.actions[: self.max_len]
        n = len(actions)

        tool_ids = torch.zeros(n, dtype=torch.long)
        args_hashes = torch.zeros(n, dtype=torch.long)
        outcome_hashes = torch.zeros(n, dtype=torch.long)
        labels = torch.zeros(n, dtype=torch.long)

        for i, action in enumerate(actions):
            tool_ids[i] = self.vocab.encode_tool(action.tool)
            args_hashes[i] = _hash_args(action.args) % (2**16)
            outcome_hashes[i] = _hash_args(action.outcome) % (2**16)
            labels[i] = action.label

        return {
            "tool_ids": tool_ids,
            "args_hashes": args_hashes,
            "outcome_hashes": outcome_hashes,
            "labels": labels,
            "length": n,
        }


def collate_traces(batch: list[dict]) -> dict:
    """Collate variable-length traces into padded tensors."""
    max_len = max(item["length"] for item in batch)
    bsz = len(batch)

    tool_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    args_hashes = torch.zeros(bsz, max_len, dtype=torch.long)
    outcome_hashes = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.zeros(bsz, max_len, dtype=torch.long)
    lengths = torch.zeros(bsz, dtype=torch.long)
    mask = torch.zeros(bsz, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["length"]
        tool_ids[i, :n] = item["tool_ids"]
        args_hashes[i, :n] = item["args_hashes"]
        outcome_hashes[i, :n] = item["outcome_hashes"]
        labels[i, :n] = item["labels"]
        lengths[i] = n
        mask[i, :n] = True

    return {
        "tool_ids": tool_ids,
        "args_hashes": args_hashes,
        "outcome_hashes": outcome_hashes,
        "labels": labels,
        "lengths": lengths,
        "mask": mask,
    }
