"""Label-direction and provenance tests for the v50 rerun.

The AgentDojo fixtures copy the upstream field semantics checked against
`runs/gpt-4o-2024-05-13/banking/user_task_0/` at commit 089ed468: `security: true` on an attacked
run means the injection succeeded, and the no-attack `none` run also records `security: true`.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from data.trace_loader import load_agentdojo


def _run(
    attack: str | None, security: bool, n_calls: int, inj: str = "injection_task_0"
) -> dict[str, object]:
    calls = [{"function": f"tool_{i}", "args": {"x": i}} for i in range(n_calls)]
    return {
        "suite_name": "banking",
        "pipeline_name": "gpt-4o-2024-05-13",
        "user_task_id": "user_task_0",
        "injection_task_id": None if attack is None else inj,
        "attack_type": attack,
        "security": security,
        "messages": [{"role": "assistant", "tool_calls": calls}],
    }


@pytest.fixture
def agentdojo_dir(tmp_path: Path) -> Path:
    base = tmp_path / "banking" / "user_task_0"
    files = {
        "none/none.json": _run(None, True, 3),
        "important_instructions/injection_task_0.json": _run("important_instructions", True, 5),
        "important_instructions/injection_task_1.json": _run(
            "important_instructions", False, 4, "injection_task_1"
        ),
        "direct/injection_task_0.json": _run("direct", True, 5),
    }
    for rel, payload in files.items():
        path = base / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return tmp_path


def test_agentdojo_successful_attack_is_anomalous(agentdojo_dir: Path) -> None:
    traces = load_agentdojo(agentdojo_dir, min_tool_calls=1)
    by_task = {t.task: int(t.has_anomaly) for t in traces}
    assert by_task == {"user_task_0|injection_task_0": 1, "user_task_0|injection_task_1": 0}


def test_agentdojo_excludes_none_and_other_attacks(agentdojo_dir: Path) -> None:
    traces = load_agentdojo(agentdojo_dir, min_tool_calls=1)
    assert len(traces) == 2
    assert all(t.agent == "gpt-4o-2024-05-13" for t in traces)


def test_agentdojo_attack_filter_is_a_parameter(agentdojo_dir: Path) -> None:
    traces = load_agentdojo(agentdojo_dir, min_tool_calls=1, attack="direct")
    assert [int(t.has_anomaly) for t in traces] == [1]


def test_cohens_d_sign_is_anomalous_minus_normal() -> None:
    from traceshift.v50 import cohens_d_anomalous

    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert cohens_d_anomalous(x, y) > 0
    assert cohens_d_anomalous(-x, y) < 0


def test_reversal_requires_both_effects_above_threshold() -> None:
    from traceshift.v50 import is_reversed

    assert is_reversed(0.3, -0.3, tau=0.1)
    assert not is_reversed(0.3, 0.3, tau=0.1)
    assert not is_reversed(0.3, -0.05, tau=0.1)
    assert not is_reversed(0.15, -0.15, tau=0.2)


def test_split_halves_are_disjoint_and_stratified() -> None:
    from traceshift.v50 import stratified_halves

    y = np.array([0] * 80 + [1] * 20)
    a, b = stratified_halves(y, seed=3)
    assert set(a).isdisjoint(b)
    assert len(a) + len(b) == len(y)
    assert y[a].sum() == 10 and y[b].sum() == 10
