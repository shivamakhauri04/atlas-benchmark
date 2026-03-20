"""Normalized trace schema for TrustLens.

Every dataset (ToolBench, ToolEmu, Agent-SafetyBench, etc.) is converted
to this canonical format before being consumed by the encoder or baselines.
"""

from __future__ import annotations

from pydantic import BaseModel


class Action(BaseModel):
    """A single agent tool call."""

    tool: str
    args: str
    context: str = ""
    timestamp: float = 0.0
    label: int = 0  # 0 = normal/safe, 1 = anomalous/unsafe
    outcome: str = ""  # tool execution result/consequence
    outcome_label: int = 0  # 0 = expected outcome, 1 = anomalous outcome

    model_config = {"frozen": True}


class Trace(BaseModel):
    """An ordered sequence of agent actions with metadata."""

    trace_id: str
    actions: list[Action]
    source: str = ""  # dataset name (toolbench, toolemu, synthetic, etc.)
    agent: str = ""  # agent model that generated the trace
    task: str = ""  # original task description

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def has_anomaly(self) -> bool:
        return any(a.label == 1 for a in self.actions)

    @property
    def anomaly_rate(self) -> float:
        if not self.actions:
            return 0.0
        return sum(a.label == 1 for a in self.actions) / len(self.actions)
