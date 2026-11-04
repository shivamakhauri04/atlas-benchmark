"""Load AgentDrift (arXiv 2609.06972, CC BY 4.0) as binary attack-vs-benign corpora.

Each corpus pairs one attack category in one domain with that domain's benign trajectories, matching
the external stratum registered in prereg-v50 A5. Pinned to repository commit 014a514f.
"""

import json
from pathlib import Path

from data.schema import Action, Trace

CATEGORIES = {
    "attacked_full": "full_hijack",
    "attacked_partial": "partial_hijack",
    "attacked_delayed": "delayed_execution",
    "failed_attack": "failed_attack",
    "hard_negative": "hard_negative",
}
DOMAINS = ("email", "banking", "web", "coding", "medical")


def _records(root: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        with (root / "data" / f"{split}.jsonl").open() as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows


def _trace(rec: dict, label: int) -> Trace:
    actions = [
        Action(tool=str(s["tool"]), args=json.dumps(s.get("args") or {}), label=label)
        for s in rec["steps"]
    ]
    return Trace(
        trace_id=rec["id"],
        actions=actions,
        source="agentdrift",
        agent=rec["agent"],
        task=rec["task"],
    )


def load_agentdrift(root: Path, *, category: str, domain: str) -> list[Trace]:
    """Return `category` trajectories (label 1) and benign trajectories (label 0) for `domain`.

    Raises:
        ValueError: If `category` or `domain` is not an AgentDrift value.
    """
    source = {v: k for k, v in CATEGORIES.items()}.get(category)
    if source is None or domain not in DOMAINS:
        raise ValueError(f"unknown AgentDrift corpus {category}/{domain}")
    out: list[Trace] = []
    for rec in _records(root):
        if rec["agent"] != domain:
            continue
        if rec["source_category"] == source:
            out.append(_trace(rec, 1))
        elif rec["source_category"] == "benign":
            out.append(_trace(rec, 0))
    return out
