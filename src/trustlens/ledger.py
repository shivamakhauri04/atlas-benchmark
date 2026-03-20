"""Append-only audit ledger.

Records every gate decision (permit / escalate / deny) with a
hash-chained certificate for post-hoc auditability.

Each entry: {decision_data, timestamp, prev_hash, entry_hash}
entry_hash = SHA-256(decision_data || timestamp || prev_hash)

Tampering with any entry breaks the chain, detectable by verify_chain().
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LedgerEntry:
    """A single entry in the audit ledger."""

    index: int
    decision: dict
    timestamp: float
    prev_hash: str
    entry_hash: str


def _compute_hash(decision: dict, timestamp: float, prev_hash: str) -> str:
    """Compute SHA-256 hash for a ledger entry."""
    payload = json.dumps(decision, sort_keys=True) + str(timestamp) + prev_hash
    return hashlib.sha256(payload.encode()).hexdigest()


class AuditLedger:
    """Hash-chained append-only log of gate decisions.

    Provides tamper-evident recording of every permission decision
    (permit / escalate / deny) with full provenance.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> LedgerEntry:
        return self._entries[idx]

    @property
    def last_hash(self) -> str:
        if not self._entries:
            return self.GENESIS_HASH
        return self._entries[-1].entry_hash

    def append(self, decision: dict) -> str:
        """Append a decision record and return its certificate hash.

        Args:
            decision: Dict containing at minimum:
                - action: str (the tool call or action taken)
                - verdict: str (permit / escalate / deny)
                - score: float (anomaly score)
                - threshold: float (current CP threshold)
                Optional:
                - trace_id: str
                - reason: str

        Returns:
            The SHA-256 hash of the new entry (the "certificate").
        """
        ts = time.time()
        prev = self.last_hash
        entry_hash = _compute_hash(decision, ts, prev)
        entry = LedgerEntry(
            index=len(self._entries),
            decision=decision,
            timestamp=ts,
            prev_hash=prev,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry_hash

    def verify_chain(self) -> bool:
        """Verify the integrity of the full ledger chain.

        Returns True if no tampering detected, False otherwise.
        """
        prev_hash = self.GENESIS_HASH
        for entry in self._entries:
            if entry.prev_hash != prev_hash:
                return False
            expected = _compute_hash(entry.decision, entry.timestamp, entry.prev_hash)
            if entry.entry_hash != expected:
                return False
            prev_hash = entry.entry_hash
        return True

    def save(self, path: Path) -> None:
        """Save ledger to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for entry in self._entries:
                row = {
                    "index": entry.index,
                    "decision": entry.decision,
                    "timestamp": entry.timestamp,
                    "prev_hash": entry.prev_hash,
                    "entry_hash": entry.entry_hash,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "AuditLedger":
        """Load ledger from a JSONL file and verify integrity."""
        ledger = cls()
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            entry = LedgerEntry(
                index=row["index"],
                decision=row["decision"],
                timestamp=row["timestamp"],
                prev_hash=row["prev_hash"],
                entry_hash=row["entry_hash"],
            )
            ledger._entries.append(entry)
        if not ledger.verify_chain():
            raise ValueError(f"Ledger integrity check failed for {path}")
        return ledger

    def query(
        self,
        *,
        verdict: str | None = None,
        trace_id: str | None = None,
        since: float | None = None,
    ) -> list[LedgerEntry]:
        """Query ledger entries by verdict, trace_id, or timestamp."""
        results = self._entries
        if verdict is not None:
            results = [e for e in results if e.decision.get("verdict") == verdict]
        if trace_id is not None:
            results = [e for e in results if e.decision.get("trace_id") == trace_id]
        if since is not None:
            results = [e for e in results if e.timestamp >= since]
        return results
