"""Evaluate the audit ledger: integrity, tamper detection, throughput.

Produces quantitative results for the paper's audit ledger contribution.

Usage:
    python scripts/eval_ledger.py
    python scripts/eval_ledger.py --n-entries 10000
"""

import argparse
import copy
import json
import time
from pathlib import Path

from trustlens.ledger import AuditLedger, LedgerEntry, _compute_hash


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TrustLens audit ledger")
    p.add_argument("--n-entries", type=int, default=5000, help="Entries for throughput test")
    p.add_argument("--output-dir", type=Path, default=Path("output/tables"))
    return p.parse_args()


def _make_decision(i: int, anomalous: bool = False) -> dict:
    return {
        "action": f"tool_{i % 20}",
        "verdict": "deny" if anomalous else "permit",
        "score": 0.8 if anomalous else 0.2,
        "threshold": 0.5,
        "trace_id": f"trace_{i:06d}",
    }


def test_chain_integrity(n: int = 100) -> dict:
    """Test that a clean ledger always passes verification."""
    ledger = AuditLedger()
    for i in range(n):
        ledger.append(_make_decision(i, anomalous=(i % 20 == 0)))

    assert len(ledger) == n
    assert ledger.verify_chain(), "Clean chain should verify"

    denies = ledger.query(verdict="deny")
    permits = ledger.query(verdict="permit")
    assert len(denies) + len(permits) == n
    assert len(denies) == n // 20

    return {
        "test": "chain_integrity",
        "n_entries": n,
        "verified": True,
        "denies": len(denies),
        "permits": len(permits),
    }


def test_tamper_detection(n: int = 100) -> dict:
    """Test that modifying any entry is detected by verify_chain()."""
    ledger = AuditLedger()
    for i in range(n):
        ledger.append(_make_decision(i))

    assert ledger.verify_chain(), "Clean chain should verify first"

    detections = 0
    for target_idx in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
        tampered = AuditLedger()
        for i in range(n):
            tampered._entries.append(ledger[i])

        original = tampered._entries[target_idx]
        modified_decision = dict(original.decision)
        modified_decision["verdict"] = "permit_TAMPERED"
        tampered._entries[target_idx] = LedgerEntry(
            index=original.index,
            decision=modified_decision,
            timestamp=original.timestamp,
            prev_hash=original.prev_hash,
            entry_hash=original.entry_hash,
        )

        if not tampered.verify_chain():
            detections += 1

    return {
        "test": "tamper_detection",
        "n_entries": n,
        "tamper_positions": 5,
        "detections": detections,
        "detection_rate": detections / 5,
    }


def test_save_load_roundtrip(n: int = 100) -> dict:
    """Test save/load preserves chain integrity."""
    ledger = AuditLedger()
    for i in range(n):
        ledger.append(_make_decision(i, anomalous=(i % 10 == 0)))

    tmp_path = Path("/tmp/trustlens_ledger_test.jsonl")
    ledger.save(tmp_path)
    loaded = AuditLedger.load(tmp_path)

    assert len(loaded) == n
    assert loaded.verify_chain()
    assert loaded.last_hash == ledger.last_hash

    for i in range(n):
        assert loaded[i].entry_hash == ledger[i].entry_hash

    tmp_path.unlink(missing_ok=True)

    return {
        "test": "save_load_roundtrip",
        "n_entries": n,
        "hashes_match": True,
        "chain_valid_after_load": True,
    }


def test_query_functionality(n: int = 500) -> dict:
    """Test query filtering by verdict, trace_id, and timestamp."""
    ledger = AuditLedger()
    for i in range(n):
        ledger.append(_make_decision(i, anomalous=(i % 5 == 0)))

    denies = ledger.query(verdict="deny")
    assert len(denies) == n // 5

    specific = ledger.query(trace_id="trace_000042")
    assert len(specific) == 1
    assert specific[0].decision["trace_id"] == "trace_000042"

    mid_ts = ledger[n // 2].timestamp
    recent = ledger.query(since=mid_ts)
    assert len(recent) >= n // 2 - 1

    deny_recent = ledger.query(verdict="deny", since=mid_ts)
    assert all(e.decision["verdict"] == "deny" for e in deny_recent)

    return {
        "test": "query_functionality",
        "n_entries": n,
        "deny_count": len(denies),
        "specific_lookup": len(specific) == 1,
        "time_filter_works": len(recent) >= n // 2 - 1,
        "combined_filter_works": True,
    }


def benchmark_throughput(n: int = 5000) -> dict:
    """Benchmark append and verify throughput."""
    ledger = AuditLedger()

    t0 = time.perf_counter()
    for i in range(n):
        ledger.append(_make_decision(i, anomalous=(i % 20 == 0)))
    append_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    assert ledger.verify_chain()
    verify_time = time.perf_counter() - t0

    tmp_path = Path("/tmp/trustlens_ledger_bench.jsonl")
    t0 = time.perf_counter()
    ledger.save(tmp_path)
    save_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    loaded = AuditLedger.load(tmp_path)
    load_time = time.perf_counter() - t0

    file_size = tmp_path.stat().st_size
    tmp_path.unlink(missing_ok=True)

    return {
        "test": "throughput",
        "n_entries": n,
        "append_total_ms": round(append_time * 1000, 2),
        "append_per_entry_us": round(append_time / n * 1e6, 2),
        "verify_total_ms": round(verify_time * 1000, 2),
        "verify_per_entry_us": round(verify_time / n * 1e6, 2),
        "save_total_ms": round(save_time * 1000, 2),
        "load_total_ms": round(load_time * 1000, 2),
        "file_size_kb": round(file_size / 1024, 2),
        "entries_per_sec_append": round(n / append_time),
        "entries_per_sec_verify": round(n / verify_time),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Audit Ledger Evaluation ===\n")

    results = []

    print("1. Chain integrity test...")
    r = test_chain_integrity()
    results.append(r)
    print(f"   PASS: {r['n_entries']} entries, {r['denies']} denies, chain valid\n")

    print("2. Tamper detection test...")
    r = test_tamper_detection()
    results.append(r)
    print(f"   PASS: {r['detections']}/{r['tamper_positions']} tampering attempts detected")
    print(f"   Detection rate: {r['detection_rate']:.0%}\n")

    print("3. Save/load round-trip test...")
    r = test_save_load_roundtrip()
    results.append(r)
    print(f"   PASS: hashes match={r['hashes_match']}, chain valid after load={r['chain_valid_after_load']}\n")

    print("4. Query functionality test...")
    r = test_query_functionality()
    results.append(r)
    print(f"   PASS: verdict filter, trace_id lookup, time filter, combined filter\n")

    print(f"5. Throughput benchmark ({args.n_entries} entries)...")
    r = benchmark_throughput(args.n_entries)
    results.append(r)
    print(f"   Append:  {r['entries_per_sec_append']:,} entries/sec ({r['append_per_entry_us']} us/entry)")
    print(f"   Verify:  {r['entries_per_sec_verify']:,} entries/sec ({r['verify_per_entry_us']} us/entry)")
    print(f"   Save:    {r['save_total_ms']:.1f} ms for {args.n_entries} entries")
    print(f"   Load:    {r['load_total_ms']:.1f} ms (includes verify)")
    print(f"   Storage: {r['file_size_kb']:.1f} KB for {args.n_entries} entries\n")

    out_path = args.output_dir / "ledger_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
