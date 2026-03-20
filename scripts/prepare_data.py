"""Download and prepare trace datasets.

Downloads ToolBench/AgentBench, normalizes to the trace schema,
saves as .jsonl, and generates fixed train/cal/test splits.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-dir /kaggle/input/trustlens-traces
    python scripts/prepare_data.py --synthetic-only --n-traces 5000
"""

import argparse
from pathlib import Path

from data.schema import Trace
from data.trace_loader import (
    TraceVocab,
    generate_synthetic_traces,
    load_toolbench,
    load_traces,
    save_traces,
    split_traces,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare TrustLens trace datasets")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-only", action="store_true", help="Skip real datasets, use synthetic only")
    p.add_argument("--n-traces", type=int, default=10000, help="Number of synthetic traces")
    p.add_argument("--anomaly-rate", type=float, default=0.05, help="Fraction of anomalous traces")
    p.add_argument("--max-toolbench", type=int, default=None, help="Max ToolBench traces to load")
    return p.parse_args()


def _print_stats(name: str, traces: list[Trace]) -> None:
    """Print dataset statistics."""
    n = len(traces)
    if n == 0:
        print(f"  {name}: 0 traces")
        return
    lengths = [t.length for t in traces]
    n_anomalous = sum(1 for t in traces if t.has_anomaly)
    total_actions = sum(lengths)
    anomalous_actions = sum(
        sum(1 for a in t.actions if a.label == 1) for t in traces
    )
    tools = set()
    for t in traces:
        for a in t.actions:
            tools.add(a.tool)

    print(f"  {name}: {n} traces, {total_actions} actions")
    print(f"    Trace lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/n:.1f}")
    print(f"    Anomalous traces: {n_anomalous}/{n} ({100*n_anomalous/n:.1f}%)")
    print(f"    Anomalous actions: {anomalous_actions}/{total_actions} ({100*anomalous_actions/max(total_actions,1):.1f}%)")
    print(f"    Unique tools: {len(tools)}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    all_traces: list[Trace] = []

    # --- Synthetic traces (always generated for dev/testing) ---
    print("Generating synthetic traces...")
    synthetic = generate_synthetic_traces(
        n_traces=args.n_traces,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    _print_stats("synthetic", synthetic)
    save_traces(synthetic, data_dir / "synthetic.jsonl")
    all_traces.extend(synthetic)

    # --- ToolBench (if available and not synthetic-only) ---
    if not args.synthetic_only:
        toolbench_file = data_dir / "toolbench_train.json"
        if toolbench_file.exists():
            print("Loading ToolBench traces...")
            toolbench = load_toolbench(data_dir, max_traces=args.max_toolbench)
            _print_stats("toolbench", toolbench)
            save_traces(toolbench, data_dir / "toolbench.jsonl")
            all_traces.extend(toolbench)
        else:
            print(f"ToolBench not found at {toolbench_file} — skipping.")
            print("  To include ToolBench, download the training file and place it there.")

    # --- Split ---
    print(f"\nSplitting {len(all_traces)} total traces (60/20/20)...")
    train, cal, test = split_traces(all_traces, seed=args.seed)

    save_traces(train, data_dir / "train.jsonl")
    save_traces(cal, data_dir / "cal.jsonl")
    save_traces(test, data_dir / "test.jsonl")

    print("\nSplit statistics:")
    _print_stats("train", train)
    _print_stats("cal", cal)
    _print_stats("test", test)

    # --- Build and save vocabulary ---
    print("\nBuilding vocabulary from training set...")
    vocab = TraceVocab().fit(train)
    vocab.save(data_dir / "vocab.json")
    print(f"  Vocabulary size: {vocab.vocab_size} (including 4 special tokens)")

    # --- Verify round-trip ---
    print("\nVerifying data round-trip...")
    loaded_train = load_traces(data_dir / "train.jsonl")
    assert len(loaded_train) == len(train), f"Round-trip mismatch: {len(loaded_train)} != {len(train)}"
    print("  Round-trip OK.")

    print(f"\nData pipeline complete. Files saved to {data_dir}/")
    print(f"  synthetic.jsonl  — {len(synthetic)} synthetic traces")
    print(f"  train.jsonl      — {len(train)} traces (training)")
    print(f"  cal.jsonl        — {len(cal)} traces (calibration)")
    print(f"  test.jsonl       — {len(test)} traces (test)")
    print(f"  vocab.json       — {vocab.vocab_size} tokens")


if __name__ == "__main__":
    main()
