#!/usr/bin/env bash
# Download all 12 ATLAS benchmark datasets from their public sources.
# Usage: bash data/download.sh [DATA_DIR]
#
# Each dataset is downloaded into DATA_DIR (default: ./data/).
# Some datasets require the `datasets` Python package (pip install datasets).

set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "=== ATLAS Dataset Download Script ==="
echo "Target directory: $DATA_DIR"
echo ""

# 1. Synthetic — generated at runtime, no download needed
echo "[1/12] Synthetic: Generated at runtime (no download needed)"

# 2. ToolACE — HuggingFace
echo "[2/12] ToolACE: Downloading from HuggingFace..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('Team-ACE/ToolACE', split='train')
ds.to_json('${DATA_DIR}/toolace/toolace.json')
print(f'  Downloaded {len(ds)} records')
" 2>/dev/null || echo "  SKIP: Install 'datasets' package (pip install datasets)"

# 3. SWE-agent — HuggingFace (SWE-bench trajectories)
echo "[3/12] SWE-agent: Downloading from HuggingFace..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
ds.to_json('${DATA_DIR}/swe_agent/swe_bench_verified.json')
print(f'  Downloaded {len(ds)} records')
" 2>/dev/null || echo "  SKIP: Install 'datasets' package (pip install datasets)"

# 4. TRAIL — Git clone from PatronusAI
echo "[4/12] TRAIL: Cloning from GitHub..."
if [ ! -d "${DATA_DIR}/trail/TRAIL-Benchmark" ]; then
    git clone --depth 1 https://github.com/patronus-ai/TRAIL-Benchmark.git "${DATA_DIR}/trail/TRAIL-Benchmark"
else
    echo "  Already exists, skipping"
fi

# 5. StableToolBench — Git clone
echo "[5/12] StableToolBench: Cloning from GitHub..."
if [ ! -d "${DATA_DIR}/stabletoolbench" ]; then
    git clone --depth 1 https://github.com/THUNLP-MT/StableToolBench.git /tmp/StableToolBench
    mkdir -p "${DATA_DIR}/stabletoolbench"
    echo "  NOTE: Copy baseline results from StableToolBench/data_baselines/ to ${DATA_DIR}/stabletoolbench/"
else
    echo "  Already exists, skipping"
fi

# 6. ATBench — HuggingFace
echo "[6/12] ATBench: Downloading from HuggingFace..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('AI-Secure/ATBench', split='test')
ds.to_json('${DATA_DIR}/atbench/atbench.json')
print(f'  Downloaded {len(ds)} records')
" 2>/dev/null || echo "  SKIP: Install 'datasets' package (pip install datasets)"

# 7. R-Judge — Git clone
echo "[7/12] R-Judge: Cloning from GitHub..."
if [ ! -d "${DATA_DIR}/rjudge" ]; then
    git clone --depth 1 https://github.com/Lordog/R-Judge.git "${DATA_DIR}/rjudge"
else
    echo "  Already exists, skipping"
fi

# 8. AgentHarm — HuggingFace
echo "[8/12] AgentHarm: Downloading from HuggingFace..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('ai-safety-institute/AgentHarm', split='test')
ds.to_json('${DATA_DIR}/agentharm/agentharm.json')
print(f'  Downloaded {len(ds)} records')
" 2>/dev/null || echo "  SKIP: Install 'datasets' package (pip install datasets)"

# 9. InjecAgent — Git clone
echo "[9/12] InjecAgent: Cloning from GitHub..."
if [ ! -d "${DATA_DIR}/injecagent" ]; then
    git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git "${DATA_DIR}/injecagent"
else
    echo "  Already exists, skipping"
fi

# 10. AgentDojo — HuggingFace / pip
echo "[10/12] AgentDojo: Install via pip (pip install agentdojo) and run evaluation"
echo "  See: https://github.com/ethz-spylab/agentdojo"

# 11. ToolBench — Git clone
echo "[11/12] ToolBench: Cloning from GitHub..."
if [ ! -d "${DATA_DIR}/toolbench" ]; then
    git clone --depth 1 https://github.com/OpenBMB/ToolBench.git /tmp/ToolBench
    mkdir -p "${DATA_DIR}/toolbench"
    echo "  NOTE: Follow ToolBench instructions to download training data"
    echo "  Place toolbench_train.json in ${DATA_DIR}/toolbench/"
else
    echo "  Already exists, skipping"
fi

# 12. API-Bank — Git clone
echo "[12/12] API-Bank: Cloning from GitHub..."
if [ ! -d "${DATA_DIR}/apibank" ]; then
    git clone --depth 1 https://github.com/AlibabaResearch/DAMO-ConvAI.git /tmp/DAMO-ConvAI
    mkdir -p "${DATA_DIR}/apibank"
    cp -r /tmp/DAMO-ConvAI/api-bank/* "${DATA_DIR}/apibank/"
else
    echo "  Already exists, skipping"
fi

echo ""
echo "=== Download complete ==="
echo "Some datasets require additional steps (see notes above)."
echo "Run 'uv run python scripts/prepare_data.py' to normalize all datasets."
