#!/usr/bin/env bash
# Download the datasets the scripts read from /tmp, at the versions used in the paper.
set -euo pipefail

git clone https://github.com/ethz-spylab/agentdojo /tmp/agentdojo_repo
git -C /tmp/agentdojo_repo checkout 089ed468cf3ed0322acc66b0211f26d9d90dbf60
mkdir -p /tmp/agentdojo_data && ln -sfn /tmp/agentdojo_repo/runs /tmp/agentdojo_data/runs

git clone https://github.com/Asif-0209/AgentDrift /tmp/agentdrift
git -C /tmp/agentdrift checkout 014a514fa998b4ac4519579fceb8a5884b379bda

uv run python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("stabletoolbench/baselines", "data_baselines.zip",
                      repo_type="dataset", local_dir="/tmp/stb_dl"))
PY
mkdir -p /tmp/stabletoolbench_baselines
unzip -q -o /tmp/stb_dl/data_baselines.zip -d /tmp/stabletoolbench_baselines

echo "TRAIL, InjecAgent, R-Judge, ToolBench and SWE-agent go under data/; see DATA_LICENSES.md."
echo "API-Bank, ToolACE, AgentHarm and ATBench load from Hugging Face at run time."
