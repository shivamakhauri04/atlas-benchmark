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

# SWE-bench Verified mini-swe-agent cohorts (labels from SWE-bench/experiments, actions from S3)
git clone https://github.com/SWE-bench/experiments /tmp/swe-exp
git -C /tmp/swe-exp checkout 40f164d5b8f1d249bf95a6df8b74b577fd8e519d
mkdir -p data/swebench_mini/labels
for e in $(cat inputs/a13_primary.txt inputs/a13_secondary.txt); do
  cp /tmp/swe-exp/evaluation/verified/$e/per_instance_details.json data/swebench_mini/labels/$e.json
done
uv run python scripts/fetch_swebench_mini.py inputs/a13_primary.txt inputs/a13_secondary.txt

# tau2-bench: Sierra-run submissions, public bucket sierra-tau-bench-public
uv run python scripts/fetch_tau2.py inputs/a19_tau2.txt retail airline

echo "TRAIL, InjecAgent, R-Judge, ToolBench and SWE-agent go under data/; see DATA_LICENSES.md."
echo "API-Bank, ToolACE, AgentHarm and ATBench load from Hugging Face at run time."
