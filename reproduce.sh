#!/usr/bin/env bash
# Re-run every result in dependency order. Each script overwrites its own JSON in output/tables/.
set -euo pipefail
export PYTHONPATH=src:scripts
run() { echo "== $*"; uv run python "$@"; }

uv run python -m pytest -q tests

# Reason: the grid caches features with no key; a stale cache silently reuses old loader output.
rm -f output/cache/v50_primary_features.npz

run scripts/run_v50_cross_agent.py
run scripts/run_v50_cross_agent_attacks.py important_instructions
run scripts/run_v50_cross_agent_attacks.py ignore_previous
run scripts/run_v50_cross_agent_robustness.py
run scripts/run_v50_cross_agent_thresholds.py
run scripts/run_v50_cross_agent_tensplit_perm.py
run scripts/run_v50_cross_agent_embed.py
run scripts/run_v50_cross_agent_filters.py
run scripts/run_v50_version_pairs.py
run scripts/run_v50_gru_monitor.py
run scripts/run_v50_utility_label.py
run scripts/run_v50_swebench_cross.py inputs/a13_primary.txt v2
run scripts/run_v50_swebench_cross.py inputs/a13_secondary.txt v1
run scripts/run_v50_reuse_agentdojo.py swebench
run scripts/run_v50_reuse_bootstrap.py swebench
run scripts/run_v50_scaffold_mechanism.py
run scripts/run_v50_injector_crossover.py
run scripts/run_v50_divergence_agentdojo.py
run scripts/run_v50_reuse_agentdojo.py
run scripts/run_v50_reuse_bootstrap.py

run scripts/run_v50_grid.py
run scripts/run_v50_decomposition.py
run scripts/run_v50_score_consensus.py
run scripts/run_v50_ratio_consensus.py
run scripts/run_provenance_annotation.py
run scripts/run_v50_correction_log.py
run scripts/run_v50_holm.py
run scripts/run_v50_reuse_policy.py
run scripts/run_v50_r2_shift.py

run scripts/gen_v50_tables.py
run scripts/gen_v50_figures.py
