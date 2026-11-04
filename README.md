# Code for "When the Agent Changes, the Monitor's Features Change Meaning"

This repository contains the code, registered inputs, and result files for the paper. It
reproduces every number in the paper from the public datasets.

## Contents

- `src/data/`: one loader per dataset. Each docstring states the label rule. `docs/label-mechanism-audit.md` describes the rule and its provenance for all 12 datasets.
- `src/traceshift/v50.py`: Cohen's d, the reversal rule, task-disjoint halves, and transfer AUROC.
- `scripts/run_v50_*.py`: one script per result. Each script writes one JSON file to `output/tables/`.
- `inputs/`: the label-free sign predictions and the two blind annotators' classifications. These were committed before any effect size was computed.
- `docs/prereg-v50.md`: the pre-registration and its dated amendments A1 to A10.
- `output/tables/v50_*.json`: the results as reported in the paper.
- `tests/`: the label rule, the sign and flip rules, transfer scoring, and hash-seed determinism.

## Setup

```bash
uv sync
bash fetch_data.sh      # downloads AgentDojo, StableToolBench, AgentDrift, SWE-bench trajectories at pinned versions
```

Some datasets load from Hugging Face at run time. Read `DATA_LICENSES.md` before you use any
dataset. This repository does not redistribute third-party data.

## Reproduce

```bash
bash reproduce.sh
```

`reproduce.sh` runs the tests and then every script in dependency order. We measured it on an
Apple M4 laptop CPU with the data already downloaded. The AgentDojo scripts took 31 minutes, and
the reuse experiment (11 minutes) and the ten-split permutation null (10 minutes) took the longest.
The 12-dataset chain took 13 minutes. Every output matched the committed JSON byte for byte.

## Which script produces which result

| Paper result | Script | Output |
|---|---|---|
| Cross-configuration reversal rates (ten splits) | `run_v50_cross_agent_attacks.py important_instructions` | `v50_cross_agent_important_instructions.json` |
| Registered single-split permutation test | `run_v50_cross_agent_thresholds.py` | `v50_cross_agent_thresholds.json` |
| Ten-split permutation null | `run_v50_cross_agent_tensplit_perm.py` | `v50_cross_agent_tensplit_perm.json` |
| Per-feature rates, configuration bootstrap | `run_v50_cross_agent_robustness.py` | `v50_cross_agent_robustness.json` |
| Second attack | `run_v50_cross_agent_attacks.py ignore_previous` | `v50_cross_agent_ignore_previous.json` |
| Learned features | `run_v50_cross_agent_embed.py` | `v50_cross_agent_embed.json` |
| Inclusion-filter robustness | `run_v50_cross_agent_filters.py` | `v50_cross_agent_filters.json` |
| Version-update pairs | `run_v50_version_pairs.py` | `v50_version_pairs.json` |
| Trained sequence monitor | `run_v50_gru_monitor.py` | `v50_gru_monitor.json` |
| Task-failure relabel of AgentDojo | `run_v50_utility_label.py` | `v50_utility_label.json` |
| Intervals for the relabel and the ten-split headline | `run_v50_utility_label_ci.py`, `run_v50_cross_agent_tensplit_ci.py` | `v50_utility_label_ci.json`, `v50_cross_agent_tensplit_ci.json` |
| Capability spread on AgentDojo | `run_v50_capability_split.py` | `v50_capability_split.json` |
| tau2-bench replication | `run_v50_tau2_cross.py` | `v50_tau2_cross_{retail,airline}.json` |
| Newest AgentDojo configurations; tau2 harness versions (exploratory) | `run_v50_generation_harness_checks.py` | `v50_generation_harness_checks.json` |
| SWE-bench Verified replication | `run_v50_swebench_cross.py` | `v50_swebench_cross_{v2,v1}.json` |
| Reuse decision on SWE-bench | `run_v50_reuse_agentdojo.py swebench`, `run_v50_reuse_bootstrap.py swebench` | `v50_reuse_swebench.json`, `v50_reuse_bootstrap_swebench.json` |
| Defense mechanism (Table 3) | `run_v50_scaffold_mechanism.py` | `v50_scaffold_mechanism.json` |
| Reuse decision on AgentDojo | `run_v50_reuse_agentdojo.py`, `run_v50_reuse_bootstrap.py` | `v50_reuse_agentdojo.json`, `v50_reuse_bootstrap.json` |
| Distance vs transfer within AgentDojo | `run_v50_divergence_agentdojo.py` | `v50_divergence_agentdojo.json` |
| 12-dataset grid, ratio test, distance regression | `run_v50_grid.py` | `v50_grid.json` |
| Sign-prediction test | `run_v50_decomposition.py`, `run_v50_score_consensus.py` | `v50_decomposition*.json` |
| Annotator agreement | `run_provenance_annotation.py` | `v50_provenance.json` |
| Holm correction | `run_v50_holm.py` | `v50_holm.json` |
| Corrections to the earlier version | `run_v50_correction_log.py` | `v50_correction_log.json` |
| Reuse decision across 12 datasets | `run_v50_reuse_policy.py` | `v50_reuse_policy.json` |
| Injector crossover | `run_v50_injector_crossover.py` | `v50_injector_crossover.json` |

## License

The code is under the MIT license (`LICENSE`), and the license covers our code only. Each dataset
keeps its own terms (`DATA_LICENSES.md`).
