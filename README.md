# ATLAS: Agent Trace Laboratory for Anomaly Semantics

A cross-domain benchmark for evaluating robust anomaly detection in AI agent traces.

ATLAS unifies 12 agent trace datasets (28,125 traces) under a common schema and organizes them into 5 semantic anomaly groups, enabling systematic evaluation of cross-domain transfer. The benchmark exposes **feature polarity reversal** as a systematic failure mode: the same feature indicates anomaly in one domain but normality in another.

## Setup

```bash
# Install dependencies (requires Python >= 3.12)
pip install uv  # if not installed
uv sync

# Download datasets (from public sources)
bash data/download.sh

# Prepare unified traces
PYTHONPATH=src uv run python scripts/prepare_data.py
```

## Repository Structure

```
src/
  data/           Unified trace loaders for all 12 datasets
  trustlens/      Core modules: encoder, conformal calibration, polarity analysis
  baselines/      All 20 baseline implementations (6 families)
scripts/          Experiment scripts (training, evaluation, figure generation)
output/
  tables/         Pre-computed result tables (JSON)
  figures/        Pre-computed figures (PDF + PNG)
data/
  download.sh     Script to fetch all public datasets
Makefile          Convenience targets for common experiments
```

## Reproducing Results

### Quick Start: Pre-computed Results

All tables and figures from the paper are included in `output/`. Key files:

- `output/tables/v30_expanded_polarity.json` — Polarity reversal statistics
- `output/tables/v30_full_cross_domain_table.json` — Full 132-pair cross-domain AUROC
- `output/tables/v31_llm_judge_merged_analysis.json` — LLM-as-judge evaluation
- `output/tables/v31_baserate_confound_analysis.json` — Base-rate confound analysis

### Running Experiments

```bash
# Core experiments
make baselines                    # Run all baselines (heuristic, LSTM, hybrid)
make train                        # Train trace encoder
make conformal                    # Run conformal calibration

# Cross-domain evaluation (v30+)
make v30-expanded-polarity        # Polarity reversal analysis (12 datasets)
make v30-expanded-cross-domain    # Full cross-domain AUROC matrix

# Diagnostic analyses
make v31-fair-da                  # DANN/CORAL on matched features
make v31-conformal-coverage       # Conformal coverage validation
make v31-transfer-prediction      # Transfer predictability analysis

# Generate paper figures
make v31-polarity-figures         # Polarity heatmaps and violin plots
```

### Full Reproduction Pipeline

```bash
# 1. Download and prepare data
bash data/download.sh
PYTHONPATH=src uv run python scripts/prepare_data.py

# 2. Run all v30 experiments (12-dataset benchmark)
make v30-all

# 3. Run all v31 experiments (diagnostic analyses)
make v31-all

# 4. Generate figures
make v31-polarity-figures
```

## Datasets

| Dataset | Source | Traces | Group |
|---------|--------|--------|-------|
| Synthetic | Generated | 2,000 | Injected |
| ToolACE | [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | 2,563 | Injected |
| SWE-agent | [SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | 5,000 | Failure |
| TRAIL | [patronus-ai/TRAIL-Benchmark](https://github.com/patronus-ai/TRAIL-Benchmark) | 148 | Failure |
| StableToolBench | [THUNLP-MT/StableToolBench](https://github.com/THUNLP-MT/StableToolBench) | 763 | Failure |
| ATBench | [AI-Secure/ATBench](https://huggingface.co/datasets/AI-Secure/ATBench) | 499 | Safety |
| R-Judge | [Lordog/R-Judge](https://github.com/Lordog/R-Judge) | 522 | Safety |
| AgentHarm | [ai-safety-institute/AgentHarm](https://huggingface.co/datasets/ai-safety-institute/AgentHarm) | 352 | Safety |
| InjecAgent | [uiuc-kang-lab/InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | 2,125 | Prompt-inj |
| AgentDojo | [ethz-spylab/agentdojo](https://github.com/ethz-spylab/agentdojo) | 4,153 | Prompt-inj |
| ToolBench | [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) | 5,000 | API-misuse |
| API-Bank | [AlibabaResearch/DAMO-ConvAI](https://github.com/AlibabaResearch/DAMO-ConvAI) | 5,000 | API-misuse |

All source datasets are publicly available under their respective licenses.

## Baselines (20 methods, 6 families)

- **Heuristic** (3): trace length, low diversity, repeat rate scoring
- **Learned** (3): LSTM encoder, hybrid (heuristic+LSTM), MLP
- **Domain adaptation** (3): DANN, Deep CORAL, MMD
- **Few-shot** (4): logistic regression, k-NN, random forest, prototype
- **Conformal** (4): split CP, PAFC-Conformal, weighted CP, Mondrian CP
- **LLM-as-judge** (3): Claude Sonnet 4.6, Claude Haiku 4.5, Llama 3.3 70B

## License

MIT
