PY=uv run python
PYPATH=PYTHONPATH=../../lib:src

train:
	$(PYPATH) $(PY) scripts/train_encoder.py

train-no-args:
	$(PYPATH) $(PY) scripts/train_encoder.py --no-args --epochs 50 --checkpoint-dir checkpoints/no_args

train-resume:
	$(PYPATH) $(PY) scripts/train_encoder.py --resume-from checkpoints/

conformal:
	$(PYPATH) $(PY) scripts/run_conformal.py --no-args --checkpoint checkpoints/no_args/latest.pt

baselines:
	$(PYPATH) $(PY) scripts/run_baselines.py

cp-comparison:
	$(PYPATH) $(PY) scripts/run_cp_comparison.py --no-args --encoder-epochs 30

shift-experiment:
	$(PYPATH) $(PY) scripts/run_shift_experiment.py --no-args --checkpoint checkpoints/no_args/latest.pt

held-out-attacks:
	$(PYPATH) $(PY) scripts/run_held_out_attacks.py --no-args --encoder-epochs 30

toolbench-transfer:
	$(PYPATH) $(PY) scripts/run_toolbench_transfer.py

trail-transfer:
	$(PYPATH) $(PY) scripts/run_trail_transfer.py --trail-dir data/trail/TRAIL-Benchmark --skip-download

agentharm:
	$(PYPATH) $(PY) scripts/run_agentharm_eval.py

paper-figs:
	$(PYPATH) $(PY) scripts/make_figures.py

prepare-data:
	$(PYPATH) $(PY) scripts/prepare_data.py

# v11: Novelty elevation experiments
rac-comparison:
	$(PYPATH) $(PY) scripts/run_rac_comparison.py --encoder-epochs 30

per-tool-coverage:
	$(PYPATH) $(PY) scripts/run_per_tool_coverage.py --encoder-epochs 30

speculative-benchmark:
	$(PYPATH) $(PY) scripts/run_speculative_benchmark.py --encoder-epochs 30

v11-all: rac-comparison per-tool-coverage speculative-benchmark

# v12: Fix experimental design flaws + implementation quality
v12-speculative:
	$(PYPATH) $(PY) scripts/run_speculative_benchmark.py --encoder-epochs 30 \
		--alpha-cons 0.05 --alpha-spec 0.15 --alpha-spec-sweep 0.10,0.15,0.20,0.30

v12-cp-comparison:
	$(PYPATH) $(PY) scripts/run_cp_comparison.py --encoder-epochs 30 --consequence

v12-per-tool:
	$(PYPATH) $(PY) scripts/run_per_tool_coverage.py --encoder-epochs 30

v12-monitoring-diagnosis:
	$(PYPATH) $(PY) scripts/run_monitoring_diagnosis.py --encoder-epochs 30

v12-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v12-all: v12-speculative v12-cp-comparison v12-per-tool v12-monitoring-diagnosis v12-figures

# v13: Outcome leakage analysis + robustness + LSTM+Outcome + N=10 seeds
v13-leakage:
	$(PYPATH) $(PY) scripts/run_leakage_analysis.py --encoder-epochs 30

v13-noisy:
	$(PYPATH) $(PY) scripts/run_noisy_outcome_sweep.py --encoder-epochs 30

v13-seeds10:
	$(PYPATH) $(PY) scripts/run_cp_comparison.py --encoder-epochs 30 --consequence \
		--seeds 42,123,456,789,1024,2048,4096,8192,16384,32768

v13-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v13-all: v13-leakage v13-noisy v13-seeds10 v13-figures

v14-swe-agent:
	$(PYPATH) $(PY) scripts/run_swe_agent_eval.py --max-traces 5000 --epochs 20

# v15: Scaled SWE-agent eval (20K traces, d_model=128, 30 epochs, baselines + per-category)
v15-swe-agent:
	$(PYPATH) $(PY) scripts/run_swe_agent_eval.py --max-traces 20000 --epochs 30 --d-model 128

# v16: Reframing eval — simple baselines on synthetic + RAC with honest scorers
v16-reframing:
	$(PYPATH) $(PY) scripts/run_reframing_eval.py --encoder-epochs 30

# v17: Domain inversion depth analysis (pure analysis, no training)
v17-domain-analysis:
	$(PYPATH) $(PY) scripts/run_domain_analysis.py --max-swe-traces 5000 --n-bootstrap 100

# v18: Score improvement (ATBench, conditional coverage, RAC SWE-agent, 10-seed)
v18-atbench:
	$(PYPATH) $(PY) scripts/run_atbench_eval.py --epochs 30 --d-model 64

v18-coverage-analysis:
	$(PYPATH) $(PY) scripts/run_coverage_analysis.py

v18-rac-swe:
	$(PYPATH) $(PY) scripts/run_rac_swe_eval.py --max-traces 20000 --epochs 30 --d-model 128

v18-domain-10seed:
	$(PYPATH) $(PY) scripts/run_domain_analysis.py --version v18 \
		--seeds 42 123 456 789 1024 2024 3024 4024 5024 6024 \
		--max-swe-traces 5000 --n-bootstrap 100 --include-atbench

v18-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v18-all: v18-atbench v18-coverage-analysis v18-rac-swe v18-domain-10seed v18-figures

# v20: Hybrid scorer + domain adaptation
v20-hybrid:
	$(PYPATH) $(PY) scripts/run_hybrid_eval.py \
		--max-swe-traces 20000 --swe-epochs 30 --swe-d-model 128 \
		--atbench-epochs 30 --atbench-d-model 64 \
		--synthetic-epochs 30 --synthetic-d-model 128

v20-hybrid-10seed:
	$(PYPATH) $(PY) scripts/run_hybrid_eval.py --10seed \
		--max-swe-traces 20000 --swe-epochs 30 --swe-d-model 128 \
		--atbench-epochs 30 --atbench-d-model 64 \
		--skip-synthetic --skip-transfer

v20-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v20-all: v20-hybrid v20-figures

# v26: FSI + PA-CP + Adversarial domain selection
v26-fsi:
	$(PYPATH) $(PY) scripts/run_fsi_analysis.py

v26-pacp:
	$(PYPATH) $(PY) scripts/run_pacp_eval.py

v26-adversarial:
	$(PYPATH) $(PY) scripts/run_adversarial_domain.py

v26-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v26-all: v26-fsi v26-pacp v26-adversarial v26-figures

# v27: D5 Fix Sprint — matched ablation + 10-seed stats + real-data validation
v27-cp-matched:
	$(PYPATH) $(PY) scripts/run_cp_comparison.py --no-args --encoder-epochs 30 --consequence \
		--seeds 42,123,456,789,1024,2048,4096,8192,16384,32768 --resplit

v27-ablations-10seed:
	$(PYPATH) $(PY) scripts/run_ablations.py --epochs 30 \
		--seeds 42 123 456 789 1024 2048 4096 8192 16384 32768

v27-swe-validation:
	$(PYPATH) $(PY) scripts/run_swe_agent_eval.py --max-traces 5000 --epochs 20 \
		--include-transformer

v27-baselines:
	$(PYPATH) $(PY) scripts/run_baselines.py

v27-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v27-all: v27-cp-matched v27-ablations-10seed v27-swe-validation v27-baselines v27-figures

# v28: DAPL + DANN/CORAL + Unified Semantics
v28-dapl:
	$(PYPATH) $(PY) scripts/run_dapl_eval.py

v28-dapl-quick:
	$(PYPATH) $(PY) scripts/run_dapl_eval.py --quick

v28-unified-semantics:
	$(PYPATH) $(PY) scripts/run_unified_semantics.py

v28-unified-quick:
	$(PYPATH) $(PY) scripts/run_unified_semantics.py --quick

v28-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v28-all: v28-dapl v28-unified-semantics v28-figures

# v29: Address NeurIPS structural weaknesses
v29-expanded-polarity:
	$(PYPATH) $(PY) scripts/run_expanded_polarity.py

v29-expanded-polarity-quick:
	$(PYPATH) $(PY) scripts/run_expanded_polarity.py --quick

v29-expanded-cross-domain:
	$(PYPATH) $(PY) scripts/run_expanded_cross_domain.py

v29-expanded-cross-domain-quick:
	$(PYPATH) $(PY) scripts/run_expanded_cross_domain.py --quick

v29-pafc-conformal:
	$(PYPATH) $(PY) scripts/run_pafc_conformal.py

v29-pafc-conformal-quick:
	$(PYPATH) $(PY) scripts/run_pafc_conformal.py --quick

v29-fewshot-baselines:
	$(PYPATH) $(PY) scripts/run_fewshot_baselines.py

v29-fewshot-baselines-quick:
	$(PYPATH) $(PY) scripts/run_fewshot_baselines.py --quick

v29-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v29-all: v29-expanded-polarity v29-expanded-cross-domain v29-pafc-conformal v29-fewshot-baselines v29-figures

v29-quick: v29-expanded-polarity-quick v29-pafc-conformal-quick v29-fewshot-baselines-quick

# v30: D&B pivot — expanded to 8 datasets (ToolBench + TRAIL)
v30-expanded-polarity:
	$(PYPATH) $(PY) scripts/run_expanded_polarity.py

v30-expanded-polarity-quick:
	$(PYPATH) $(PY) scripts/run_expanded_polarity.py --quick

v30-expanded-cross-domain:
	$(PYPATH) $(PY) scripts/run_expanded_cross_domain.py

v30-expanded-cross-domain-quick:
	$(PYPATH) $(PY) scripts/run_expanded_cross_domain.py --quick

v30-figures:
	$(PYPATH) $(PY) scripts/make_figures.py

v30-all: v30-expanded-polarity v30-expanded-cross-domain v30-figures

v30-quick: v30-expanded-polarity-quick v30-expanded-cross-domain-quick

# v31: Pivot to "When Anomaly Detectors Anti-Predict" — gap-filling experiments
v31-fair-da:
	$(PYPATH) $(PY) scripts/run_fair_da_comparison.py

v31-fair-da-quick:
	$(PYPATH) $(PY) scripts/run_fair_da_comparison.py --quick

v31-conformal-coverage:
	$(PYPATH) $(PY) scripts/run_conformal_coverage_validation.py

v31-conformal-coverage-quick:
	$(PYPATH) $(PY) scripts/run_conformal_coverage_validation.py --quick

v31-transfer-prediction:
	$(PYPATH) $(PY) scripts/run_transfer_prediction.py

v31-polarity-figures:
	$(PYPATH) $(PY) scripts/gen_polarity_paper_figures.py

v31-all: v31-fair-da v31-conformal-coverage v31-transfer-prediction v31-polarity-figures

v31-quick: v31-fair-da-quick v31-conformal-coverage-quick v31-transfer-prediction v31-polarity-figures
