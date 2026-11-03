# Pre-registration v50: analysis choices (frozen at commit; amendments append-only)

**Date:** 2026-10-14. **No corrected d, grid, or decomposition exists at the time of writing.**
**Gate artifact:** the gate record (not included)
(status PASS, amendments A1–A2). This file fixes every analysis choice the gate leaves open.
**Evidence used to write it:** `notes/label-mechanism-audit.md` (construction only, no labels).

## 1. Data

- **AgentDojo:** commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60` (`min_tool_calls=1`, see A3).
  - Attack `important_instructions` only.
  - Positive = `security: true` (the injection succeeded); negative = `security: false` (resisted).
  - `none` runs are excluded.
  - Primary corpus is gpt-4o-2024-05-13. Every other backbone with ≥30 positives is a sub-corpus
    (E1 and the A2 executed stratum).
- **StableToolBench:** HF snapshot `ea70398f`. Label = `win == False`, which means no final answer
  was produced (verified in 765/765 files). `chatgpt_cot` is primary. The other 9 folders are
  sub-corpora.
- **TRAIL:** the existing loader, label = error_frac > 0.3. GAIA and SWE splits are sub-corpora if
  each has ≥30 positives.
- **SWE-agent:** the existing loader (first 5,000 streamed rows). A sensitivity run without
  `MAX_TRACE_LEN` is secondary.
- **Excluded from all headline analyses** (reported in the audit only):
  - InjecAgent and AgentHarm: no agent execution
  - ATBench: simulated
  - R-Judge: curated, provenance unconfirmed
- **Injected datasets** (Synthetic, ToolBench, ToolACE, API-Bank): stratum (iii), mechanism only.
- **Features:** `trace_length`, `low_diversity`, `repeat_rate`, `mean_arg_length`, `arg_diversity`,
  `arg_empty_rate` (`src/data/domain_loader.py:19-78`).

## 2. Provenance classification

- **R1, R2, R3 as in the gate prereg.** R2 threshold: a shift of ≥0.2 pooled SD on ≥1,000 clean
  traces (or all of them, if fewer).
- **Sign predictions:** `notes/prereg-v50-sign-predictions.csv`. Inputs are code, documentation, and
  label-free clean-trace statistics only. It is committed before `v50_grid.json` exists.
- **Annotators:** 2 blind subagents, each with a different prompt. They get loader code and dataset
  cards only, never any d value or any part of this file's §5. Agreement is Fleiss' κ, and κ < 0.6
  means the headline uses the annotators' majority classification.

## 3. Estimation

- **Sign estimation and scoring are split.** Each dataset is split 50/50, stratified by label, with
  seeds 0–9. Signs come from half A, and AUROC and reversal scoring use half B. The d used in scoring
  is the mean of the 10 seed estimates on half A.
- **Reversal of a (pair, feature) cell:** both |d| > τ and opposite signs. τ ∈ {0.1, 0.2}, with 0.1
  primary.
- **Detectors:**
  - LR, kNN and RF (`make_classifiers` in `scripts/run_polarity_invariant_training.py`)
  - TrajAD (Qwen3-4B QLoRA, GPU)
  - a 2-layer transformer: hashed tool names, 512 buckets, width 64, Adam with lr 1e-3, batch 64,
    20 epochs, dropout 0.1
  - Primary for anti-prediction: kNN and RF. LR is secondary.
- **Transfer:** train on source half A, test on target half B. Seeds resample both. PAFC is reported
  as an oracle only. PAFC-k uses signs from k stratified target labels drawn from target half A.

## 4. Tests

- **Primary:** `sign_prediction_accuracy` per stratum, with a Clopper-Pearson 95% CI, against
  `pooled_sign_baseline` (leave-one-dataset-out pooled sign).
- **Secondary:** `reversal_rate_ratio` under the 4 analyses, plus a placebo (1,000 permutations of
  provenance labels with class counts kept). The ratio must exceed `placebo_p95`.
- **Residual:** the within-target flip effect on construction-free features of executed datasets,
  tested by domain-level sign-flip permutation (minimum p 1/16 at 4 targets). Effect size is reported
  whatever its sign. No significance claim is made.
- **Multiplicity:** primary tests are Holm-corrected across the 2 primary strata. Everything else is
  descriptive.

## 5. Decision table (applied on the corrected grid)

| Executed accuracy > pooled baseline + 10 | External accuracy > 60% | Ratio > placebo_p95 in all 4 | Headline |
|---|---|---|---|
| yes | yes | yes | "Built Into the Labels": label construction predicts signs and reversals, in-pool and out of sample |
| yes | no | any | Claim scoped to the earlier benchmark's pool; the external result is reported as a failed generalization |
| no | any | yes | The ratio holds but the signs are not predicted. Claim: "construction-tied features reverse more"; no predictive claim |
| no | any | no | Thesis dead. The paper becomes the audit, the correction log, and the residual. |

## 6. Power statement

The executed stratum is the pooled tied cells across ~20 AgentDojo backbones, 10 StableToolBench
folders, and up to 2 TRAIL splits, so about 30–50 cells. With 40 cells, a 20-point gap over a 50%
baseline has a binomial power of about 0.8 at α = 0.05. The external stratum's cell count is unknown
until the TrajBench and AgentDrift structure is checked. That count is recorded in an amendment before
scoring.

## Amendments

### A3 — 2026-10-14, BEFORE any d (found while testing the fixed loader)

- **Why.** `load_agentdojo` defaults to `min_tool_calls=3`. On gpt-4o `important_instructions`
  that keeps 457 runs (60.2% positive). With `min_tool_calls=1` it keeps 620 runs (48.4% positive);
  over all 629 runs, 300 succeed, which matches the upstream ASR of 47.7%. The filter is a selection
  on trace length, and trace length is tied to the label. That would distort exactly the d this
  study predicts.
- **Change.** AgentDojo uses `min_tool_calls=1` (primary), with ≥3 as a sensitivity analysis.
  Every other length-based inclusion filter is reported both with and without the filter:
  - StableToolBench drops traces with < 2 actions
  - SWE-agent applies `MIN_TRACE_LEN..MAX_TRACE_LEN`
  - TRAIL applies none
- **Made before any d was computed:** yes.

### A4 — 2026-10-14, declared before the author saw any d result

- **Deviation.** §2 says the two blind annotators classify "before any d is computed". They were
  launched while `run_v50_decomposition.py` was already computing d in the background. They read
  only `src/data/{trace_loader,domain_loader,schema}.py` and public dataset documentation, and they
  are instructed not to open `output/`, `notes/`, `papers/`, `.claude/`, or any results or data
  file. Each must confirm that in writing.
- **What still holds.** Neither the annotators nor the author had seen any d value when the
  annotators started. The author's own predictions were committed earlier (fa9840c).
- **External stratum.** TrajBench and AgentDrift predictions are being derived from their papers
  and code only. They will be committed before any of their d values is computed, with the cell
  count recorded here at that time.

### A5 — 2026-10-14, external stratum fixed before any AgentDrift data was downloaded

- **TrajBench is unavailable.** The paper has no data release, no repository, no HF dataset and no
  availability statement, and the community reproduction (github.com/virtualmase/trajad-reproduction)
  confirms this. It is dropped from scoring and recorded as unavailable in the predictions file.
  Its construction is still audited from the paper:
  - process_inefficiency inserts steps: length +, R2
  - unwarranted_continuation continues past completion: length +, R3
- **AgentDrift** (CC BY 4.0, github.com/Asif-0209/AgentDrift) is the external stratum.
  - **Corpora:** 4 attack categories × 5 domains = 20 binary corpora, each vs that domain's 800
    benign trajectories.
  - **Prediction:** trace_length +, by R3, from the label grammars. All other features are free.
  - **Hard negatives:** they share benign's grammar and are not predicted.
  - **Count:** 20 predicted cells. These are appended to `prereg-v50-sign-predictions.csv` in the
    same commit as this amendment.
- **Quantitative replication.** AgentDrift has 5 domains, so decomposition C also runs on its 20
  corpora, which closes ledger L7's condition. Its executed/constructed status is noted:
  trajectories are LLM-generated from a protocol, not live runs, and they are reported as such.

### A6 — 2026-10-14, provenance agreement result (recorded before the author read any d result)

- **Agreement.** Fleiss' κ = 0.33 (sign) and 0.26 (tied/free) across the authors, annotator A and
  annotator B, over 72 cells (`output/tables/v50_provenance.json`). Annotator A ran as 3+1 agents
  and B as 2 agents, because single agents hit the output limit. Every part confirmed blindness in
  writing.
- **Pre-registered consequence (§2).** κ < 0.6, so the headline uses the annotators'
  classification. With two annotators, "majority" means agreement: a cell is tied only when A and B
  give the same non-zero sign. The authors' predictions are reported as a secondary arm.
- **Effect on the executed stratum.**
  - TRAIL `trace_length` −1 and AgentDojo `trace_length` +1 are unanimous and stay.
  - StableToolBench `trace_length` (authors +1, A 0, B +1) has no annotator consensus. It leaves
    the headline stratum and stays in the authors' arm.
- **Disagreement pattern** (reported in the paper):
  - On injected datasets, the authors' R2 used measured label-free shifts, while the annotators
    reasoned from code and tied fewer argument features.
  - The authors marked excluded datasets as 0 rather than tied.
  - Annotator B tied more cells overall (27 vs 21 authors vs 18 A).

### A7 — 2026-10-14, outcome of the decision table (after all registered tests ran)

Recorded after the results, as §5 requires. Nothing in A1–A6 was changed.

- **Primary test refuted.** Executed: 95.5% vs pooled baseline 90.9%. External: 100% vs 100%.
- **Secondary ratio refuted under the primary (annotator-consensus) classification.** The ratio is
  0.84 at τ = 0.1 and 0.86 at τ = 0.2, both below the placebo 95th percentile.
- **Decision-table row:** "no | any | no → Thesis dead. The paper becomes the audit, the correction
  log, and the residual." The "Built Into the Labels" headline is withdrawn.
- **Registered experiments that now carry the paper:**
  - **E1:** cross-agent flip rate 22.5% vs split-half 5.8%. Seed-0 permutation null 9.0%
    (p = 0.005). Backbone bootstrap CI [15%, 32%]. Same-model scaffold pairs flip 23.6% vs 23.8%
    for different models. The registered refutation of agent-driven reversal is not met.
  - **E2:** always-retrain has the lowest regret in all 12 detector × k cells (0.007–0.020), against
    reuse at 0.13–0.25.
  - **Divergence on genuine transfer:** leave-one-target-out R² < 0 for KL, MMD and Wasserstein.
- **TrajAD baseline is infeasible.** No model, adapter or training data was released (TrajBench is
  unreleased; see A5). It is dropped, and the strongest available detectors are RF, kNN and LR,
  plus the existing LLM-judge runs.
- **Headline selection.** E1 and E2 were registered with their own refuting outcomes in the gate
  prereg A1 and proposal §6, so leading with them stays inside the registered branches, and no new
  critic pass is required (gate rule). The paper states that its original thesis was refuted.

### A8 — 2026-10-30, correction of a reporting error in A7 (after the results)

- **Error.** A7 listed the external stratum as refuted ("100% vs 100%"). The registered external
  condition (gate prereg A1.5, A2.1) is accuracy ≤ 60%, not the "+10 over baseline" rule that
  applies only to the executed stratum. `run_v50_decomposition.py::score` applied the executed rule
  to every stratum, and A7 copied its `refuted` flag.
- **Correct reading.** External AgentDrift: 12/12 = 100% > 60%, so it **passes**. Binomial p vs
  0.6 = 0.0022, Holm-adjusted 0.0044. The pooled null also scores 100%, which the paper states.
- **Unchanged.** The thesis is still dead. The executed stratum fails (95.5% vs 90.9%, needs +10),
  and the consensus ratio fails. The decision table makes either failure sufficient. E1's
  Holm-adjusted p is unchanged at 0.0015 (`v50_holm.json` regenerated).

### A9 — 2026-10-30, robustness checks requested by the simulated TMLR venue review (exploratory, after the results)

None of these is confirmatory. All are reported as robustness or exploratory results.
- **Ten-split rates at each threshold** (`v50_cross_agent_important_instructions.json`):
  - τ = 0.1: 22.5% vs 5.8%
  - τ = 0.2: 13.3% vs 1.2%
  - τ = 0.3: 7.5% vs 0.2%
- **Pair-type comparison, with 95% CIs over pairs.** Different-model pairs (220) give 22.5%
  [20.3, 24.7]. Same-model scaffold pairs (11) give 21.8% [12.0, 32.5]. That is underpowered: the
  intervals overlap, but they do not establish equality.
- **Second attack.** AgentDojo ran other attacks mostly on gpt-4o. Only `ignore_previous` has more
  than one configuration with at least 30 runs per class: 3 configurations, 3 pairs.
  - Cross vs within: 34.0% vs 14.7% (τ = 0.1), 19.3% vs 4.7%, 9.3% vs 2.0%.
  - The direction replicates. The pair CIs include 0 at τ ≥ 0.2.
- **E2b distance gate.** Retrain when the label-free mean per-feature Wasserstein distance exceeds
  the in-sample median over pairs. Adding it left the other four policies identical.
  - Lowest regret at k = 20 in 6 of 6 cells, and at k = 10 in 4 of 6 (reuse in 2 of 6).
  - Retrain is lowest at k ≥ 50 in 6 of 6.
  - The sign check is never lowest.
- **Within-AgentDojo distance vs transfer** (`v50_divergence_agentdojo.json`): leave-one-target-out
  R² = +0.30 and Spearman ρ = −0.51 over 462 pairs. That p is pair-level and overstated. This
  contrasts with R² < 0 across the 12 datasets with different label meanings.
- **E1 on a learned representation** (`scripts/run_v50_cross_agent_embed.py`,
  `v50_cross_agent_embed.json`). The tool-call text is embedded with all-MiniLM-L6-v2. A label-free
  PCA reduces it to 5 components, and E1's design runs on them.
  - Raw: 17.1% vs 4.3% (τ = 0.1), 10.3% vs 1.0%, 6.5% vs 0.2%.
  - Two components correlate with log length (r = 0.46, −0.37).
  - With log length partialled out within each configuration: 13.6% vs 4.8%, 6.4% vs 1.3%,
    2.8% vs 0.5%. The gap shrinks but remains.
- **Ratio test, R1-only arm under the authors' classification** (`v50_grid.json`, registered
  co-primary, previously unreported in the draft): 1.59 and 1.84 against placebo p95 of 2.02 and
  2.35, so it is refuted. Tied-cell Fleiss' κ = 0.26 (`v50_provenance.json`) is now reported next
  to the sign κ of 0.33.
- **Reproduction check, 2026-10-30.** Every 12-dataset script was re-run from scratch:
  grid, decomposition, consensus, ratio, provenance, correction log, Holm and reuse policy.
  - Two corpora had been deleted from `/tmp`: StableToolBench and AgentDrift. They were re-fetched
    at their pinned versions (HF `data_baselines.zip`; AgentDrift 014a514f).
  - **Found:** the APIBank and ToolACE injectors drew wrong tools from `list(set(...))`, so the
    injected corpora depended on PYTHONHASHSEED. The fix is `sorted(...)`. The test
    `tests/test_v50_determinism.py` failed before the fix. The API-Bank revision is now pinned and
    its files are passed in sorted order.
  - **Effect:** only APIBank's d moves, by ≤ 0.005 (low_diversity −0.400 → −0.398). No
    prediction, hit or baseline changes in 107 cells. All other outputs are byte-identical, and no
    number in the paper changes.

### A10 — 2026-10-30, second adversarial review (exploratory, after the results)

- **Holm family.** §4 named the 2 primary sign strata. `v50_holm.json` corrects over 3 tests: those
  2, plus E1's registered single-split permutation test (τ = 0.1). That is more conservative than §4,
  and it is recorded here because it was not written down before.
- **Ten-split permutation null** (`scripts/run_v50_cross_agent_tensplit_perm.py`,
  `v50_cross_agent_tensplit_perm.json`). The registered test used seed-0 halves, but the paper quotes
  ten-split averages. This test runs the same shuffle-between-configurations null on all ten splits'
  halves and averages it. 1,000 permutations.
  - Null maximum: 9.7% / 3.0% / 0.9% at τ = 0.1 / 0.2 / 0.3.
  - Observed: 22.5% / 13.3% / 7.5%.
  - p ≤ 0.001 at each threshold. Not part of the Holm family.
- **Excluded configurations.** 6 of the 7 have 6–22 successful attacks (ASR ≤ 5%). The seventh has no
  qualifying runs. The limitations section now states this selection.
- **Release reproduction, 2026-10-30.** The full `reproduce.sh` ran in a fresh anonymized export
  (`scripts/export_v50_release.py`). It found two more problems.
  - `run_v50_grid.py` read a feature cache built on Oct 14, before the hash-seed fix. The cache has no
    key, so the in-repo rerun reused stale features. The cache is deleted, and `reproduce.sh` now
    deletes it first.
  - `run_v50_r2_shift.py` had the same `list(set(...))` tool vocabulary. It is now `sorted`.
  - **Check:** the grid and the R² shift script are byte-identical under two hash seeds.
  - **Effect:** grid, reuse-policy and R²-shift values move by ≤ 0.06 on raw KL values and ≤ 0.005
    on regret. No tied flag, no predicted sign and no paper-cited number changes.
  - **Timing:** every output reproduces byte-identically in the release, in 31 min (AgentDojo) plus
    13 min (12 datasets) on an M4 CPU.

### A11 — 2026-11-02, inclusion-filter sensitivity (exploratory, after the results)

Requested by the paper-review construct-validity audit. Script `scripts/run_v50_cross_agent_filters.py`,
output `v50_cross_agent_filters.json`. Ten-split E1 design, `important_instructions`.

| Filter | Configurations | τ=0.1 | τ=0.2 | τ=0.3 |
|---|---|---|---|---|
| ≥30 per class, ≥1 call (main, reproduced) | 22 | 22.5% vs 5.8% | 13.3% vs 1.2% | 7.5% vs 0.2% |
| ≥20 per class | 26 | 27.6% vs 10.4% | 16.7% vs 4.1% | 9.6% vs 2.1% |
| ≥40 per class | 22 (same) | identical to main | identical | identical |
| ≥3 tool calls | 22 | 23.8% vs 8.0% | 12.2% vs 1.8% | 6.3% vs 0.5% |

Also on 2026-11-02: `run_v50_decomposition.py` now lists only directories (it had recorded a
`stabletoolbench/.DS_Store` load error and continued), and it fails if any corpus fails to load. No
d value, consensus number or Holm p changed.

## Registrations for the 2026-11-02 extension (committed BEFORE any of these numbers exist)

All three are exploratory, and none enters the Holm family. Each is written before its script runs.

### A12 — version-update pairs within AgentDojo

- **Question.** Does a vendor's model update, with the same scaffold and the same tasks, reverse
  features as often as any model change?
- **Strict set (3 pairs, same model line, version bump):**
  - gemini-1.5-pro-001 → gemini-1.5-pro-002
  - gpt-4-0125-preview → gpt-4-turbo-2024-04-09
  - gemini-2.0-flash-exp → gemini-2.0-flash-001
- **Successor set (9 pairs).** The strict set, plus:
  - gpt-4-turbo-2024-04-09 → gpt-4o-2024-05-13
  - claude-3-5-sonnet-20240620 → claude-3-7-sonnet-20250219
  - claude-3-sonnet-20240229 → claude-3-5-sonnet-20240620
  - meta-llama_Llama-3-70b-chat-hf → meta-llama_Llama-3.3-70B-Instruct
  - gemini-1.5-flash-001 → gemini-2.0-flash-001
  - gpt-3.5-turbo-0125 → gpt-4o-mini-2024-07-18
- **Design.** Identical to E1: ten task splits, class-balanced, five length-free features. τ=0.1 is
  primary; τ=0.2 and τ=0.3 are reported.
- **Prediction.** The successor-set reversal rate exceeds the within-configuration rate (5.8% at
  τ=0.1).
- **Refuted if** the successor-set rate is ≤ 5.8% at τ=0.1. The strict set is descriptive only,
  because 3 pairs are too few for any test.

### A13 — a second benchmark: SWE-bench Verified, many systems on the same 500 tasks

- **Label.** 1 = instance not resolved, as graded by the benchmark's own tests. This is a failure
  label, not an injection label.
- **Systems.** Every SWE-bench Verified submission whose trajectories use the SWE-agent `.traj`
  format, with at least 30 resolved and 30 unresolved instances. The list is fixed and committed
  before any d is computed.
- **Features.** The six from E1, computed from the `.traj` actions. The tool is the first token of
  each action; the argument is the rest. The five length-free features are primary, and trace length
  is reported separately. Length is not tied to this label by construction, but long failing runs
  that hit step limits are plausible.
- **Design.** As in E1: task-disjoint halves (crc32 of `seed:instance_id`), class-balanced, ten
  splits, τ=0.1 primary, a permutation null that shuffles runs between the two systems of a pair
  (1,000 permutations), and a bootstrap over systems.
- **Prediction.** The cross-system reversal rate exceeds the within-system rate and the null's 95th
  percentile.
- **Refuted if** cross ≤ within at τ=0.1, or cross ≤ the null's 95th percentile. A refutation would be
  reported in the main text as: the effect is specific to AgentDojo or to injection labels.

### A14 — a trained sequence monitor

- **Model.** A one-layer GRU over tool-name tokens, with an action-length bucket per step,
  embedding 32, hidden 32. It is trained on each source configuration's task half A, with 5 seeds
  and early stopping on a 20% split of half A. It is scored by transfer AUROC on the target's task
  half B, over the 462 AgentDojo pairs, next to the random forest on the same halves.
- **Descriptive.** No directional prediction is registered.
- **Reported:** mean transfer AUROC, the share of pairs below 0.5, and the within-configuration AUROC
  (source half A to source half B).
- **Question.** Does a learned monitor lose more or less across configurations than the shallow one?

### A13a — 2026-11-02, amendment to A13 BEFORE any download or d (system list fixed)

- **Scaffold.** The SWE-agent `.traj` cohort has 11 entries, spread over different scaffold variants
  (SWE-agent, live-SWE-agent, codesweep). We use mini-swe-agent instead, which fixes the scaffold
  version so that only the model changes, matching the AgentDojo design. Its trajectories are
  `bash-only/<entry>/trajs/<id>/<id>.traj.json` in `s3://swe-bench-submissions`.
- **Actions.** The bash commands the assistant issues.
- **Labels.** `resolved` from each entry's `per_instance_details.json` in SWE-bench/experiments.
  Entries without that file are excluded. This is an availability rule, decided before any result.
- **Primary cohort, mini-swe-agent v2.0.0** (12 systems): mini-v2.0.0_claude-4-5-haiku-high, mini-v2.0.0_claude-4-5-opus-high, mini-v2.0.0_claude-4-5-sonnet-high, mini-v2.0.0_claude-4-6-opus, mini-v2.0.0_deepseek-3-2-high, mini-v2.0.0_gemini-3-flash-high, mini-v2.0.0_glm-5-high, mini-v2.0.0_gpt-5-2-high, mini-v2.0.0_gpt-5-mini, mini-v2.0.0_kimi-k2-5-high, mini-v2.0.0_minimax-2-5-high, mini-v2.0.0_gemini-3-pro-high.
- **Secondary cohort, mini-swe-agent v1.0.0** (7 systems): mini-v1.0.0_claude-sonnet-4-20250514, mini-v1.0.0_gemini-2.5-pro, mini-v1.0.0_o3-2025-04-16, mini-v1.0.0_o4-mini-2025-04-16, mini-v1.0.0_claude-4-opus-20250514, mini-v1.0.0_qwen3-coder-480b-a35b-instruct, mini-v1.0.0_qwen2-5-coder-32b-instruct.
- Every listed system has at least 30 tasks in each class (from the published labels; no trace data
  was read). The prediction and the refuting outcome are unchanged from A13, and they apply to the
  primary cohort. The secondary cohort is reported the same way.

### A12 / A14 results (2026-11-02, after running; designs registered in fa6b10c)

- **A12, version-update pairs** (τ=0.1, ten splits; `v50_version_pairs.json`).
  - Successor set: 12.0% vs the within floor of 5.8%, so **not refuted**.
  - Strict version bumps: 4.0%, at the floor (3 pairs, descriptive only).
  - Other different-model pairs: 23.0%.
  - At τ=0.2 the successor set is 4.4% and the strict set 1.3%.
- **A14, GRU monitor** (seed-0 split, 5 seeds; `v50_gru_monitor.json`).

  | Monitor | Within AUROC | Transfer AUROC | Transfers below 0.5 |
  |---|---|---|---|
  | GRU | 0.895 | 0.744 | 2.6% |
  | Random forest, all six features | 0.829 | 0.718 | 1.7% |

  The learned monitor is better both within and after transfer, but it loses more when the
  configuration changes: a drop of 0.151 against 0.111.

### A13 result (2026-11-02, after running; registered in fa6b10c, system list in 778827b)

**REFUTED on the primary cohort**, per the registered rule. Output: `v50_swebench_cross_{v2,v1}.json`.

| Cohort | τ=0.1 cross vs within | Null p95 | p | τ=0.2 cross vs within |
|---|---|---|---|---|
| mini-swe-agent v2.0.0, 11 systems | 9.7% vs 9.1% | 11.5% | 0.94 | 3.0% vs 2.0% (null p95 2.3%, p=0.001) |
| mini-swe-agent v1.0.0, 7 systems | 5.3% vs 2.6% | 5.8% | 0.17 | — |

- **Excluded:** Gemini 3 Pro. Its published `per_instance_details.json` marks all 500 tasks
  unresolved, so the registered ≥30 rule removes it.
- **Trace length** never reverses, in either cohort.
- **The null is not an artifact.** Per-system d is large and has the same sign everywhere: failing
  runs are longer (d +0.37 to +0.91) and less diverse (+0.27 to +1.09).
- **Run filter:** E1's ≥1-action filter was applied, as A13 specified ("as in E1"). It removed 11
  runs in total.
- **Reading, hypothesis only:** reversals across agents appear when what the label selects depends
  on the agent (injection success), and not under a task-failure label that looks alike across
  models. The τ=0.2 v2 excess is small and is not the registered test.

### A15 — 2026-11-02, reuse vs retrain on SWE-bench Verified (registered BEFORE running)

- **Design.** E2b's script and design unchanged: 5 policies, k ∈ {10, 20, 50, 100}, three
  detectors, all six features and length-free. It runs on the A13a primary cohort (11 systems, 110
  ordered pairs), and the regret-difference bootstrap is over systems.
- **Prediction.** Features kept their meaning across models here (A13), so a reused monitor should
  hold up better than on AgentDojo: reuse has lower regret than retraining at k=20 for the random
  forest on all six features.
- **Refuted if** reuse regret ≥ retrain regret at k=20 (RF, all six features). Reported for every
  cell.

### A15 result (2026-11-02, after running; registered in ab31f82) — **not refuted**

`v50_reuse_swebench.json`, `v50_reuse_bootstrap_swebench.json`; 11 systems, 110 ordered pairs.
Regret difference, reuse minus retrain, with a 95% bootstrap interval over systems:

| Detector | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|
| RF | −0.092 [−0.121, −0.065] | −0.078 [−0.102, −0.056] | −0.052 [−0.065, −0.038] | −0.025 [−0.039, −0.012] |
| kNN | −0.034 [−0.049, −0.017] | −0.021 [−0.040, −0.005] | −0.006 [−0.018, 0.007] | +0.007 [−0.005, 0.021] |
| LR | −0.094 [−0.115, −0.072] | −0.076 [−0.095, −0.057] | −0.039 [−0.052, −0.028] | −0.016 [−0.028, −0.003] |

- Reuse wins 87% of pairs at k=10 for RF. On AgentDojo, retraining wins from k=50.
- **Confound:** SWE-bench monitors are weaker (the reused RF reaches an AUROC of 0.658), so k new
  labels add less.

### A16 — 2026-11-02, the label hypothesis within AgentDojo (registered BEFORE running)

- **Question.** Does the label, rather than the benchmark, decide whether features reverse across
  agents? On AgentDojo every run also records `utility`, whether the user's task was completed. So
  the same traces can be relabeled while the domain, agents, tasks and features stay fixed.
- **Primary.** The attacked `important_instructions` runs, with ≥1 tool call. Label 1 = user task
  failed (`utility=False`). Configurations need ≥30 runs per class under this label. Design: E1
  unchanged (ten task splits, balanced, five length-free features, τ=0.1 primary), plus the ten-split
  permutation null with 1,000 permutations.
- **Secondary.** The clean no-attack (`none`) runs under the same label, the closest analogue to
  SWE-bench's task-failure label.
- **Hypothesis H-label.** The paper's reading: reversals appear when the label's selection depends on
  agent behaviour. It predicts a smaller cross-minus-within excess under the utility label than under
  the security label (16.7 percentage points at τ=0.1).
- **H-label is refuted if** the utility-label excess on the primary set is ≥ 16.7 points.
- **Reported either way, in the main text.**

### A16 result (2026-11-02, after running; registered in a969cab)

`v50_utility_label.json`.

| Set | Configurations | τ=0.1 cross vs within | Excess | Null p95 | p |
|---|---|---|---|---|---|
| Primary: attacked runs, utility label | 28 | 18.7% vs 2.6% | 16.1 points | 5.7% | ≤0.001 |
| Secondary: clean runs | 22 | 28.2% vs 19.9% | 8.3 points | 22.7% | ≤0.001 |

For reference, the security label gives an excess of 16.7 points.

- **Registered rule:** H-label is refuted if the excess is ≥ 16.7 points. 16.1 < 16.7, so the rule
  does **not** fire.
- **Substantive reading:** the excess under the task-failure label is nearly identical to the
  injection label's. The data do not support H-label: label type does not explain why AgentDojo
  reverses and SWE-bench does not. The paper drops the label-selection reading, together with any
  practical rule derived from it.
- **The difference lies in the setting:** benchmark, environment, agents or model generation. These
  data cannot say which.
- **Also disclosed now (blind review):** SWE-bench v2 at τ=0.2 is 3.0% vs 2.0% within, null p95
  2.3%, p=0.001. This small excess is not in the v1 cohort (1.1% vs null p95 1.8%). The paper
  previously reported only τ=0.1.
