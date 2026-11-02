# Label-mechanism audit (Phase 0 step 2, 2026-10-13)

The question for each of the 12 datasets: does the anomaly label fix the sign of a structural feature
(`trace_length`, `low_diversity`, `repeat_rate`) by construction? If it does, a "polarity reversal"
on that feature measures how we built the labels, not how domains differ.

This file records construction only. It does not use any corrected Cohen's d value. Those values are
computed after `prereg-v50.md` is committed.

Code paths are relative to the repository root.

## Pinned data sources

| Source | Pin |
|---|---|
| AgentDojo | `github.com/ethz-spylab/agentdojo` commit `089ed468cf3ed0322acc66b0211f26d9d90dbf60`, `runs/` (29 run directories, 477 MB) |
| StableToolBench | HF `stabletoolbench/baselines` snapshot `ea70398f955e7b58f8807187936818e794ce5140`, `data_baselines.zip` |

## The shared injector, measured

`_inject_anomaly` (`src/data/trace_loader.py:217-262`) keeps `actions[:inject_pos]`, where
`inject_pos` is drawn from `[L/2, L-1]`. It then appends a 2- or 3-step pattern and up to 2 tail
actions.

Mean change in length after injection (4,000 draws per L, seed 0):

| L | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δlength | +1.80 | +1.48 | +1.45 | +1.02 | +0.56 | +0.06 | −0.44 | −0.89 | −1.45 | −2.40 | −4.89 | −9.98 |

The crossover is at about 10 actions. Any dataset whose traces are mostly shorter than 10 gets a
positive `d_length` after injection, and any dataset whose traces are mostly longer gets a negative
one. The sign depends only on the length distribution of the clean data. Synthetic (d_length −0.32,
published) and ToolBench (+1.67, published) both use this injector. Their reversal is this table.

## Per-dataset audit

"Executed" means a real agent run produced the trace. "Constructed" means we or the dataset authors
assembled the trace or label without a run.

| Dataset | Label definition (file:line) | Trace origin | Feature fixed by construction | Verdict |
|---|---|---|---|---|
| Synthetic | `_inject_anomaly` at 15% (`domain_loader.py:93-101`, `trace_loader.py:217-262`) | Constructed (our generator, `trace_loader.py:265`) | `trace_length` sign set by injector crossover; pattern tools also set diversity/repeat | By construction |
| ToolBench | `_inject_anomaly` at 10% on real ToolBench traces (`domain_loader.py:202-242`) | Executed traces, constructed labels | `trace_length` sign set by injector crossover (short traces → longer) | By construction |
| ToolACE | `_inject_api_misuse`, length-preserving (`domain_loader.py:302-369, 423-470`) | Executed or curated traces, constructed labels | `trace_length` d ≈ 0 by design. The docstring (`domain_loader.py:431-435`) says the injection was chosen so that "trace_length polarity aligns with Synthetic". Wrong-function swaps move `low_diversity` and `repeat_rate`. | By construction, and the choice was made with the target polarity in mind |
| API-Bank | `_inject_api_misuse` (`domain_loader.py:372-421`) | Curated API sequences, constructed labels | Same as ToolACE | By construction |
| InjecAgent | Attack traces = user tool + attacker tools; benign traces = fixed 2 actions `user_tool`, `{user_tool}_response` (`trace_loader.py:1025-1048`) | Constructed, no agent run. 17 user cases (`data/injecagent/data/user_cases.jsonl`). Both `test_cases_*_base` and `*_enhanced` files are read. | Benign length is always 2, so all three structural features are fixed for the negative class | By construction. Remove from the main analysis. |
| AgentHarm | Trace = the gold `target_functions` list (`trace_loader.py:596-614`); label = harmful vs benign prompt | Constructed (gold tool lists, not executions) | Harmful and benign pairs share tool lists, so d ≈ 0 by construction | Not an execution dataset. Remove from the main analysis. |
| TRAIL | `n_errors / n_actions > 0.3` (`domain_loader.py:289-294`) | Executed (annotated GAIA/SWE runs) | `trace_length` is the denominator of the label. With few annotated errors, longer traces fall under the threshold, so `d_length < 0` is built in. | Length by construction. Keep the other features. |
| AgentDojo (old loader) | `label=0 if security else 1` (`trace_loader.py:1183`) over all 16 attack types plus `none` | Executed | Upstream `security: true` = the injection succeeded, verified on `important_instructions/injection_task_0.json`, where the agent sent $50 to the attacker. `none/none.json` also has `security: true`. So the old "anomalous" class = attacked and resisted, and the old "normal" class = clean runs plus successful attacks. | Mislabeled (a mixture). Fix per the prereg rule. |
| AgentDojo (prereg rule) | One attack (`important_instructions`): positive = `security: true`, negative = `security: false`, `none` excluded | Executed | gpt-4o: 629 runs, 47.7% `security: true`. Mean tool calls 5.15 (success) vs 3.93 (resisted), because a successful attack adds the injection task's calls. `trace_length` sign is set by the label concept. | Length by construction. Keep the other features. |
| StableToolBench | `win=False → anomalous` (`trace_loader.py:1607-1617`) | Executed (chatgpt_cot) | In all 765 files, `win` equals `answer_generation.valid_data` equals "a final answer exists" (468 true, 297 false). Mean `query_count` is 4.00 (win) vs 5.72 (lose). The label means "did not produce an answer", not the GPT-4 ToolEval judgment that `submission/sections/datasets.tex:16` describes. | Length tied to the termination label. Fix the paper text. |
| SWE-agent | `0 if solved else 1` (`domain_loader.py:134-143`), with a `MIN_TRACE_LEN..MAX_TRACE_LEN` selection filter, first 5,000 streamed rows | Executed | Unresolved runs tend to hit step limits (long). The upper length cut removes the longest failures. This is selection, not construction. | Plausible tie, not by construction. Keep, and report a sensitivity analysis without the length cap. |
| ATBench | Dataset `label` (`trace_loader.py:676-725`, HF `AI45Research/ATBench`) | **Constructed.** The dataset card describes "a taxonomy-guided data generation engine" with "risk injection" and "tool call simulation, tool response simulation, and agent response generation". Apache-2.0. | None in our code. The risk is injected during generation, so the label is a design choice of the generator. | Simulated, not executed. Not organic. |
| R-Judge | Dataset `label` in {"1","unsafe"} (`trace_loader.py:840-870`) | Curated: "569 records of multi-turn agent interaction", with labels from "the consensus of human annotators". The README does not say whether the records come from real runs. From memory, the R-Judge paper sources many records from ToolEmu emulated trajectories; check this in arXiv 2401.10019 before writing it in the paper. **No license in the repo.** | None in our code | Human-labeled. Origin probably emulated. Cannot be redistributed. |

## Consequences for the analysis

- **Executed traces with labels the dataset defines: only SWE-agent, TRAIL, AgentDojo and
  StableToolBench.** That leaves 4 domains and 12 ordered pairs. ATBench is simulated, and R-Judge is
  curated and probably emulated. Of the four, three have `trace_length` tied to the label, so a
  length-free "organic" headline rests on 4 domains. The pre-registered estimand is a
  per-target flip effect. A sign-flip test over 4 target-level effects has 2^4 = 16 arrangements, so
  its smallest one-sided p is 1/16 = 0.0625. **On executed-organic data alone, a domain-level test
  cannot reach p < 0.05, however strong the effect.** The prereg has to choose one of the following
  before any rerun: (i) a feature-level design across the 3 length-free structural features plus the
  3 argument features, with domains as clusters; (ii) report effect sizes with exact permutation p and
  no significance claim; (iii) make the label-audit branch the default.
- **Datasets where only `trace_length` is tied:** TRAIL, AgentDojo (prereg rule) and StableToolBench.
  These keep `low_diversity`, `repeat_rate` and the argument features.
- **Injected datasets** (Synthetic, ToolBench, ToolACE, API-Bank) carry injector-determined signs on
  every structural feature. They belong in a separate "injected" analysis, never in the organic
  headline.
- **InjecAgent and AgentHarm** are not agent executions. They leave the main analysis.
- **The paper's "8 of 12 datasets have organic labels" is false** under any reading that requires an
  executed trace and a label the dataset itself defines.

## Licence terms to check before release (Phase 3 artifact)

- TRAIL embeds GAIA tasks, and GAIA asks that its content not be reshared.
- AgentHarm carries a canary string and use restrictions.
- R-Judge, ATBench, the nebius SWE-agent trajectories, ToolBench (RapidAPI-derived), ToolACE,
  API-Bank, StableToolBench and AgentDojo each need their terms read.

MIT can cover only our code. Data ships as download scripts.
