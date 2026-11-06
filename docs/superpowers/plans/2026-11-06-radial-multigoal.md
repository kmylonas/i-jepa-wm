# Radial Multi-goal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a separate Colab comparing matched single-goal and multi-goal distance adapters on reserved near and distant goals.

**Architecture:** A new fitting module owns pairs and regression; an experiment module owns signed preparation/checkpoints/evaluation; a plotting module and notebook expose the results. Reuse the existing source loader, adapter, geometry, metrics and artifact utilities without changing their semantics.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, Colab.

**Spec:** `docs/superpowers/specs/2026-11-06-radial-multigoal-design.md`

## Global Constraints

- Existing worktree and branch; earlier notebooks/results/source weights unchanged.
- Same final R34 region, frozen encoder/renderer, fp32 batch 8 encoding.
- Two identity adapters, seed 0, 2,000 updates, 32 pairs, lr 1e-4, bottleneck 64; target d/34, no anchoring.
- Reserve all 15 shifted goal coordinates/rendered images from both fitting endpoints.
- Shared heldout query cohort and raw scalar fitted on training pairs only.
- Full-right x35..60/y4..60 heatmaps, separate timestamped outputs and pinned Colab.

## Review Focus

- A withheld goal rasterizing like a training point must exclude that node before either arm fits (task 1).
- Distinct training coordinates with identical pixels must not produce contradictory positive-distance pairs (task 1).
- Lost local cache must still permit numerical report and missing plot regeneration (task 2).
- Changed pairs/checkpoints/source/settings must block completed-result reuse (task 2).
- Original goal is a seen reference and must not count in withheld-goal aggregates (task 3).

### Task 1: Pair populations and matched regression

**Files:** Create `src/mylonas_ijepa_wm/radial_multigoal.py`, `tests/test_radial_multigoal.py`.

**Interfaces:** `make_split(points,candidate_ids,goal_ids,frame_hashes)` returns train IDs, heldout mask and pair populations; `fit(tokens,positions,pairs,seed=0,updates=2000,batch_size=32,bottleneck=64,lr=1e-4,device='cpu',progress=None)` returns adapter/training report; `pair_costs(adapter,tokens,pairs,device='cpu',batch_size=16)` supports adapter=None for raw MSE.

- [ ] Write tests: aliases/withheld endpoints absent, all multi pairs within34 and unique, single pairs reference original; two-ended cost gradients/linear-distance target; fit reduces synthetic error and reports exactly2*batch*updates endpoint exposures; invalid pair populations rejected.
- [ ] Run focused tests; expected missing-module assertion failures.
- [ ] Implement split, chunked costs, train-only fixed probe logging and fitting.
- [ ] Run focused tests; expected all pass. Commit fitting module/tests.

### Task 2: Reproducible run and common evaluation

**Files:** Create `src/mylonas_ijepa_wm/radial_multigoal_experiment.py`; extend new tests.

**Interfaces:** `prepare(source_root,output,cache,env,encode,execution,*,new_goals=DEFAULT_GOALS,fitting=None,batch_size=8)`; `train(output,cache,branch,device='cpu')`; `evaluate(output,cache,device='cpu',render=True)`; `report(output,render=True)`. Branch names single/multi. CLI prepare/train/evaluate/report.

- [ ] Add failing synthetic tests for complete prepare/two-arm/evaluate flow, unchanged source, shared masks and metrics, training-only raw scalar, source consistency, completed reuse without refit, corruption and report after cache deletion.
- [ ] Implement signed persistent inputs/local tokens, both checkpoints, frozen legacy comparator, common metrics and CLI, consuming task1 interfaces and existing transfer measure().
- [ ] Run focused tests; expected all pass. Commit experiment module/tests.

### Task 3: Figures, Colab and publication

**Files:** Create `src/mylonas_ijepa_wm/radial_multigoal_plots.py`, `notebooks/radial_multigoal_colab.ipynb`, `docs/radial_multigoal.md`; extend tests.

**Interfaces:** `plot_run(output,report)` writes coverage/loss/summary and16 goal heatmaps PNG/PDF. Notebook runs CLI in separate processes and displays verified reports.

- [ ] Add failing figure-regeneration test and notebook command/report smoke; use actual report data with synthetic encoding.
- [ ] Implement figures/notebook/docs and verify all actual code cells compile and report cells execute with synthetic results.
- [ ] Run focused tests and full repository suite, inspect representative rendered plots. Expected all pass, readable figures and all goal IDs shown.
- [ ] Fresh final review against spec/plan, resolve substantive findings with regression tests. Commit production code, pin notebook to that source commit, publish same branch and verify remote HEAD. Actual GPU results remain pending user run.

## Execution

User already approved this experiment and the existing same-worktree Colab workflow. Continue implementation natively in this session without another permission round. A builder check found no material unanswered questions.
