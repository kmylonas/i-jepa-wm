# Manual Multi-goal Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a separate manual multi-goal ranking Colab and compare ordinal transfer to the existing distance-fitted model.

**Architecture:** Reuse the unchanged multi-goal preparation in a new run's data directory. New modules own ranking triples/fitting/metrics, signed orchestration and plotting. All previous notebooks and source runs remain intact.

**Tech Stack:** Python, NumPy, PyTorch, Matplotlib, Colab.

**Spec:** `docs/superpowers/specs/2026-11-06-multigoal-ranking-design.md`

## Global Constraints

- Existing worktree/branch and final R34 region; same15 reserved goals and image exclusions.
- Manual comparisons only; no planning. Ranking softplus margin0.2, physical gap0.5, one training-only raw scale.
- Two identity adapters seed0, bottleneck64, lr1e-4, 2,000 updates,32 triples; gradients to G,F,N.
- Heldout ranking/uphill/tie metrics; no distance MAE/calibration of ranking scores.
- Frozen successful multi-goal distance comparator, source/checkpoint verification.
- Full-right x35..60/y4..60 maps, new timestamp outputs and pinned notebook.

## Review Focus

- Reversed/tied physical gaps and reserved images cannot enter offered triples (task1).
- Reference gradients and reference-balanced sampling must be real (task1).
- Historical code mismatch is allowed only for the frozen read-only distance reference (task2).
- Corruption/changed settings reject reuse; reports survive lost local cache (task2).
- All16 goals display, seen control excluded from aggregates, arbitrary cost units explicit (task2).

### Task 1: Ranking populations, loss and ordinal diagnostics

**Files:** New `src/mylonas_ijepa_wm/multigoal_ranking.py`, `tests/test_multigoal_ranking.py`.

**Interfaces:** `triples(positions,reference_ids)` returns [G,F,N] tensor; `triplet_costs(adapter,tokens,triples,device='cpu',batch_size=16)` returns far/near costs; `fit(tokens,triples,scale,**fitting)` returns adapter/report; `ordinal_rows(...)` returns common ranking/directional rows.

- [ ] Write tests for valid triplet geometry/roles, reference gradients, reference-balanced sampling/exposure, loss improvement, scale-independent ordinal counts and empty groups.
- [ ] Run tests RED (missing module), implement, verify GREEN, commit.

### Task 2: Reproducible experiment and Colab

**Files:** New `src/mylonas_ijepa_wm/multigoal_ranking_experiment.py`, `src/mylonas_ijepa_wm/multigoal_ranking_plots.py`, `notebooks/radial_multigoal_ranking_colab.ipynb`, `docs/multigoal_ranking.md`; extend tests.

**Interfaces:** `prepare(source,baseline,output,cache,env,encode,execution,*,fitting=None,new_goals=DEFAULT_GOALS,batch_size=8)`, `train(output,cache,branch,device='cpu')`, `evaluate(output,cache,device='cpu',render=True)`, `report(output,render=True)` and CLI prepare/train/evaluate/report.

- [ ] Add failing synthetic complete-flow, frozen-baseline/code-signature, reuse/corruption, cache deletion, figures and notebook-report tests.
- [ ] Implement signed parent ranking run over unchanged shared preparation, both ranking arms, frozen multi-distance comparator and ordinal evaluation.
- [ ] Add separate full-right plots/notebook/docs; verify actual report cells and16-goal CLI workflow.
- [ ] Run focused and full suite, inspect figures, fresh final review and fix substantive findings with regression tests. Pin notebook to verified source, publish existing branch, verify remote and clean state.

User authorized this experiment and the established same-worktree Colab workflow. Execute natively without another permission round; retain prior work.
