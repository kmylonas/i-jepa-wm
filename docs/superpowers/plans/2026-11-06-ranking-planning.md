# Ranking Adapter Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish a paired multi-goal world-model planning comparison, frozen ranking adapter versus raw MSE.

**Architecture:** A new protocol/loader owns frozen artifacts and deterministic cases; a runtime/runner reuses PilotRuntime and plan_proprio with per-goal explicit scoring; report/plots and Colab expose paired outcomes. Existing fitting and planning modules retain their semantics.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, Colab.

**Spec:** `docs/superpowers/specs/2026-11-06-ranking-planning-design.md`

## Global Constraints

- Existing worktree/branch, all prior artifacts unchanged; no fitting.
- Six goals, eight starts each,48 paired cases96 episodes; starts/goals inside originalR34 and right wall side.
- Existing world model/encoder/renderer pins and PLANNING settings, fp32.
- Per-goal shared references, same reset/CEM seed per pair, raw histories.
- Main success distance<4.5; preserve all cases and report region excursions.
- Resumable signed records/journals and model-free report; pinned separate Colab.

## Review Focus

- Historical code changes allowed only in read-only frozen source loading (task1).
- Hard-coded original goal must never leak into shifted-goal costs (task2).
- Raw/adapted seeds and original world-model inputs must match (task2).
- Interrupted attempts, corrupted companion files and config changes cannot silently reuse results (task2).
- Both-success efficiency subsets and out-of-region successes must remain explicit (task3).

### Task 1: Frozen source, manifest and paired metrics

**Files:** New `src/mylonas_ijepa_wm/ranking_planning_protocol.py`, `tests/test_ranking_planning.py`.

**Interfaces:** `load_ranking_source(root)` returns frozen adapter/config/design/snapshot; `make_manifest(design,starts_per_goal=8)`; `summarize(raw,adapted)`.

- [ ] Write failing tests for old-code source loading/corruption, six-goal domain/distance/holdout pairing, summaries/efficiency/empty subsets.
- [ ] Implement, verify GREEN, commit.

### Task 2: Per-goal runtime and resumable evaluation

**Files:** New `src/mylonas_ijepa_wm/ranking_planning.py`; extend tests.

**Interfaces:** `prepare(source,output)`, `prepare_goals(source,output,runtime)`, `evaluate_condition(source,output,condition,runtime,limit=None)`, `report(source,output,render=True)`; `PlanningRuntime(PilotRuntime).run_case(case,target,metric=None)`; CLI prepare/goals/evaluate/report.

- [ ] Add failing tiny CEM/model tests proving shifted references, identical seeded action proposals and raw histories, and signed synthetic orchestration interruption/resume/companions.
- [ ] Implement explicit terminal scoring, primitive capture and journal, immutable goal references and episode hashes, source/runtime validation.
- [ ] Verify focused tests GREEN; commit.

### Task 3: Figures and runnable notebook

**Files:** New `src/mylonas_ijepa_wm/ranking_planning_plots.py`, `notebooks/ranking_planning_colab.ipynb`, `docs/ranking_planning.md`; extend tests.

- [ ] Add failing plot/model-free-report and actual notebook reporting/command tests.
- [ ] Implement coverage, paired per-goal trajectories, outcome/efficiency figures; notebook initial paired timing then reuse completed cases, all reports.
- [ ] Verify synthetic full48-case flow and figures, full suite, fresh final review. Fix substantive findings via regression tests. Pin notebook, publish same branch and verify remote/clean tree. Actual GPU planning remains pending user run.

User requested this benchmark and the established same-worktree Colab workflow. Execute natively without another approval round.
