# Radial Position Readout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline; one fresh final review. Steps use checkbox syntax.

**Goal:** Deliver a separate increasing-radius ball-position Colab.
**Architecture:** Spatial coordinate MLP/fitter and position diagnostics, signed
position-stage runner sharing frozen-feature preparation, own plots and notebook.
**Tech Stack:** PyTorch, NumPy, matplotlib, pytest, Colab.
**Spec:** docs/superpowers/specs/2026-11-06-radial-position-design.md

## Global Constraints

- Same worktree/branch, existing notebook/source pins unchanged.
- Frozen tokens only; shared D->32 projection then flattened spatial grid->64->2.
- Targets(xy-[50,42])/34; unbounded output;500updates,batch32,Adam1e-4.
- Same geometry/stages6,14,22,34; cumulative data, warm starts, seeds0..3.
- Separate artifacts; fitting never consumes probe coordinates/features or goal.
- Real GPU results pending; no planning/old replay/world-model checkpoint.

## Review Focus

- Spatial token order and x/y ordering must survive normalization and scoring.
- Oracle coordinates, future features and goal image must not leak into fitting.
- Outside/fixed-near metrics and position vs derived-distance distinctions.
- Independent model checkpoints, whole-chain corruption and unsafe alias checks.
- Existing ranking/distance defaults/pins intact; missing figures recover safely.

### Task 1: Position model, fitting and diagnostics

**Files:** radial_position.py, tests/test_radial_position.py.
**Interfaces:** PositionReadout; fit(tokens,positions,...); summarize_positions(...).

- [x] RED tests for spatial order, unbounded normalized output, actual learning/warmstarts, sample counts and correctly excluded heldout cohorts/empty bins.
- [x] Implement fixed-center/scale fitter and physical-unit metrics/baseline.
- [x] Focused tests GREEN; commit.

### Task 2: Signed stage runner and plots

**Files:** radial_experiment.py, radial_position_experiment.py, radial_position_plots.py; tests.
**Interfaces:** prepare(...objective='position'); shared stage dispatches validated position config.

- [x] RED tests for current-stage-only fitting, objective/scale guards, lineage, corrupted artifacts, goal exposure and completed-stage/figure recovery.
- [x] Implement checkpoint chain, streaming predictions, numeric/derived-distance diagnostics and all figures.
- [x] Run position+legacy tests; synthetic four CLI stages/plots; commit.

### Task 3: Separate Colab and publication

**Files:** notebooks/radial_position_colab.ipynb, docs/radial_position.md.
**Interfaces:** prepare --objective position -> four stages -> immediate plots and final error tables.

- [x] Compile cells and execute real result/summary cells against synthetic artifacts; verify old notebooks unchanged.
- [x] Full suite, commit; one fresh final review and one Important/Critical RED->GREEN fix pass if needed.
- [x] Pin verified source; push existing branch, verify remote HEAD/clean state; retain worktree and delete only plan scratch.
