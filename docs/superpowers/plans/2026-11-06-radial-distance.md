# Radial Distance Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline; one fresh final review. Steps use checkbox syntax.

**Goal:** Deliver a separate cumulative distance-regression Colab.
**Architecture:** Add a training-only regression fitter and distance diagnostics,
extend the existing signed runner with explicit objective dispatch, then expose
a separate pinned notebook. Existing ranking behavior is the default.
**Tech Stack:** PyTorch, NumPy, matplotlib, pytest, Colab.
**Spec:** docs/superpowers/specs/2026-11-06-radial-distance-design.md

## Global Constraints

- Same worktree/branch, same24rays/geometry/stages6,14,22,34/frozen encoder.
- Target C=r/34, fixedscale34, mean squared residual; no anchor/ranking term.
- Identity then cumulative warm starts,500updates,batch32,Adam1e-4,bottleneck64.
- Separate notebook/output/cache paths; retain existing ranking notebook/pin.
- Actual GPU results pending; no planning/old replay/world-model loads.

## Review Focus

- Target must be distance, not squared distance or stage-normalized distance.
- Future/probe points cannot reach fitting or calibrate the raw baseline.
- Legacy ranking config/defaults and published notebook remain intact.
- Empty extrapolation denominators, duplicate IDs and image-overlap exclusions.
- Objective/scale mismatch, alias/corrupt artifacts must reject before fitting.

### Task 1: Regression fitter and numeric diagnostics

**Files:** radial_distance.py, tests/test_radial_distance.py.
**Interfaces:** fit(tokens,radii,goal,...); distance_summary(design,raw,adapted,report).

- [x] RED: targetr/34 and fixedscale, optimization/warm starts/counts, training-only raw calibration, heldout-region empty behavior.
- [x] Implement fitter and report diagnostics; no heldout tensors in fitting API.
- [x] Focused tests GREEN; commit.

### Task 2: Signed runner dispatch and plots

**Files:** radial_experiment.py, radial_plots.py, radial_distance_plots.py; tests.
**Interfaces:** prepare(...,objective='ranking'); stage infers objective from signed config.

- [x] RED: explicit objective signature, four cumulative stages, changed-kind refusal, fitted-input counts, final outside N/A and legacy ranking compatibility.
- [x] Implement separate kind, fixedscale, defaultbatch32, dispatch and correctly labeled regression/calibration figures.
- [x] Focused tests GREEN; synthetic four stages/all plots; commit.

### Task 3: Separate Colab and publication

**Files:** notebooks/radial_distance_colab.ipynb, docs/radial_distance.md.
**Interfaces:** separate setup -> prepare --objective distance -> stage displays -> final table.

- [x] Compile five code cells; execute stage and summary cells on synthetic outputs; verify legacy notebook unchanged.
- [x] Full suite; commit; one fresh final review and one Important/Critical RED→GREEN fix pass if needed.
- [x] Pin tested source; publish same branch; verify remote HEAD, clean status; retain worktree, remove only current plan scratch.
