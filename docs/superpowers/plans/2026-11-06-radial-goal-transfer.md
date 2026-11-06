# Radial Goal Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline, one fresh final review. Steps use checkboxes.

**Goal:** Evaluate a frozen Euclidean-distance adapter on a changed goal.
**Architecture:** Read-only historical checkpoint verifier, batched fresh grid and
goal scoring, common-cohort metrics/plots and separate Colab.
**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.
**Spec:** docs/superpowers/specs/2026-11-06-radial-goal-transfer-design.md

## Global Constraints

- Same worktree/branch; existing source/results/notebooks untouched.
- Final distance adapter by default; goals50,42 and46,38; no fitting.
- Fixedscale34 and saved raw calibration, fresh original-goal control.
- Full-right grid; common original-region/known-distance metrics.
- Historical source cache not required; whole checkpoint chain verified.

## Review Focus

- Correct checkpoint kind, lineage, target scale and historical-source handling.
- New goal reference must move ball; encoder images contain no goal marker.
- Common-region distance/exposure/edge masks and empty denominators.
- No recalibration/refitting, source writes or mutable adapter parameters.
- Output alias/corruption rejection and notebook pin/reuse behavior.

### Task 1: Frozen transfer evaluation and plots

**Files:** radial_goal_transfer.py, radial_goal_transfer_plots.py, tests/test_radial_goal_transfer.py.
**Interfaces:** load_source(root,stage); evaluate(source,output,stage,new_goal,env,encode,signature,device).

- [ ] RED tests: wrong source kind/corrupt chain, actual new reference and selfcost, frozen/source integrity, matched masks, changed goal/output guards, completed reuse.
- [ ] Implement read-only loader, fresh scoring and correctly labeled metrics/plots/CLI.
- [ ] Focused tests GREEN, synthetic evaluation/figures; commit.

### Task 2: Colab and publication

**Files:** notebooks/radial_goal_transfer_colab.ipynb, docs/radial_goal_transfer.md.
**Interfaces:** pinned setup -> evaluate -> display goal images/heatmaps/tables.

- [ ] Compile and execute notebook result cells on synthetic artifacts; verify legacy notebooks unchanged.
- [ ] Full suite, commit, fresh final review and one Important/Critical RED->GREEN fix pass if required.
- [ ] Pin tested source, push existing branch, verify remote HEAD/clean state; retain worktree and remove only plan scratch.
