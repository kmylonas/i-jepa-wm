# Multi-goal Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline, one fresh final review. Steps use checkboxes.

**Goal:** Evaluate eight equal-distance goal shifts with a frozen adapter.
**Architecture:** Generalize frozen evaluator/plots and add a separate pinned Colab.
**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.
**Spec:** docs/superpowers/specs/2026-11-06-radial-goal-sweep-design.md

## Global Constraints

- Same branch/worktree; no fitting/recalibration/source changes.
- Original50,42 plus8shifts sqrt32 at45degree increments, same stage4 default.
- One common heldout ball-position cohort across all goals, empty=N/A.
- More references than encoder batch size must work without re-encoding.
- Old notebook/pin intact; new radial_goal_sweep_v1 output.

## Review Focus

- Multi-batch goal references and token/query order cannot mix or duplicate.
- All-goal cohort and image-reference exclusion, including no-data cases.
- Dynamic artifact count, changed-goal/order guards and figure recovery.
- Equal displacement/default safety, no optimizer or calibration changes.
- Legacy one-goal callers and published notebook remain valid.

### Task 1: Generalized evaluator and summary

**Files:** radial_goal_transfer.py, radial_goal_transfer_plots.py, tests/test_radial_goal_sweep.py.
**Interfaces:** existing evaluate/load_report accept [x,y] or [[x,y],...]; CLI repeated --new-goal.

- [ ] RED:9reference goals with batch8 and unique coordinate encodes, exact source/adapter integrity, common counts, dynamic files/reuse, invalid goal lists and empty summary plots.
- [ ] Implement normalized goal lists, buffered reference scoring, dynamic hashes/recovery and summary plots.
- [ ] Focused new+legacy tests GREEN, synthetic evaluator/report CLI/figures; commit.

### Task 2: Sweep notebook and publication

**Files:** notebooks/radial_goal_sweep_colab.ipynb, docs/radial_goal_sweep.md.
**Interfaces:** pinned setup -> repeated-goal evaluate -> summary and per-goal heatmaps.

- [ ] Compile code cells and execute result/summary cells against synthetic output; verify old notebooks unchanged.
- [ ] Full suite, commit, fresh final review and one Important/Critical RED->GREEN fix pass if needed.
- [ ] Pin verified code, push same branch, verify remote HEAD/clean state, retain worktree and remove only plan scratch.
