# Adjacent Band Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare three frozen planning costs on 20 identical starts in x [40,54], y [22,30].

**Architecture:** Add a separate evaluation runner that references the original
and clean artifacts, generates one fixed manifest, and reuses PilotRuntime and
run_stage in distinct condition directories. Provide a standalone Colab.

**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.

**Spec:** docs/superpowers/specs/2026-11-05-adjacent-band-evaluation-design.md

## Global Constraints

- Keep the existing worktree and branch; source pilot_v1 and clean_pair_v1 are read-only.
- Exactly 20 adjacent starts and 60 evaluation episodes; no training or collection.
- Same x [40,54], y [22,30], goal (50,42), layout (32,30), planner and CEM seeds across conditions.
- Use saved clean goal tokens; frozen source adapter runtime IDs remain distinct from evaluation IDs.
- Raw competence >=18/20 and adapter retention <=2 paired losses; report inconclusive transfer if raw fails.

## Review Focus

- Resolved output paths or aliases beneath sources are rejected before writes.
- Source or output manifests, runtime IDs, adapter hashes and goal tokens cannot change on resume.
- Wrong source models and mixed condition/case records fail before simulator actions or reporting.
- Interrupted runs retain journal actions and resume only complete cases, with independent conditions.
- The report handles single-band cases, ceiling success and offsetting wins/losses without misleading claims.

### Task 1: Three-condition adjacent evaluation and Colab

**Files:**
- Create: src/mylonas_ijepa_wm/curriculum_adjacent_band.py
- Create: tests/test_curriculum_adjacent_band.py
- Create: notebooks/curriculum_adjacent_band_colab.ipynb
- Create: docs/curriculum_adjacent_band.md
- Modify: .gitignore

**Interfaces:**
- Produces prepare_evaluation(parent, clean, output), evaluate_condition(parent, clean, output, condition, runtime), make_report(parent, clean, output, render=True).
- CLI stages prepare/evaluate/report; evaluate selects raw/original/clean. Source validation uses existing clean-ablation provenance checks and loads frozen adapters with their training runtime IDs.

- [x] Write synthetic tests for deterministic 20-case geometry, immutable/idempotent prepare, path aliases, changed source rejection, all three condition records and paired reporting, checkpoint provenance, interrupted journal/resume and mixed cases.
- [x] Run pytest -q tests/test_curriculum_adjacent_band.py; expect failure because the runner is absent.
- [x] Implement the runner, shared saved goal and source/runtime validation; add report and trajectory figures.
- [x] Build Colab with original Drive sources, child adjacent_band_v1 and raw/original/clean evaluation, then display report and figures.
- [x] Verify focused tests, notebook compilation, synthetic CLI prepare/report, figure inspection and full pytest suite.
- [x] Commit implementation, request one fresh final review, and fix important findings with regression tests.
- [x] Push existing curriculum-cost-adapter branch and give direct Colab link; state live results remain pending.
