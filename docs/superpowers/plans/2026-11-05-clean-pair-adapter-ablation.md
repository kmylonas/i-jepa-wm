# Clean-Pair Adapter Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a second adapter from the original replay while admitting only temporal pairs whose later saved state is at least 0.5 units closer to the goal.

**Architecture:** Extend replay fitting with an optional oracle-progress margin whose default preserves the original path. Add a separate staged ablation runner and GPU Colab notebook that treat `pilot_v1` as immutable parent data and write a child runtime under `clean_pair_v1`.

**Tech Stack:** Python, NumPy, PyTorch, matplotlib, pytest, Colab.

**Spec:** `docs/superpowers/specs/2026-11-05-clean-pair-adapter-ablation-design.md`

## Global Constraints

- Parent `pilot_v1` is read-only; no collection, replay encoding, original adapter changes or parent stage writes.
- Run seed 0, margin `>= 0.5`, all original successful replay episodes, and the original 500-update fitting schedule are fixed.
- Sampling remains uniform episode then uniform retained pair.
- Child evaluation uses the exact parent evaluation cases and CEM seeds, but its own runtime ID and interaction journal.
- Persist the computed goal tokens and parent/child provenance; do not auto-run candidate replay diagnostics.

## Review Focus

- A margin of `None` must not inspect goal coordinates or alter original eligible-pair order, summaries, RNG draws or trained parameters.
- Exactly-at-0.5 pairs are included; nonfinite goals/margins and missing case goals fail explicitly.
- Parent and child runtime IDs legitimately differ, while all evaluation case payloads remain identical.
- Resume rejects changed parent replay, pair rule, adapter hash or child runtime rather than mixing results.
- A partially interrupted evaluation keeps its primitive actions in the child journal and resumes only complete cases.

---

### Task 1: Optional oracle-progress replay filter

**Files:**
- Modify: `src/mylonas_ijepa_wm/curriculum_replay.py`
- Modify: `tests/test_curriculum_replay.py`

**Interfaces:**
- Produces: `eligible_pairs(..., goal=None, min_goal_progress=None)` with the existing return type.
- Produces: `fit_adapter(..., min_goal_progress=None)` and training metadata describing the active pair rule and pre/post-filter counts.

- [ ] **Step 1: Write failing tests** proving default pairs and seeded parameters are unchanged, an inclusive 0.5 margin removes ambiguous/contradictory pairs, per-episode counts are saved, and invalid oracle inputs stop.
- [ ] **Step 2: Run** `pytest -q tests/test_curriculum_replay.py` and confirm the new assertions fail for missing arguments/metadata.
- [ ] **Step 3: Implement** the optional filter after existing revisit/gap/separation selection; access `episode["case"]["goal"]` only when the margin is active.
- [ ] **Step 4: Run** the focused replay tests and confirm all pass.
- [ ] **Step 5: Commit** the isolated replay-filter change.

### Task 2: Immutable-parent ablation runner and Colab workflow

**Files:**
- Create: `src/mylonas_ijepa_wm/curriculum_clean_pair_ablation.py`
- Create: `tests/test_curriculum_clean_pair_ablation.py`
- Create: `notebooks/curriculum_clean_pair_ablation_colab.ipynb`
- Create: `docs/curriculum_clean_pair_ablation.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces CLI stages `prepare`, `train`, `evaluate`, `report` with `--parent-dir`, `--output-dir`, and fixed run seed/margin provenance.
- Consumes parent baseline/collect/evaluate records and `replay/run_0/*.pt`; produces child manifest/config/runtime, saved goal tokens, adapter, evaluation records, report and figures.

- [ ] **Step 1: Write failing synthetic workflow tests** for parent completeness, replay hashes and no parent mutation; clean training provenance; cross-runtime paired reporting; exact cases; operational screen; and resume rejection after changed inputs.
- [ ] **Step 2: Run** the new test file and confirm it fails because the runner is absent.
- [ ] **Step 3: Implement** parent validation and atomic `prepare` metadata, replay loading, goal-token persistence, clean training, child evaluation and raw/original/clean paired reporting.
- [ ] **Step 4: Add** success, trajectory and training-loss figures plus plain-language report fields that distinguish avoiding degradation from improving far performance.
- [ ] **Step 5: Build** the standalone GPU notebook. It clones the feature branch, validates `pilot_v1`, uses `clean_pair_v1`, runs no collection/diagnostics, and displays pair counts, losses, paired wins/losses, distances, budgets and figures.
- [ ] **Step 6: Verify** notebook JSON/code compilation, a synthetic CLI workflow, parent-file hashes, focused tests, and the full project suite.
- [ ] **Step 7: Request** one fresh whole-change review; fix important findings with regression tests.
- [ ] **Step 8: Commit and push** the feature branch, then provide the direct Colab link and state that empirical clean-pair results remain pending until the notebook is run.

