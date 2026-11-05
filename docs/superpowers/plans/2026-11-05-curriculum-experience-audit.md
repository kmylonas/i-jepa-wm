# Curriculum Experience Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution with one fresh final review.

**Goal:** Audit saved collection trajectories and training labels without training or environment interaction.

**Architecture:** A CPU analysis module validates saved artifacts and computes
geometry/pair statistics using the exact existing fitter filter. A figure module
renders the derived report; a standalone Colab notebook runs analysis against Drive.

**Tech Stack:** Python, NumPy, PyTorch CPU, matplotlib, Colab, pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-curriculum-experience-audit-design.md`

## Global Constraints

- Preserve all original pilot files and runtime identity; write derived outputs only.
- No encoder/world-model load, simulator execution, adapter updates or new rollouts.
- Revisit radius 0.5, pair gap >=2, pair separation >=2; default diagnostic progress tolerance 0.25.
- Match uniform-episode/uniform-pair sampling; distinguish pooled pair fractions.
- Goal embeddings unavailable: do not invent exact adapter-cost ordering statistics.

## Review Focus

- Zero motion/short paths must produce defined JSON values, not NaN/division errors.
- Successful but inefficient episodes and later-farther pairs must be detected.
- Long episodes must not dominate estimates of the trainer's sampled labels.
- Mixed runtime/case/timing or changed saved filter counts must fail explicitly.
- Macro-level evaluation paths must retain their actual timestamps and be labeled coarse.

### Task 1: Validated CPU audit and metrics

**Files:** `src/mylonas_ijepa_wm/audit_curriculum_experience.py`,
`tests/test_curriculum_experience_audit.py`, `.gitignore`.

**Interfaces:** `analyze_episode(raw, success_radius=4.5, progress_tolerance=.25)`;
`summarize_pair_quality(rows, field)`; `run_audit(pilot_dir, run_seed=0, progress_tolerance=.25)`.

- [x] Write failing tests for straight/wandering trajectories, loop pair filtering,
  actual timestamps, zero motion, matched sampler weighting and provenance validation.
- [x] Implement metrics, loading, saved-fitting validation, evaluation comparisons,
  derived summary/CSV/traces and CLI; test synthetic audit without input mutation.
- [x] Verify focused tests and commit.

### Task 2: Figures and standalone Colab delivery

**Files:** `src/mylonas_ijepa_wm/curriculum_audit_figures.py`,
`notebooks/curriculum_experience_audit_colab.ipynb`, `docs/curriculum_experience_audit.md`.

**Interfaces:** `render_audit(report, traces, examples, output_dir)` consumes only
derived arrays plus a few explicitly selected raw-image episode paths.

- [x] Extend the synthetic workflow test to require readable figures, review
  representative plots, implement explanatory notebook and report text.
- [ ] Compile notebook cells, smoke the CLI, run the full project suite, and
  request one fresh whole-change review; fix important findings with regression tests.
- [ ] Commit and push to the existing feature branch; provide direct Colab link
  and exact cells to run, explicitly distinguishing readiness from actual data findings.
