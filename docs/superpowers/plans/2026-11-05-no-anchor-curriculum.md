# No-anchor Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline; one fresh final reviewer. Steps use checkbox syntax.

**Goal:** Deliver a reproducible Colab that retrains both initial adapters and all curriculum updates using ranking loss alone.

**Architecture:** Preserve existing anchored defaults. Add a no-anchor configuration variant, separate initial-fit checkpoint provenance, and a standalone notebook. Reuse the existing sequential runner and fixed-case checks.

**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.

**Spec:** docs/superpowers/specs/2026-11-05-no-anchor-curriculum-design.md

## Global Constraints

- Same worktree/branch; output multiband_no_anchor_v1; source and old results read-only.
- Both recipes retrain from identity on saved near replay; all fits anchor_weight=0.
- Keep shared goal, cases, pair filters, architecture, 500 updates, batch16, lr1e-4 and fresh Adam.
- Preserve the established three-band loop and budget <=840 episodes.
- Print ranking-only loss explicitly, full-range maps and vertical profiles; compare historical anchored results with unavailable cells marked.
- Existing anchored protocol/configuration schemas and fitting defaults remain compatible.

## Review Focus

- Initial weights must come from identity rather than the old anchored source adapters.
- A partial or modified initial checkpoint must not permit new episode actions or be shown as a completed version.
- No-anchor fitting must exclude the anchor term itself, while preserving pair RNG and ranking normalization.
- The new output cannot alias the prior anchored run used for comparison, including symlinks.
- Historical comparisons require identical cases/goal/planner, and incomplete branches remain readable.

### Task 1: Ranking-only fitter

**Files:** curriculum_replay.py; tests/test_curriculum_replay.py.
**Interfaces:** fit_adapter(..., anchor_weight=0) optimizes rank_loss only, logs anchor_loss=0 and anchoring_enabled=False. Positive-weight behavior is unchanged.

- [ ] Write a ranking-only regression checking loss equality, zero disabled anchor, deterministic pair sampling and actual ranking improvement.
- [ ] Run pytest tests/test_curriculum_replay.py; expected failure on missing disabled-anchor metadata.
- [ ] Skip anchor forward/loss computations at zero weight; reserve RNG draw and preserve normal scale. Keep nonfinite safeguards.
- [ ] Run focused tests, expect pass; commit.

### Task 2: Fresh initial fits and immutable experiment variant

**Files:** curriculum_multiband_protocol.py, curriculum_multiband.py, new curriculum_initial_fit.py; extend tests/test_curriculum_multiband.py.
**Interfaces:** prepare(..., without_anchor=False); CLI prepare --without-anchor; config variant selects anchor_weight=0 and fresh initial fits. Initial checkpoint v0 is saved inside each recipe with inputs/goal/replay/seed/config/checkpoint/training hashes. Later references resolve to that v0.

- [ ] Write failing tests for zero-anchor manifest, unchanged deterministic cases, config mismatch rejection, identity initial fits instead of source adapters, separate original/clean filters, full cumulative chain, cached initial fit reuse, and corrupt/partial initial provenance before actions.
- [ ] Run focused tests; expect missing variant errors.
- [ ] Implement configuration factory with unchanged default schema, initial fitter, reference validation and partial report handling. Reuse existing branch stages.
- [ ] Run focused tests and source immutability checks; expect pass; commit.

### Task 3: Colab, profiles and historical comparison

**Files:** new curriculum_no_anchor_colab.ipynb; new docs/curriculum_no_anchor.md; optional reporting helper in curriculum_initial_fit.py and tests.
**Interfaces:** Notebook reuses staged GPU setup, prepares --without-anchor, displays per-update maps/losses, runs both recipes independently, prints final matrices and exact-case comparisons against multiband_curriculum_v1. Embedded CPU profile plotting uses saved maps with a global non-saturating scale.

- [ ] Add regression for comparison case mismatch/unavailable prior cells/output alias if helper is used; run RED then implement.
- [ ] Build notebook with explicit identity near fitting, ranking-only objective, preserved goal, separate output, source pin and all profiles.
- [ ] Verify notebook code compilation and synthetic staged run/report. Run full suite; expect pass.
- [ ] Commit and request one fresh whole-change review. Fix Important/Critical findings with regression tests and full suite.
- [ ] Pin runtime, push same branch, verify remote HEAD, provide Colab link. Mark GPU outcomes pending.
