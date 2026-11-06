# Radial Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline; one fresh final review. Steps use checkbox syntax.

**Goal:** Deliver the approved four-stage hand-positioned radial Colab.
**Architecture:** Pure radial design/fitter, bounded-memory data preparation and
stage runner, plots and staged Colab. Existing trajectory experiments untouched.
**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.
**Spec:** docs/superpowers/specs/2026-11-06-radial-ranking-design.md

## Global Constraints

- Same worktree/branch. No planning, actions, world model or historical replay.
- Goal50,42; wall32/door30;24rays15degrees;radii2..34step4;stages6,14,22,34.
- Identity then warm starts;500updates,batch16,bottleneck64,lr1e-4,ranking only.
- Fresh shared goal; physical bounds; fixed unseen rays/interpolation/offsets.
- Stream fp32 token shards on local disk; compact Drive artifacts with provenance.

## Review Focus

- Heldout coordinates/images must never enter optimizer inputs accidentally.
- Clipping must omit rather than collapse boundary points; stage counts/domain correct.
- Mixed/in-range/extrapolation comparisons must use both endpoints and preserve empty denominators.
- Cached/checkpoint corruption, changed seed/data/config and unsafe output aliases must reject reuse.
- Offset comparisons must use one matched cohort; fresh goal and identical tokens for all costs.

### Task 1: Radial design and ranking fitter

**Files:** radial_geometry.py, radial_training.py, tests/test_radial_ranking.py.
**Interfaces:** make_design(bounds), training_pairs(radii), fit(tokens,radii,goal,...).

- [x] Add RED tests for stage populations, clipping, fixed holdouts/offset cohorts, cross-ray pair rules and actual optimization with no probe inputs.
- [x] Implement deterministic coordinate design, heldout radii/rays/segments, and standalone ranking-only fitter retaining the adapter architecture.
- [x] Run focused tests GREEN; commit.

### Task 2: Cached preparation, stage runner and plotting

**Files:** radial_experiment.py, radial_plots.py; extend radial tests.
**Interfaces:** prepare(output,cache,env,encode,signature); stage(output,cache,index,device); renderer/encoder-only CLI.

- [x] Write RED integration tests for fresh reference, source/output isolation, cache/checkpoint hash rejection, cumulative warm starts and no heldout optimizer exposure.
- [x] Implement streaming cache, immutable config, fresh goal, train-only loading, stage lineage and numeric summaries stratified by radius/exposure; plot after each stage.
- [x] Run focused tests GREEN; execute all four stages/plots on synthetic tokens and real renderer design; commit.

### Task 3: Staged GPU Colab and publication

**Files:** notebooks/radial_ranking_colab.ipynb, docs/radial_ranking.md.
**Interfaces:** pinned setup -> prepare once -> four stage calls and immediate figures -> final comparisons.

- [x] Build notebook with no prior-experiment prerequisites; compile and execute result/display cells on synthetic workflow.
- [x] Run full suite; commit; one fresh final review; one Important/Critical RED→GREEN fix pass and suite if required.
- [x] Pin tested production code; push same branch; verify remote HEAD/clean status; mark actual GPU results pending.
