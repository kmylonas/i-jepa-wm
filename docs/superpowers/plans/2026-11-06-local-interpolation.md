# Local Interpolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans inline; one fresh final review. Steps use checkbox syntax.

**Goal:** Deliver the approved read-only local-generalization Colab.
**Architecture:** Pure probe/sampling/metric helpers, standalone encoder audit,
and plotting notebook. Extend new grids while preserving legacy cache loading.
**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.
**Spec:** docs/superpowers/specs/2026-11-06-local-interpolation-design.md

## Global Constraints

- Same branch/worktree; existing sources read-only; no fits or episodes.
- Clean no-anchor v0, 50 sampled fitting pairs, 11 positions, offsets 0,+/-0.5,+/-1,+/-2.
- Real rendered images, same saved goal and frozen encoder; batch8 fp32.
- Exact sampled-pair reconstruction and coverage of all fitting inputs; ties explicit.
- New maps extend to physical right border; legacy caches remain readable.

## Review Focus

- Incorrect RNG reconstruction must not label eligible-only pairs as fitted.
- Coincident rendered frames or other fitted replay locations must not be called unseen.
- Goal-distance-increasing, obstructed or out-of-bounds segments must be excluded and counted.
- Existing source metadata and caches must not be rewritten by audit setup.
- Flat curves, empty eligible cohorts and missing checkpoints must be explained rather than reported as success.

### Task 1: Sampling, probe geometry and metrics

**Files:** new curriculum_local_geometry.py; tests/test_curriculum_local_geometry.py.
**Interfaces:** sampled_pairs(episodes,training), select_pairs(...), profiles(...), curve_metrics(...), coverage(...).

- [ ] Add behavioral tests for actual fit sampler parity, segment monotonicity/bounds, conservative coverage and flat/reversed/ridged curve metrics.
- [ ] Run focused tests RED (missing module); implement pure helpers, retaining RNG draw order and no cost-conditioned selection.
- [ ] Run focused tests GREEN and commit.

### Task 2: Read-only audit and full-right maps

**Files:** new curriculum_local_audit.py, curriculum_local_plots.py; modify curriculum_heatmaps.py; tests/test_curriculum_heatmaps.py and local geometry tests.
**Interfaces:** audit(curriculum, output, env, encode, ...), CLI constructs only WallAdapter/frozen encoder; plotting consumes report.json and probes.json.

- [ ] Add failing full-right/legacy cache tests and synthetic end-to-end audit tests checking immutable source hashes, source alias rejection, actual fresh scores, exposure and expanded domain.
- [ ] Implement dynamic grid shapes and explicit new-grid geometry metadata, legacy loading; implement streaming audit with checkpoint validation and replay/fresh endpoint comparison.
- [ ] Run focused tests GREEN; execute plot export on synthetic numeric results; commit.

### Task 3: GPU Colab and publication

**Files:** notebooks/curriculum_local_interpolation_colab.ipynb, docs/curriculum_local_interpolation.md.
**Interfaces:** pinned clone/renderer/encoder setup -> diagnostic CLI -> numeric summaries and PNG/PDF displays.

- [ ] Build standalone notebook using existing setup with timestamped separate output; compile all code cells and execute diagnostic/plot cells on synthetic fixtures.
- [ ] Run full suite; commit; one fresh whole-change review; fix Important findings with RED→GREEN tests and suite.
- [ ] Pin verified code, push existing branch, verify remote HEAD and clean worktree. Report actual GPU execution pending.
