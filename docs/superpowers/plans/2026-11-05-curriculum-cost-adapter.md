# Curriculum Cost Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a reproducible, resumable near-to-far cost-adaptation pilot.

**Architecture:** Add a shared residual token metric and temporal replay trainer.
Use small optional hooks in the existing proprioceptive planner for fixed
resets, primitive observation capture, and terminal cost substitution. A staged
runner owns manifests, provenance, gates, collection, training and reporting.

**Tech Stack:** Python, PyTorch, NumPy, pinned DotWall, pytest, Colab.

**Spec:** `docs/superpowers/specs/2026-11-05-curriculum-cost-adapter-design.md`

## Global Constraints

- Exact identity cost initially; frozen encoder/world model and original histories.
- Layout (32,30), goal (50,42), near [40,54]x[30,36], far [40,54]x[14,24].
- Twenty held-out cases per band, forty collection attempts per seed, three seeds.
- CEM 5/5 macros, 300 population, 30 elites, 10 iterations, cap 3.5, elite_mean.
- Primitive timestamps, partial-macro accounting, no evaluation replay in training.
- Fixed 500-step training schedule; record every builder default in spec.
- Resume only matching provenance; stop clearly on missing competence/success/pairs.

## Review Focus

- Identity path and adapted goal symmetry must survive batched planner integration.
- Partial successful macros must save executed actions without invented duration.
- Resets and persisted cases must not silently resample starts or mix train/test.
- No-success/no-pair runs must stop without emitting a misleading checkpoint.
- Interrupted/resumed stages must reject changed checkpoints/config/manifest.

### Task 1: Metric, fixed resets, and planner hooks

**Files:** `src/mylonas_ijepa_wm/cost_adapter.py`,
`environment_adapters.py`, `plan_pointmaze_proprio.py`, `plan_proprio.py`,
`tests/test_curriculum_cost_adapter.py`.

**Interfaces:** `ResidualTokenAdapter(dim=1280,bottleneck=64)`,
`adapter_goal_cost(adapter,goal)` returns a frozen-episode terminal scorer;
`WallAdapter.reset(seed,layout=None,start=None,goal=None)`;
`plan_action(...,latent_cost_fn=None)`;
`run_episode(...,initial=None,latent_cost_fn=None,observation_observer=None,plan_observer=None)`.

- [x] Write tests for bitwise identity MSE, gradients to adapter only, symmetric
  learned cost, changed CEM actions under a replacement scorer, deterministic
  valid fixed reset, primitive/partial-macro capture.
- [x] Run tests, confirm missing feature failures.
- [x] Implement the metric and backward-compatible optional hooks.
- [x] Run new tests and relevant existing planner/environment tests; commit.

### Task 2: Replay preparation and fixed-schedule fitting

**Files:** `src/mylonas_ijepa_wm/curriculum_replay.py`,
`tests/test_curriculum_replay.py`.

**Interfaces:** `eligible_pairs(states,steps)`; `fit_adapter(episodes,goal,*,seed,...)`
consumes successful CPU token trajectories and returns adapter plus loss history
and training metadata; `save/load` checkpoint uses safe torch data only.

- [x] Write and run failing tests for loop erasure, true elapsed times, empty
  data rejection, train/eval provenance exclusion, nonzero learning and restore.
- [x] Implement spec loss, bounded batches, seeding, and replay validation.
- [x] Verify tests with tiny latents, including decreasing rank inversions; commit.

### Task 3: Staged experiment runner and diagnostics

**Files:** `src/mylonas_ijepa_wm/curriculum_cost_pilot.py`,
`curriculum_diagnostics.py`, `tests/test_curriculum_cost_pilot.py`.

**Interfaces:** CLI stages `prepare`, `baseline`, `position`, `collect`, `train`,
`evaluate`, `report`, `diagnostics`; explicit output/run seed and checkpoint paths.
Manifest stores train cases per seed and shared eval cases. Runtime identity
is a digest of protocol, manifest, model/encoder/source hashes.

- [x] Write and run failing tests for disjoint cases, paired report alignment,
  baseline gate, empty replay status, interrupted/resumed stages, changed
  provenance refusal, and a synthetic end-to-end pilot.
- [x] Implement resumable per-case artifacts, progress messages, hashes, gate,
  all-attempt accounting, training-only encoding, paired summaries and plots.
- [x] Implement cost maps and optional initial/elite simulator candidate audit.
- [x] Verify synthetic workflow and real DotWall resets; commit.

### Task 4: Colab workflow and delivery verification

**Files:** `notebooks/curriculum_cost_adapter_colab.ipynb`,
`docs/curriculum_cost_adapter.md`, `.gitignore`.

**Interfaces:** Notebook launches the staged CLI from `curriculum-cost-adapter`,
using Drive model paths; gate failure halts collection/training automatically.

- [x] Document all stages, budgets, thresholds, expected files, resumes and
  interpretation; provide first-run and conditional replication cells.
- [x] Compile notebook code cells and smoke CLI help/prepare/report.
- [x] Run full pytest with PYTHONPATH=.:src; report unavailable optional deps.
- [x] Perform bounded local real-model plumbing smoke; fresh whole-branch
  review, fix important findings with regression tests, commit final changes.
