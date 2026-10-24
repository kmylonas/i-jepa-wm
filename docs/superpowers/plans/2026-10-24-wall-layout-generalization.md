# Wall Layout Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a reproducible DotWall experiment that trains one I-JEPA world model on nine layouts and evaluates prediction and planning on unseen wall and doorway positions.

**Architecture:** A versioned episode manifest assigns each ID its split, seed, layout, and collection policy. Existing collection, embedding, training, rollout evaluation, and planning entry points consume that information while retaining old defaults. Small experiment-specific commands prepare manifests, audit raw data, probe collisions, and run the planning matrix.

**Tech Stack:** Python, PyTorch, NumPy, pinned DINO-WM DotWall, pytest/unittest, JSON.

**Spec:** `docs/superpowers/specs/2026-10-24-wall-layout-generalization-design.md`

## Global Constraints

- Train layouts: all pairs of wall x `[24, 32, 40]` and door y `[18, 30, 42]`; validation `[[28, 24], [36, 36]]`; test `[[26, 22], [26, 38], [38, 22], [38, 38]]`.
- Target counts: 2,000 train, 200 validation, 400 test; 50 primitive actions per episode; five per WM step; embedding stride 1; `num_hist=1`; one-step training from scratch.
- Correlated wide-policy action lengths span `[0.2, 3.5]` with meaningful upper-tail coverage; alternate wide and door-directed episodes per layout. Keep the old `[0.2, 1.8]` DINO policy unchanged.
- DotWall `wall_width=6` and `door_space=4` remain fixed. Layout positions must not enter WM inputs.
- Planning modes are unbounded, Euclidean cap 4.0, and Euclidean cap 3.5; all use identical seeds and CEM settings. For capped modes, disable the adapter component box before applying the norm cap.
- Do not modify the user's existing `PROJECT_CONTEXT.md` edit. Do not claim a full multi-layout result without its checkpoint and outputs.

## Review Focus

- Missing or duplicated manifest episode IDs: generation and training reject them before partial output.
- Manifest layout disagrees with raw/cache metadata: resume and training fail with the offending ID.
- A layout appears in more than one split: training fails before any optimizer work.
- Wide-policy upper tail is absent despite a 3.5 maximum: pilot audit reports the high-length bins and flags inadequate support.
- Capped CEM still inherits `[-1, 1]` component limits: planning tests verify executed lengths above `sqrt(2)` and at or below the requested norm cap.

---

### Task 1: Manifest and layout-selected collection

**Files:**
- Create: `src/mylonas_ijepa_wm/wall_layout_manifest.py` (schema, deterministic preparation, validation, CLI)
- Modify: `src/mylonas_ijepa_wm/environment_adapters.py` (`WallAdapter.reset` optional layout override)
- Modify: `src/mylonas_ijepa_wm/generate_episodes.py` (manifest-aware collection)
- Test: `tests/test_mylonas_ijepa_wall_layout_manifest.py`

**Interfaces:**
- `prepare_manifest(base_seed: int = 0, train_episodes: int = 2000, val_episodes: int = 200, test_episodes: int = 400) -> dict`: returns schema version 1 with `train_layouts`, `val_layouts`, `test_layouts`, and a list of `{"episode_id", "seed", "split", "wall_layout", "collection_policy"}` entries; IDs contiguous from zero, assignments balanced per layout, policies alternate within each layout. Pilot counts are 90/20/40.
- `load_manifest(path: Path) -> dict`: validates counts, IDs, unique/disjoint layout sets, valid positions, and policies.
- `WallAdapter.reset(seed: int, layout: tuple[int, int] | None = None) -> StepResult`: explicit layout overrides seed-modulo selection; old callers keep existing behavior.
- Collector accepts `--episode-manifest PATH`, uses recorded IDs/seeds/layouts/policies, and stores manifest identity plus seed in raw episodes. Existing CLI mode stays unchanged.

- [ ] **Step 1: Write failing tests** for exact layout sets and split counts, per-layout policy balance, duplicate/missing IDs, split overlap, explicit layout override, and collector raw metadata.
- [ ] **Step 2: Run** `pytest -q tests/test_mylonas_ijepa_wall_layout_manifest.py`; require expected failures before implementation.
- [ ] **Step 3: Implement** the interfaces above and a `prepare` CLI that writes manifest and one superset environment config using a supplied DINO-WM checkout path.
- [ ] **Step 4: Run** the focused test and existing `tests/test_mylonas_ijepa_multi_environment.py`; require zero failures.
- [ ] **Step 5: Commit** only this task's files.

### Task 2: Wide collection policy and pre-embedding coverage audit

**Files:**
- Modify: `src/mylonas_ijepa_wm/generate_episodes.py`
- Create: `src/mylonas_ijepa_wm/audit_wall_dataset.py`
- Test: `tests/test_mylonas_ijepa_wall_coverage.py`

**Interfaces:**
- `sample_wide_wall_actions(rng, steps) -> np.ndarray`: correlated heading process with lengths uniformly sampled in `[0.2, 3.5]`; raw metadata records that distribution and bound.
- `classify_transition(start, action, end, layout) -> str`: labels `solid_contact`, `door_crossing_left_to_right`, `door_crossing_right_to_left`, `border_contact`, or `other` from actual state motion and DotWall geometry; reused by rollout evaluation.
- `audit_dataset(raw_dir: Path, manifest: dict) -> dict`: per-layout/per-policy episode counts, length bins `>1.8`, `>3.0`, `<=3.5`, solid-wall contacts, doorway crossings by direction, and border contacts, with example transitions and coverage pass/fail.
- Manifest collection policies are `wide` and `door`; keep existing `random`, `mixed`, and `dino` behavior.

- [ ] **Step 1: Write failing tests** using synthetic trajectories for wall contact, free doorway crossing, both directions, border contact, and upper-tail action sampling; assert existing DINO samples still cap at 1.8.
- [ ] **Step 2: Run** `pytest -q tests/test_mylonas_ijepa_wall_coverage.py`; require expected failures.
- [ ] **Step 3: Implement** the wide policy and JSON audit CLI; the audit validates every raw episode against the manifest and reports the spec's full-data thresholds (25 contacts, 25 crossings, five each direction per training layout) without enforcing them for the 10/layout pilot.
- [ ] **Step 4: Run** focused and existing multi-environment tests; require zero failures.
- [ ] **Step 5: Commit** only this task's files.

### Task 3: Layout-safe training and per-layout rollout metrics

**Files:**
- Modify: `src/mylonas_ijepa_wm/embed_frames.py` (copy seed and manifest identity to sidecar)
- Modify: `src/mylonas_ijepa_wm/wm_proprio.py` (`--episode-manifest` and split validation)
- Modify: `src/mylonas_ijepa_wm/evaluate_rollouts_proprio.py` (`--by-wall-layout` and transition subsets)
- Test: `tests/test_mylonas_ijepa_wall_layout_training.py`

**Interfaces:**
- `episode_ids_by_split(manifest: dict, trajectory_dir: Path) -> tuple[list[int], list[int], list[int]]`: validates saved layout/seed/manifest identity and disjoint layout groups before returning IDs.
- Trainer uses manifest IDs instead of `wm.create_episode_splits` when supplied, computes normalization from train IDs only, and records exact split layouts and manifest hash in checkpoint metadata; resume rejects changes.
- Rollout evaluator groups checkpoint test IDs by saved `wall_layout` and reports existing state/token horizons and persistence baseline per layout. It also uses Task 2's transition classifier to select windows containing solid contacts or doorway crossings, and reports those subsets separately.

- [ ] **Step 1: Write failing tests** for exact manifest IDs, mismatched seed/layout/hash, layout leakage, train-only statistics, per-layout rollout grouping, and contact/crossing window selection.
- [ ] **Step 2: Run** `pytest -q tests/test_mylonas_ijepa_wall_layout_training.py`; require expected failures.
- [ ] **Step 3: Implement** metadata propagation, trainer selection/validation, and grouped evaluation without changing episode-random default behavior.
- [ ] **Step 4: Run** focused tests plus `tests/test_mylonas_ijepa_wm_proprio.py` and `tests/test_mylonas_ijepa_wm_rollout_eval.py`; require zero failures.
- [ ] **Step 5: Commit** only this task's files.

### Task 4: Collision probes on unseen geometry

**Files:**
- Create: `src/mylonas_ijepa_wm/evaluate_wall_collision_probes.py`
- Test: `tests/test_mylonas_ijepa_wall_collision_probes.py`

**Interfaces:**
- `build_probes(layout: tuple[int, int]) -> list[dict]`: deterministic new-wall blocked, doorway-crossing, and old-training-wall free-passage cases in both feasible directions, with five primitive actions each.
- `evaluate_probes(checkpoint, encoder_checkpoint, layouts, seeds, output) -> dict`: obtains real rendered observations and simulator outcomes, predicts one WM step, and saves actual/predicted state, crossing classification, progress error, and token MSE per case and layout.
- The command uses the same I-JEPA encoder and normalization as planning; simulator geometry defines only probe setup/labels.

- [ ] **Step 1: Write failing tests** that assert every probe's real DotWall outcome matches its intended class, including free passage at an old wall coordinate, and that a tiny mock WM yields correctly denormalized records.
- [ ] **Step 2: Run** `pytest -q tests/test_mylonas_ijepa_wall_collision_probes.py`; require expected failures.
- [ ] **Step 3: Implement** probe setup, one-step model inference, classification, JSON output, and CLI.
- [ ] **Step 4: Run** focused tests; require zero failures.
- [ ] **Step 5: Commit** only this task's files.

### Task 5: Matched planning matrix and runbook

**Files:**
- Modify: `src/mylonas_ijepa_wm/plan_proprio.py` (record actual layout per episode)
- Create: `src/mylonas_ijepa_wm/run_wall_planning_matrix.py`
- Create: `docs/wall_layout_generalization.md`
- Test: `tests/test_mylonas_ijepa_wall_planning_matrix.py`

**Interfaces:**
- Matrix CLI takes encoder checkpoint, multi-layout checkpoint, fixed-layout checkpoint, manifest, output root, base seed, and 20 episodes. It emits one fixed-layout config per test layout and runs both models under modes `unbounded`, `cap4`, `cap_train` with the spec's common CEM settings.
- `cap_train` reads 3.5 from the manifest's recorded collection maximum; both capped modes call planning with `--unbounded-actions --max-action-length VALUE`.
- Matrix summary includes per-model/layout/mode successes, Wilson 95% intervals, final distances, failure counts, and links to raw JSON outputs. Fixed-layout reference is labelled as a non-isolated baseline.
- Runbook gives pilot and full collection/audit, dense embedding, CUDA training, rollout/probe evaluation, and matrix commands with resumable batches and expected artifact paths.

- [ ] **Step 1: Write failing tests** for six planning invocations per test layout, matched seeds/settings, cap values, actual layout in records, and successful actions longer than `sqrt(2)` but within norm caps.
- [ ] **Step 2: Run** `pytest -q tests/test_mylonas_ijepa_wall_planning_matrix.py`; require expected failures.
- [ ] **Step 3: Implement** runner, result aggregation, layout recording, and runbook.
- [ ] **Step 4: Run** focused tests, the existing planning tests, and `git diff --check`; require zero failures.
- [ ] **Step 5: Commit** only this task's files.

### Task 6: Local pilot and final verification

**Files:**
- Modify: `docs/wall_layout_generalization.md` with measured pilot findings and limitations.

**Interfaces:**
- Pilot manifest uses 10 episodes per layout with the same layout sets and policies; outputs live outside Git.

- [ ] **Step 1: Run** one small manifest-based collection batch through the local DotWall adapter and audit it; inspect example transitions and action-tail counts.
- [ ] **Step 2: Embed a minimal sample** using the local I-JEPA checkpoint and verify frame/trajectory alignment; do not launch full dense embedding or full training on this MPS-only machine.
- [ ] **Step 3: Run** the full relevant pytest selection and a CLI help smoke check for each new command; record exact command outcomes and any unavailable dependencies.
- [ ] **Step 4: Update** the runbook with the pilot's real counts, storage/compute notes, and exact next CUDA commands.
- [ ] **Step 5: Commit** the runbook update and review Git status without staging `PROJECT_CONTEXT.md`.
