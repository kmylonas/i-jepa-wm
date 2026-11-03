# Matched Wall Layout Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an uploadable Colab notebook that trains and evaluates new fixed-layout and multi-layout Wall world models with layout diversity as the sole planned training-data difference.

**Architecture:** Extend the existing manifest and audit to describe separate fixed and multi datasets using the correlated wide policy with action lengths 0.2–3.5. Reuse collection, dense embedding, manifest-based training, rollout evaluation, and planning. Update comparison utilities to accept two manifest-backed checkpoints; orchestrate sequential runtime-local datasets in one Colab notebook while saving metadata, checkpoints, and results to Drive.

**Tech Stack:** Python, PyTorch, NumPy, pinned DINO-WM DotWall, Jupyter/Colab, pytest, JSON.

**Spec:** `docs/superpowers/specs/2026-11-03-matched-wall-layout-comparison-design.md`

## Global Constraints

- Training layouts are fixed `[[32, 30]]` versus multi `[24, 32, 40] × [18, 30, 42]`; validation `[[28, 24], [36, 36]]`; test `[[26, 22], [26, 38], [38, 22], [38, 38]]`.
- Each model gets a separate dataset of 2,000 training, 200 validation, and 400 test episodes; 50 primitive actions per episode; only the existing `wide` correlated policy with uniform lengths `[0.2, 3.5]`.
- Both use the same frozen I-JEPA checkpoint, DotWall commit, model configuration, training settings, and seed. Embedding stride is 1; history and rollout training steps are 1.
- Primary planning uses position cost, 50 seeds per test layout, identical CEM settings, and Euclidean action cap 3.5. The 0.2 training lower bound is not a planning lower bound.
- Raw frames, episodes, latent caches, and trajectory sidecars stay under `/content`; Drive holds only manifests/configs, checkpoints/logs, and results.
- Preserve old wide/door manifests, historical results, and their current default behavior.

## Review Focus

- Fixed manifest with only the default `[32, 30]` training layout: accept it, while still rejecting train/validation/test coordinate leakage.
- Mixed policy or action maximum in a new manifest: reject before collecting or training.
- Paired seeds whose realized training actions diverge unexpectedly: expose the mismatch in the pilot audit.
- Insufficient runtime disk or interrupted embedding batch: fail before filling disk, then resume only verified cache pairs.
- Fixed checkpoint with a manifest in visual probes/planning: validate against its own manifest rather than treating it as the historical checkpoint.

---

### Task 1: Matched manifests and coverage audit

**Files:**
- Modify: `src/mylonas_ijepa_wm/wall_layout_manifest.py`
- Modify: `src/mylonas_ijepa_wm/audit_wall_dataset.py`
- Test: `tests/test_mylonas_ijepa_wall_layout_manifest.py`
- Test: `tests/test_mylonas_ijepa_wall_coverage.py`

**Interfaces:**
- Extend `prepare_manifest(...)` with explicit training-layout mode and collection-policy arguments while retaining existing defaults. Its matched modes produce fixed and multi manifests with identical validation/test assignments and paired episode seeds; every entry uses `wide`, and `max_training_action_length` is 3.5.
- `validate_manifest(...)` accepts this matched profile, rejects inconsistent policy/action maxima, and continues to accept the existing wide/door profile.
- The audit checks action norms against `[0.2, 3.5]`, reports per-layout contacts and doorway crossings, and records the old mixed-policy coverage threshold as diagnostic for the matched profile. It exports compact per-episode action hashes keyed by seed; the second run compares those with the first run's saved audit, without retaining the first raw directory.

- [ ] Add failing tests for both exact manifest layouts/counts, common validation/test assignments, seed pairing, policy/max mismatches, and preservation of the old profile.
- [ ] Run focused tests and confirm the new cases fail for the expected reasons.
- [ ] Implement the manifest arguments and profile-aware validation; keep collector and trainer entry points unchanged because they already consume explicit manifests and `wide` entries.
- [ ] Add failing audit tests for action-range violations, contact/crossing counts, and paired action divergence; implement the matched audit path.
- [ ] Run the two focused test modules and existing manifest/collection tests; require zero failures. Commit this task.

### Task 2: Matched evaluation utilities

**Files:**
- Modify: `src/mylonas_ijepa_wm/run_wall_planning_matrix.py`
- Modify: `src/mylonas_ijepa_wm/evaluate_wall_visual_pairs.py`
- Test: `tests/test_mylonas_ijepa_wall_planning_matrix.py`
- Test: `tests/test_mylonas_ijepa_wall_visual_pairs.py`

**Interfaces:**
- The planning runner accepts fixed and multi manifest paths, validates each checkpoint against its own manifest, and offers a matched mode with only `cap_train=3.5`, 50 paired seeds per held-out layout, and no mandatory videos. Preserve the historical three-mode runner as its default.
- Visual-pair evaluation accepts a new manifest-backed fixed checkpoint, reads its `[32, 30]` training layout from checkpoint metadata, and validates identity. Preserve historical fixed-checkpoint support.
- Summary records both manifest hashes, encoder/model hashes, exact planner arguments, per-layout paired success counts, and the overall paired success difference. Its 95% interval comes from 10,000 bootstrap samples of the 50 seed IDs, with all four layouts carried together per seed and bootstrap seed `20261103`. The matched planner caps primitive actions at 3.5.

- [ ] Add failing tests for both checkpoint identities, 3.5 cap, 50 shared seeds, matched layout configs, paired aggregation, and historical-run compatibility.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement matched planning and fixed-manifest visual-pair support.
- [ ] Run focused planning/probe tests and `git diff --check`; require zero failures. Commit this task.

### Task 3: Uploadable Colab notebook and runbook

**Files:**
- Create: `notebooks/wall_matched_fixed_multi_colab.ipynb`
- Create: `docs/wall_matched_fixed_multi.md`
- Test: `tests/test_wall_matched_fixed_multi_notebook.py`

**Interfaces:**
- Notebook mounts Drive, pins/checks the GitHub branch commit and DINO-WM commit, copies the frozen encoder locally, creates separate fixed and multi manifests/datasets, and saves small run identities to a new Drive root.
- For each model in sequence: generate/audit raw episodes under `/content`; verify and embed 100-episode batches; delete verified raw batches; train/resume from its Drive checkpoint; save validation/test rollouts and collision/visual probes; verify the Drive checkpoint/results; delete that model's entire `/content` dataset and cache. Then generate the other model's separate dataset. Run matched planning using both saved checkpoints.
- Disk checks use measured free space and conservative per-episode estimates. The notebook refuses paths that put raw/cache data below `/content/drive` and explains regeneration after a runtime reset.
- The runbook lists exact Drive inputs and outputs, step order, expected file sizes, retry behavior, and the fixed-versus-multi control table.

- [ ] Add notebook-structure tests for valid notebook JSON, expected commands/paths, 3.5 policy and planner cap, separate dataset directories, absence of raw/cache Drive paths, and ordered verification before deletion.
- [ ] Run the new test and confirm expected failure.
- [ ] Create the notebook and runbook, reusing the existing Colab setup and training commands.
- [ ] Execute notebook code cells that do not require Colab/GPU in a local smoke harness, then run the focused tests and notebook JSON validation. Commit this task.

### Task 4: Final preflight

**Files:**
- Modify: `docs/wall_matched_fixed_multi.md` only if preflight findings require clarification.

- [ ] Run all relevant unit tests in the project `pointmaze` environment and verify both old and new manifest behavior.
- [ ] Execute a small local paired collection, audit realized action norms and matching policy samples, and embed at least one episode per model using the frozen encoder if the checkpoint is available locally.
- [ ] Verify notebook paths against the existing Drive conventions, estimate disk footprint, inspect Git status, and report that full Colab training remains for the user to run.
- [ ] Commit any runbook correction and provide the notebook path plus exact steps for upload.
