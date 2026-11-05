# Multi-band Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run two independent original/clean curricula through three downward bands, updating from each band's experience and displaying comparable raw/adapted MSE maps.

**Architecture:** Extend the replay fitter with optional warm starting and band-balanced sampling, preserving its default behavior. A separate protocol/runner owns immutable source references, fixed cases, checkpoint lineage, collections and evaluations. A diagnostic module owns a cached fixed grid and checkpoint heatmaps; the notebook invokes resumable stages and prints results.

**Tech Stack:** Python, PyTorch, NumPy, matplotlib, pytest, Colab.

**Spec:** docs/superpowers/specs/2026-11-05-multiband-curriculum-design.md

## Global Constraints

- Same worktree/branch; source pilot_v1, clean_pair_v1 and adjacent_band_v1 are read-only.
- x [40,54]; near y [30,36], new bands [22,30], [14,22], [8,14]; goal (50,42), layout (32,30).
- Two independent branches retain temporal versus >=0.5 oracle-progress pair rules and their own policy experience.
- Twenty held-out cases per band, 40 paired collection starts per band/branch, disjoint from all evaluation starts by >=0.25.
- Warm-start each adapter, fresh Adam each band, 500 updates/batch16/lr1e-4/anchor1; uniform band then episode then pair replay.
- Evaluate before learning the current band and afterward on all learned bands; no automatic performance gate or rollback.
- Stop a branch if its current-band successes provide no usable pairs; never substitute raw-policy experience.
- Use one saved goal and one float32 diagnostic grid (176 points); exclude grid/evaluation data from fitting.
- Print initial and per-band raw/adapted/difference heatmaps with fixed color limits and explicit saturation metadata.
- Up to 840 episodes across raw plus both curricula; count interruptions separately.

## Review Focus

- Later bands must initialize from the exact previous checkpoint and never reuse stale optimizer moments or initialize to identity.
- Resume must validate sources, input replay, cached cases/journals, checkpoints and grid hashes before new actions or accepting completed artifacts.
- Original/clean branch experience must remain separate; only successful current-band collection enters that branch's replay.
- A failed/stopped branch must not prevent the other branch's report; future/missing evaluations are explicitly unavailable.
- Raw and adapted maps use the same real points/goal/scale, never feed training, and expose costs beyond display limits.

### Task 1: Continued fitting with balanced replay

**Files:** src/mylonas_ijepa_wm/curriculum_replay.py; tests/test_curriculum_replay.py.

**Interfaces:** Extend fit_adapter(..., initial_adapter=None, balance_bands=False). Warm-start by copying and unfreezing the module, validating dimensions. Band labels use episode['case']['band']; balance only usable bands. Metadata records fresh optimizer, warm start, band and episode sample counts when enabled; default RNG/schema/parameters remain unchanged.

- [x] Write failing tests for warm-start weights/input immutability, band sampling despite unequal episode counts, invalid dimensions and missing band labels; retain default compatibility tests.
- [x] Run pytest -q tests/test_curriculum_replay.py; expect new-argument failures.
- [x] Implement optional initialization and sampling, including per-band/episode counts.
- [x] Run focused tests; expect all pass; commit isolated fitter change.

### Task 2: Immutable-source sequential runner

**Files:** create src/mylonas_ijepa_wm/curriculum_multiband_protocol.py, src/mylonas_ijepa_wm/curriculum_multiband.py, tests/test_curriculum_multiband.py; modify curriculum_runtime.py only for optional explicit journal stage; update .gitignore.

**Interfaces:** Protocol prepare/load validation, source references, manifest and case-stage preflight. Runner Curriculum with baseline(runtime), initial(recipe,runtime), band(recipe,index,runtime), and report(render=False). Task 3 adds heatmap calls through the runner's diagnostic methods. CLI stages prepare/baseline/grid/initial/band/report.

- [x] Write failing synthetic workflow tests for deterministic geometry and source case reuse, train/eval disjointness, paired independent collection, cumulative branch-isolated replay, checkpoint lineage, zero-success stopping, post-update retention evaluations and interrupted resume/provenance rejection.
- [x] Run pytest -q tests/test_curriculum_multiband.py; expect missing runner/protocol failures.
- [x] Implement source and output checks; fixed manifest; saved goal; stage preflight; raw baseline; branch initialization and current-band evaluation/collection/encoding/update/post-evaluation, using Task 1 fit_adapter options.
- [x] Implement report data with raw/current/incoming comparisons, actual interaction journals and unavailable cells. Add runtime journal_stage override with unchanged defaults.
- [x] Run focused tests and source hash checks; expect all pass; commit runner.

### Task 3: Fixed-grid diagnostics, reporting and standalone Colab

**Files:** create src/mylonas_ijepa_wm/curriculum_heatmaps.py, tests/test_curriculum_heatmaps.py, notebooks/curriculum_multiband_colab.ipynb, docs/curriculum_multiband.md; extend runner/report and tests.

**Interfaces:** prepare_grid(runtime, output, goal, signature), load_grid(output, signature), checkpoint_heatmap(output, recipe, version, adapter, replay_positions, signature), plot_curriculum_report(output, manifest, report). Numeric/cache hashes bind fixed points, preprocessing, execution and goal identities.

- [x] Write failing tests for grid reuse/hash rejection, numeric raw/adapted/difference maps and invariant raw scale, diagnostic exclusion from fitting, and partial report display.
- [x] Run pytest -q tests/test_curriculum_heatmaps.py; expect missing diagnostic module failures.
- [x] Implement chunked real-frame grid encoding/cache and three-panel maps with boundaries/replay points, unclipped values and saturation counts; initial and after-update calls; report success/actions matrices, collection paths and losses.
- [x] Build Run-all Colab with both independent curricula, immediate heatmap display after every band update, stopped-branch handling and final report/download instructions.
- [x] Verify notebook compilation, synthetic full CLI workflow and rendered figures; run full pytest suite; expect all pass.
- [x] Commit; request one fresh whole-change review; fix Important/Critical findings with regression tests and full suite.
- [ ] Pin notebook revision, push current branch, provide direct Colab link; actual GPU results remain pending.
