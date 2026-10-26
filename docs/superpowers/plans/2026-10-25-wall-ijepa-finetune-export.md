# Wall I-JEPA Fine-Tune and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune all trainable parts of the released ImageNet-1K I-JEPA ViT-H/14 encoder and predictor on the existing multi-layout Wall frames, then export an encoder-only checkpoint compatible with the current embedding pipeline.

**Architecture:** A manifest-validated dataset reads frames directly from raw Wall episodes. The existing `src/train.py` pipeline gets a Wall data mode and strict released-checkpoint initialization while keeping its I-JEPA model, masks, Smooth L1 loss, schedules, and EMA update. Wall mode adds deterministic validation, best-checkpoint selection, resume checks, and provenance. An export command writes the selected online encoder in the existing checkpoint format.

**Tech Stack:** Python, PyTorch, existing `src.models.vision_transformer`, `src.masks.multiblock`, `src.mylonas_ijepa_wm.wall_layout_manifest`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-wall-ijepa-finetuning-design.md`, limited in this phase to data, I-JEPA fine-tuning, and encoder export. Re-embedding, world-model training, and comparison evaluations are later phases.

## Global Constraints

- Use the existing full episode manifest and its nine training layouts, two validation layouts, and four held-out test layouts. Test-layout images must not enter fine-tuning or model selection.
- Load the released ImageNet-1K ViT-H/14 online encoder, predictor, and EMA target encoder strictly. Start a fresh optimizer and schedule.
- Use original within-image masked patch-token prediction, full 224×224 frames, ImageNet normalization, no horizontal flip, and no geometry-changing crop. Default frame stride is five.
- Train online encoder and predictor; update target encoder by EMA. The primary export is the **online** encoder to match the frozen baseline extraction.
- Full raw episodes and full training outputs stay outside Git. Local pilot frames and the full 1K checkpoint support smoke checks; full CUDA training is a later execution step when the raw data and GPU are available.
- Preserve the current ImageNet mode of `src/train.py` and the existing embedding checkpoint format. Do not create a second I-JEPA training loop.

## Review Focus

- Missing or extra raw episode files: index creation must reject a partial or mixed full dataset before training (Task 1 test).
- Incorrect episode seed/layout/hash or a pilot manifest mixed with full files: reject with the episode ID (Task 1 test).
- Predictor-only or encoder-only released checkpoint, or mismatched ViT shape: reject before optimizer creation (Task 2 test).
- Resume with changed manifest, source checkpoint, frame stride, or hyperparameters: reject before updating weights (Task 3 test).
- Exporting an EMA encoder by accident or writing `module.` prefixes in the encoder-only file: reject/verify through the existing loader (Task 4 test).

---

### Task 1: Wall frame index and dataset for the existing collator

**Files:**
- Create: `src/datasets/wall_frames.py`
- Test: `tests/test_mylonas_ijepa_wall_ijepa_frames.py`

**Interfaces:**
- Consumes: `load_manifest(path)`, `manifest_digest(manifest)`, raw `episode_XXXX.pt` fields from `generate_episodes`.
- Produces: `build_frame_index(raw_dir: Path, manifest: dict, frame_stride: int = 5) -> dict`; `WallFrames(raw_dir: Path, index: dict, split: str) -> Dataset` returning `(normalized_image, 0)` as expected by the current `MaskCollator` and `src/train.py`. The index has manifest SHA-256, stride, per-split episode records (`episode_id`, `seed`, `wall_layout`, `frame_count`), and frame counts. The trainer constructs only train and validation datasets.

- [ ] **Step 1: Write failing tests** for deterministic indices 0,5,...,50 from a 51-frame episode; split membership; identical values to `embed_frames.preprocess_frames`; missing/extra episode files; wrong manifest hash, seed, layout, frame count, image shape, or frame stride.
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_mylonas_ijepa_wall_ijepa_frames.py` and confirm the tests fail because the new module/API is absent.
- [ ] **Step 3: Implement** the two interfaces. Build the index by validating all manifest episodes with `torch.load(..., weights_only=True, map_location="cpu")`; keep only episode metadata in memory. Resolve a sample to one episode/frame when read. Use `preprocess_frames` for the exact normalization and return a dummy target solely to match the upstream collator contract.
- [ ] **Step 4: Run** the focused tests and confirm zero failures; then commit the dataset file and its tests.

### Task 2: Strict initialization of the upstream I-JEPA model

**Files:**
- Modify: `src/helper.py`
- Test: `tests/test_ijepa_wall_checkpoint_init.py`

**Interfaces:**
- Consumes: the `init_model` encoder and predictor plus its deep-copied target encoder, before DDP wrapping and optimizer creation.
- Produces: `initialize_from_released_checkpoint(path, encoder, predictor, target_encoder) -> dict` in `src/helper.py`. It loads all three state dictionaries strictly, stripping a leading `module.` only, and returns source epoch/checkpoint identity. The existing `src/train.py` loss, mask repetition, and EMA remain the sole implementation of the objective.

- [ ] **Step 1: Write failing tests** using tiny modules for strict loading of all three components, stripping only leading `module.`, missing predictor or target weights, architecture mismatch, and proof that optimizer state is not loaded. Assert that malformed input raises rather than falling through to random initialization.
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_ijepa_wall_checkpoint_init.py` and confirm expected failures.
- [ ] **Step 3: Implement** the helper using `torch.load(..., weights_only=True, mmap=True, map_location="cpu")`; keep the existing `load_checkpoint` resume helper unchanged for ImageNet mode. Verify the local released file's `module.` prefix and 224/14 ViT-H, 12-layer/384-dim predictor match the run config.
- [ ] **Step 4: Run** focused tests; run a read-only preflight on the local `src/checkpoints/IN1K-vit.h.14-300e.pth.tar` to verify encoder/predictor/target keys and shapes; then commit.

### Task 3: Wall mode in the existing I-JEPA training pipeline

**Files:**
- Modify: `src/train.py`
- Create: `configs/wall_ijepa_finetune.yaml`
- Test: `tests/test_ijepa_wall_train.py`

**Interfaces:**
- Consumes: Task 1 dataset/index and Task 2 strict initializer; the existing `main.py --fname ... --devices cuda:0` launcher, `init_model`, `init_opt`, `MaskCollator`, target/context forward calls, and EMA update.
- Produces: `python main.py --fname configs/wall_ijepa_finetune.yaml --devices cuda:0` with `data.dataset: wall`, `data.raw_dir`, `data.episode_manifest`, and `meta.init_checkpoint`. It saves the upstream-style latest checkpoint plus a best-validation checkpoint, `history.json`, and `frame_index.json`. The released checkpoint initializes weights only; a later Wall run resumes from its own latest checkpoint.

- [ ] **Step 1: Write failing tests** for Wall data-loader selection versus unchanged ImageNet selection, strict source initialization before a fresh optimizer, one masked-loss optimization step changing online/predictor weights and EMA target, deterministic validation masks/frame order, best-checkpoint selection, exact Wall resume, and rejection of changed manifest/source/stride/config before an optimizer step. Test that only train/val splits are constructed and that a one-process CPU smoke need not construct DDP.
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_ijepa_wall_train.py` and confirm expected failures.
- [ ] **Step 3: Extend** `src/train.py` behind `data.dataset: wall` while keeping ImageNet as the default. Reuse its forward target/context, Smooth L1 loss, optimizer and weight-decay schedulers, and EMA update. Add a Wall validation loader and fixed validation-mask seed, best-checkpoint selection, run identity, atomic Wall checkpoints, and strict Wall resume; do not route released initialization through `load_checkpoint`, which also restores the pretraining optimizer and catches errors. Step schedulers and EMA once per optimizer update when accumulating gradients. Allow the single-process pilot to run without DDP; preserve DDP on initialized multi-process CUDA runs. Set the YAML to 20 epochs, batch size 1, gradient accumulation 16, learning rate `1e-5`, weight decay `0.04`, two warmup epochs, EMA `0.996` to `1.0`, frame stride 5, upstream 1 context/4 target masks, and BF16 on supported CUDA.
- [ ] **Step 4: Run** focused tests, the existing ImageNet CLI/config smoke, and `main.py --help`; then commit.

### Task 4: Encoder-only export and compatibility check

**Files:**
- Create: `src/mylonas_ijepa_wm/export_wall_ijepa_encoder.py`
- Test: `tests/test_mylonas_ijepa_wall_ijepa_export.py`
- Modify: `docs/superpowers/specs/2026-10-25-wall-ijepa-finetuning-design.md` only if implementation reveals a decision change.
- Create: `docs/wall_ijepa_finetuning.md` (commands and artifact identities for this phase)

**Interfaces:**
- Consumes: Task 3 upstream-format best checkpoint and existing `embed_frames.load_checkpoint`.
- Produces: `python -m src.mylonas_ijepa_wm.export_wall_ijepa_encoder --checkpoint .../jepa-best.pth.tar --output .../jepa-best.encoder-only.pth.tar`; output contains `{"encoder": state_dict_without_module_prefix, "epoch": int}` plus nonbreaking provenance fields.

- [ ] **Step 1: Write failing tests** for online-encoder selection, exact parameter equality to `jepa-best.pth.tar`, absence of `module.` keys, rejection of missing or malformed encoder weights, and loading through `embed_frames.load_checkpoint` with a 224/14 ViT-H instance (a tiny fixture may cover export logic; a separate real-model smoke covers compatibility).
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_mylonas_ijepa_wall_ijepa_export.py` and confirm expected failures.
- [ ] **Step 3: Implement** the export command, reusing the format demonstrated by `src/extract_encoder.py` but taking explicit paths and validating `best` selection. Document a pilot config using the existing local pilot raw directory, the full local checkpoint, and a short run in a separate output directory; document the full CUDA config and which raw/manifest paths must be provided. Do not start full training in this phase.
- [ ] **Step 4: Run** focused tests, all four new test modules, and the repository test suite; inspect export with the existing loader, run `git diff --check`, and commit. Report any unrelated test failures by name.

## Self-review

- Dataset and split controls live in Task 1; strict source initialization in Task 2; the existing objective plus Wall optimizer, validation, resume, and reproducibility in Task 3; export and downstream file compatibility in Task 4.
- Re-embedding and world-model training are deliberately excluded from this plan under the user's step-by-step scope.
- Task interfaces use one `frame_stride` field and the same `jepa-best.pth.tar` path throughout.
