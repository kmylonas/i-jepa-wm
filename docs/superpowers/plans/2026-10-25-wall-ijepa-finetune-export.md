# Wall I-JEPA Fine-Tune and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune all trainable parts of the released ImageNet-1K I-JEPA ViT-H/14 encoder and predictor on the existing multi-layout Wall frames, then export an encoder-only checkpoint compatible with the current embedding pipeline.

**Architecture:** A manifest-validated dataset reads frames directly from raw Wall episodes. A standalone trainer uses upstream I-JEPA masking, predictor, target encoder, and Smooth L1 loss, while adding Wall-specific splits, fresh optimization, deterministic validation, resume, and provenance. An export command writes the selected online encoder in the existing checkpoint format.

**Tech Stack:** Python, PyTorch, existing `src.models.vision_transformer`, `src.masks.multiblock`, `src.mylonas_ijepa_wm.wall_layout_manifest`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-wall-ijepa-finetuning-design.md`, limited in this phase to data, I-JEPA fine-tuning, and encoder export. Re-embedding, world-model training, and comparison evaluations are later phases.

## Global Constraints

- Use the existing full episode manifest and its nine training layouts, two validation layouts, and four held-out test layouts. Test-layout images must not enter fine-tuning or model selection.
- Load the released ImageNet-1K ViT-H/14 online encoder, predictor, and EMA target encoder strictly. Start a fresh optimizer and schedule.
- Use original within-image masked patch-token prediction, full 224×224 frames, ImageNet normalization, no horizontal flip, and no geometry-changing crop. Default frame stride is five.
- Train online encoder and predictor; update target encoder by EMA. The primary export is the **online** encoder to match the frozen baseline extraction.
- Full raw episodes and full training outputs stay outside Git. Local pilot frames and the full 1K checkpoint support smoke checks; full CUDA training is a later execution step when the raw data and GPU are available.
- Preserve upstream ImageNet training and existing embedding checkpoint formats.

## Review Focus

- Missing or extra raw episode files: index creation must reject a partial or mixed full dataset before training (Task 1 test).
- Incorrect episode seed/layout/hash or a pilot manifest mixed with full files: reject with the episode ID (Task 1 test).
- Predictor-only or encoder-only released checkpoint, or mismatched ViT shape: reject before optimizer creation (Task 2 test).
- Resume with changed manifest, source checkpoint, frame stride, or hyperparameters: reject before updating weights (Task 3 test).
- Exporting an EMA encoder by accident or writing `module.` prefixes in the encoder-only file: reject/verify through the existing loader (Task 4 test).

---

### Task 1: Wall frame index and dataset

**Files:**
- Create: `src/mylonas_ijepa_wm/wall_ijepa_frames.py`
- Test: `tests/test_mylonas_ijepa_wall_ijepa_frames.py`

**Interfaces:**
- Consumes: `load_manifest(path)`, `manifest_digest(manifest)`, raw `episode_XXXX.pt` fields from `generate_episodes`.
- Produces: `build_frame_index(raw_dir: Path, manifest: dict, frame_stride: int = 5) -> dict`; `WallIJepaFrames(raw_dir: Path, index: dict, split: str) -> Dataset` returning normalized `[3,224,224]` float tensors. The index has manifest SHA-256, stride, per-split episode records (`episode_id`, `seed`, `wall_layout`, `frame_count`), and frame counts. No test samples are exposed through the training or validation dataset.

- [ ] **Step 1: Write failing tests** for deterministic indices 0,5,...,50 from a 51-frame episode; split membership; identical values to `embed_frames.preprocess_frames`; missing/extra episode files; wrong manifest hash, seed, layout, frame count, image shape, or frame stride.
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_mylonas_ijepa_wall_ijepa_frames.py` and confirm the tests fail because the new module/API is absent.
- [ ] **Step 3: Implement** the two interfaces. Build the index by validating all manifest episodes with `torch.load(..., weights_only=True, map_location="cpu")`; keep only episode metadata in memory. Resolve a sample to one episode/frame when read. Use `preprocess_frames` for the exact normalization.
- [ ] **Step 4: Run** the focused tests and confirm zero failures; then commit the dataset file and its tests.

### Task 2: Strict full-checkpoint loading and I-JEPA objective

**Files:**
- Create: `src/mylonas_ijepa_wm/wall_ijepa_objective.py`
- Test: `tests/test_mylonas_ijepa_wall_ijepa_objective.py`

**Interfaces:**
- Consumes: upstream `vit_huge`, `vit_predictor`, `MaskCollator`, `apply_masks`, `repeat_interleave_batch`.
- Produces: `load_released_ijepa(path: Path, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module, dict]`, returning online encoder, predictor, target encoder, and source identity; `masked_prediction_loss(images, masks_enc, masks_pred, encoder, predictor, target_encoder) -> Tensor`; `ema_update(encoder, target_encoder, momentum: float) -> None`.

- [ ] **Step 1: Write failing tests** using a tiny same-interface ViT fixture for strict key loading (strip only leading `module.`), all three required components, expected Smooth L1 target masking and layer normalization, gradient only through online encoder/predictor, and exact EMA interpolation. Include an architecture-mismatch failure.
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_mylonas_ijepa_wall_ijepa_objective.py` and confirm expected failures.
- [ ] **Step 3: Implement** `load_released_ijepa` with the 224/14 ViT-H encoder and 12-layer/384-dim predictor, loading with `weights_only=True` and `mmap=True` where supported. Match upstream `src/train.py` loss and mask repetition. Freeze the EMA target.
- [ ] **Step 4: Run** focused tests; run a read-only preflight on the local `src/checkpoints/IN1K-vit.h.14-300e.pth.tar` to verify encoder/predictor/target keys and shapes; then commit.

### Task 3: Resumable Wall fine-tuning command

**Files:**
- Create: `src/mylonas_ijepa_wm/finetune_ijepa_wall.py`
- Create: `configs/wall_ijepa_finetune.json`
- Test: `tests/test_mylonas_ijepa_wall_ijepa_finetune.py`

**Interfaces:**
- Consumes: Task 1 dataset/index and Task 2 model/loss functions.
- Produces: `python -m src.mylonas_ijepa_wm.finetune_ijepa_wall --raw-dir ... --episode-manifest ... --checkpoint ... --output-dir ... --config configs/wall_ijepa_finetune.json [--resume]`. Saves `last.pt`, `best.pt`, `history.json`, and `frame_index.json`; reports train/validation loss. `best.pt` is selected by validation loss only.

- [ ] **Step 1: Write failing tests** for one tiny-model optimization step changing online/predictor weights and EMA target, deterministic validation masks/frame order, best-checkpoint selection, exact resume of optimizer/scheduler/RNG state, and rejection of changed manifest/source/stride/config before an optimizer step. Test that only train/val splits are constructed.
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_mylonas_ijepa_wall_ijepa_finetune.py` and confirm expected failures.
- [ ] **Step 3: Implement** the CLI with configurable epochs, microbatch, gradient accumulation, learning rate, weight decay, warmup, EMA schedule, precision, mask settings, workers, and seed. Set initial config to 20 epochs, microbatch 1, accumulation 16, learning rate `1e-5`, weight decay `0.04`, two warmup epochs, EMA `0.996` to `1.0`, frame stride 5, upstream 1 context/4 target masks, and BF16 on supported CUDA. Use fixed validation mask seeds and one saved run identity containing config, manifest hash, source checkpoint hash, and frame-index hash. Save atomically after each epoch; refuse to overwrite unrelated outputs.
- [ ] **Step 4: Run** focused tests and CLI `--help`; then commit.

### Task 4: Encoder-only export and compatibility check

**Files:**
- Create: `src/mylonas_ijepa_wm/export_wall_ijepa_encoder.py`
- Test: `tests/test_mylonas_ijepa_wall_ijepa_export.py`
- Modify: `docs/superpowers/specs/2026-10-25-wall-ijepa-finetuning-design.md` only if implementation reveals a decision change.
- Create: `docs/wall_ijepa_finetuning.md` (commands and artifact identities for this phase)

**Interfaces:**
- Consumes: Task 3 `best.pt` and existing `embed_frames.load_checkpoint`.
- Produces: `python -m src.mylonas_ijepa_wm.export_wall_ijepa_encoder --checkpoint .../best.pt --output .../best.encoder-only.pth.tar`; output contains `{"encoder": state_dict_without_module_prefix, "epoch": int}` plus nonbreaking provenance fields.

- [ ] **Step 1: Write failing tests** for online-encoder selection, exact parameter equality to `best.pt`, absence of `module.` keys, rejection of missing or malformed encoder weights, and loading through `embed_frames.load_checkpoint` with a 224/14 ViT-H instance (a tiny fixture may cover export logic; a separate real-model smoke covers compatibility).
- [ ] **Step 2: Run** `/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q tests/test_mylonas_ijepa_wall_ijepa_export.py` and confirm expected failures.
- [ ] **Step 3: Implement** the export command. Document a pilot command using the existing local pilot raw directory, the full local checkpoint, and a short run in a separate output directory; document the full CUDA command and which raw/manifest paths must be provided. Do not start full training in this phase.
- [ ] **Step 4: Run** focused tests, all four new test modules, and the repository test suite; inspect export with the existing loader, run `git diff --check`, and commit. Report any unrelated test failures by name.

## Self-review

- Dataset and split controls live in Task 1; full checkpoint and objective in Task 2; optimizer, selection, resume, and reproducibility in Task 3; export and downstream file compatibility in Task 4.
- Re-embedding and world-model training are deliberately excluded from this plan under the user's step-by-step scope.
- Task interfaces use one `frame_stride` field and the same `best.pt` path throughout.
