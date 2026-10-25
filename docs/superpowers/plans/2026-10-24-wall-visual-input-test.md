# Matched-layout Visual-Input Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled DotWall visual-input probe with saved trajectory figures and Colab instructions.

**Architecture:** A new evaluation module constructs fixed two-layout comparisons, runs the same position and actions through both environments and the checkpoint, and writes JSON. A small plotting function in that module renders two panels from the records using the environment's fixed wall geometry.

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, pinned DINO-WM Wall adapter.

**Spec:** `docs/superpowers/specs/2026-10-24-wall-visual-input-test-design.md`

## Global Constraints

- Keep each pair's start and five actions identical; only the rendered layout may differ as model input.
- Use held-out manifest layouts only, validate checkpoint against manifest, and never tune or discard cases based on model predictions.
- Plot actual primitive trajectory and only the model's final predicted point.
- Run from a fresh Colab clone without cached training episodes.

## Review Focus

- A supposed matched pair whose actual simulator outcomes do not contrast must fail rather than silently enter the summary; test in Task 1.
- A model accidentally receiving a layout identifier or a different start/action across the pair would invalidate the claim; test in Task 1.
- A plot that shows a model trajectory the one-step checkpoint never produced would mislead readers; test plot labels and marks in Task 2.
- A checkpoint from another manifest must be rejected before loading the encoder; test in Task 1.
- Missing output directories on fresh Colab must be created; test in Task 2.

---

### Task 1: Matched simulator and model comparisons

**Files:**
- Create: `src/mylonas_ijepa_wm/evaluate_wall_visual_pairs.py`
- Create: `tests/test_mylonas_ijepa_wall_visual_pairs.py`

**Interfaces:**
- Produces: `build_matched_pairs() -> list[dict]`, `evaluate_pair(adapter, encoder, model, metadata, pair, seed, device, precision) -> dict`, `evaluate_pairs(checkpoint, encoder_checkpoint, manifest, output, env_config, device_name, precision_name) -> dict`.

- [x] Write tests proving fixed pair starts/actions, held-out layouts, contrasting actual crossings, and a fake model whose only varying input is the latent.
- [x] Run the test file and observe failure because the module or functions are absent.
- [x] Implement fixed comparisons, prediction and simulation records, manifest validation, and JSON output.
- [x] Run the test file and observe all tests pass.
- [x] Commit Task 1.

### Task 2: Figures and Colab usage

**Files:**
- Modify: `src/mylonas_ijepa_wm/evaluate_wall_visual_pairs.py`
- Modify: `tests/test_mylonas_ijepa_wall_visual_pairs.py`
- Modify: `docs/wall_layout_generalization.md`

**Interfaces:**
- Consumes: Task 1 JSON records, including every primitive position and predicted endpoint.
- Produces: one PNG per pair in the output directory and one documented Colab CLI invocation.

- [x] Write a failing plot test checking file creation, two layout panels, actual path, predicted endpoint marker, and faint training x guides.
- [x] Run that test and observe the expected failure.
- [x] Implement figures and document the fresh-Colab command and interpretation.
- [x] Run the new tests and relevant existing Wall tests; verify CLI `--help`.
- [x] Commit Task 2.
