# PointMaze CEM Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live CEM-MPC evaluator that controls PointMaze from frozen I-JEPA observations with the trained latent world model and an image-specified goal.

**Architecture:** A pure `planning.py` module owns CEM, action alignment, chunked latent rollout, and image-goal costs without importing Gymnasium. A separate `plan_pointmaze.py` runner owns the frozen I-JEPA encoder, PointMaze rendering and stepping, MPC history, CLI configuration, and JSON reporting.

**Tech Stack:** Python, PyTorch, NumPy, Gymnasium-Robotics, MuJoCo, unittest

**Spec:** `docs/superpowers/specs/2026-10-09-pointmaze-cem-planning-design.md`

## Global Constraints

- Load only the frozen I-JEPA image encoder, never its predictor or target encoder.
- Give the world model only I-JEPA tokens and actions, never simulator state.
- Default to a target RGB observation and probe distance between predicted and target-image latents.
- Permit numeric `desired_goal` only in explicitly named `oracle` mode and evaluation metrics.
- Represent one macro action as five 2D controls flattened to dimension 10.
- Execute one macro action and then replan from a real observation.
- Preserve the user's existing uncommitted rollout-comparison changes.

## Review Focus

- Candidate actions must be ordered as two historical macro actions followed by the candidate sequence.
- Normalize world-model actions without modifying actions executed in the environment.
- Chunking must preserve candidate order and equal unchunked evaluation.
- Target rendering must restore MuJoCo position and velocity after success or failure.
- Probe and latent costs must use the target image; only oracle mode may use numeric goal coordinates.

---

### Task 1: Candidate rollout and image-goal costs

**Files:**
- Create: `src/mylonas_ijepa_wm/planning.py`
- Create: `tests/test_mylonas_ijepa_wm_planning.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `evaluate_rollouts.autoregressive_rollout` and `evaluate_rollouts.decode_positions`.
- Produces: `assemble_rollout_actions(...)`, `rollout_action_sequences(...)`, and `compute_terminal_cost(...)`.

- [ ] **Step 1: Expose the new test file**

Add `!tests/test_mylonas_ijepa_wm_planning.py` beside the existing world-model test exceptions.

- [ ] **Step 2: Write failing action-alignment tests**

```python
def test_prepends_history_and_preserves_raw_candidates():
    history = torch.tensor([[10.0], [20.0]])
    candidates = torch.tensor([
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
    ])
    actions = planning.assemble_rollout_actions(history, candidates)
    torch.testing.assert_close(actions, torch.tensor([
        [[10.0], [20.0], [1.0], [2.0], [3.0]],
        [[10.0], [20.0], [4.0], [5.0], [6.0]],
    ]))


def test_normalizes_model_actions_without_changing_candidates():
    history = torch.tensor([[10.0], [20.0]])
    candidates = torch.tensor([[[1.0], [2.0]]])
    original = candidates.clone()
    actions = planning.assemble_rollout_actions(
        history,
        candidates,
        action_mean=torch.tensor([1.0]),
        action_std=torch.tensor([2.0]),
    )
    torch.testing.assert_close(
        actions,
        torch.tensor([[[4.5], [9.5], [0.0], [0.5]]]),
    )
    torch.testing.assert_close(candidates, original)
```

- [ ] **Step 3: Run the action tests and verify RED**

Run `/Users/kmylonas/miniforge3/envs/ML/bin/python -m unittest tests.test_mylonas_ijepa_wm_planning.ActionAlignmentTest`.

Expected: import failure because `planning.py` is absent.

- [ ] **Step 4: Implement action alignment**

Implement `assemble_rollout_actions(action_history, candidate_actions, action_mean=None, action_std=None)`. Validate `[history, action_dim]` and `[population, horizon, action_dim]`, expand history across candidates, concatenate it before candidates, and normalize only the new combined tensor.

- [ ] **Step 5: Run the action tests and verify GREEN**

Run Step 3 again. Expected: PASS.

- [ ] **Step 6: Write failing chunked-rollout tests**

```python
def test_chunked_rollout_preserves_candidate_order():
    history = torch.tensor([[[[0.0]], [[1.0]], [[3.0]]]])
    action_history = torch.tensor([[1.0], [2.0]])
    candidates = torch.tensor([
        [[3.0], [4.0]],
        [[5.0], [6.0]],
        [[7.0], [8.0]],
    ])
    predictions = planning.rollout_action_sequences(
        model=AddLastActionWorldModel(),
        latent_history=history,
        action_history=action_history,
        candidate_actions=candidates,
        device=torch.device("cpu"),
        chunk_size=2,
    )
    torch.testing.assert_close(
        predictions[:, :, 0, 0],
        torch.tensor([[6.0, 10.0], [8.0, 14.0], [10.0, 18.0]]),
    )
```

Also reject a shared latent-history batch other than one and `chunk_size < 1`.

- [ ] **Step 7: Run the rollout tests and verify RED**

Expected: failure because `rollout_action_sequences` is absent.

- [ ] **Step 8: Implement chunked autoregressive rollout**

Slice candidates in stable order, expand the shared history only to the current chunk, assemble aligned actions, call `autoregressive_rollout`, and concatenate predictions in original order. Support FP32 and BF16 autocast.

- [ ] **Step 9: Run the rollout tests and verify GREEN**

Expected: PASS.

- [ ] **Step 10: Write failing cost tests**

```python
def test_latent_cost_is_terminal_patch_mse():
    predictions = torch.tensor([
        [[[0.0], [2.0]]],
        [[[1.0], [3.0]]],
    ])
    goal = torch.tensor([[1.0], [1.0]])
    costs = planning.compute_terminal_cost(
        predicted_latents=predictions,
        target_latent=goal,
        mode="latent",
    )
    torch.testing.assert_close(costs, torch.tensor([1.0, 2.0]))


def test_oracle_cost_requires_numeric_goal():
    with self.assertRaisesRegex(ValueError, "oracle_goal"):
        planning.compute_terminal_cost(
            predicted_latents=predictions,
            target_latent=goal,
            mode="oracle",
            position_probe=ScalarPositionProbe(),
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
        )
```

Add a probe-mode test proving that the target is decoded from `target_latent`, not taken from an oracle coordinate.

- [ ] **Step 11: Run cost tests and verify RED**

Expected: failure because `compute_terminal_cost` is absent.

- [ ] **Step 12: Implement terminal costs**

Decode only the last predicted horizon. For `probe`, decode both predicted and target-image latents and compute Euclidean distance. For `latent`, compute patch-token MSE. For `oracle`, decode predictions and compare with a required numeric goal. Reject missing probe statistics and unknown modes.

- [ ] **Step 13: Run Task 1 tests and commit**

Run `/Users/kmylonas/miniforge3/envs/ML/bin/python -m unittest tests.test_mylonas_ijepa_wm_planning` and expect PASS.

Commit only `.gitignore`, `planning.py`, and its test with message `feat: add world-model planning primitives`.

---

### Task 2: Cross-entropy optimizer

**Files:**
- Modify: `src/mylonas_ijepa_wm/planning.py`
- Modify: `tests/test_mylonas_ijepa_wm_planning.py`

**Interfaces:**
- Consumes: `score_fn(candidate_actions) -> costs`.
- Produces: `CEMResult` and `cem_optimize(...)`.

- [ ] **Step 1: Write failing validation tests**

Test invalid horizon, action dimension, population, elite count, iterations, minimum standard deviation, and bounds. Include `num_elites > population`.

- [ ] **Step 2: Run validation tests and verify RED**

Expected: failure because `cem_optimize` is absent.

- [ ] **Step 3: Add the result type and validation**

```python
@dataclass(frozen=True)
class CEMResult:
    action_sequence: torch.Tensor
    cost: float
```

The action sequence has `[horizon, action_dim]` and remains on the optimizer device.

- [ ] **Step 4: Write failing convergence and reproducibility tests**

Use a bounded quadratic with optimum `[0.35, -0.45]` repeated over two horizons. With population 512, 64 elites, six iterations, and seed 7, assert the solution is within absolute tolerance 0.08. Repeat with an independently seeded generator and assert identical actions and cost. Assert all actions respect bounds.

- [ ] **Step 5: Run convergence tests and verify RED**

Expected: validation passes but optimization does not converge because it is not implemented.

- [ ] **Step 6: Implement diagonal-Gaussian CEM**

Initialize mean zero and standard deviation one. Each iteration samples with the supplied `torch.Generator`, clamps to bounds, scores candidates, selects the lowest-cost elites, and updates mean and standard deviation. Clamp standard deviation to `min_std`. Return the best actually scored sequence from the final iteration.

- [ ] **Step 7: Run Task 2 tests and commit**

Run the complete planning test module and expect PASS. Commit `planning.py` and its test with message `feat: add CEM action optimizer`.

---

### Task 3: PointMaze runtime helpers

**Files:**
- Create: `src/mylonas_ijepa_wm/plan_pointmaze.py`
- Create: `tests/test_mylonas_ijepa_wm_pointmaze_planning.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: planning primitives from Tasks 1 and 2.
- Produces: `render_target_observation(...)`, `execute_macro_action(...)`, `initialize_mpc_history(...)`, `update_mpc_history(...)`, and `plan_action(...)`.

- [ ] **Step 1: Expose the PointMaze test file**

Add `!tests/test_mylonas_ijepa_wm_pointmaze_planning.py` to `.gitignore`.

- [ ] **Step 2: Write failing target-render tests**

Build a fake environment with `unwrapped.goal`, `point_env.data.qpos`, `point_env.data.qvel`, `point_env.set_state`, and `render`. Assert the target frame is rendered with the ball at the goal and original qpos/qvel are restored after success and after a render exception.

- [ ] **Step 3: Run target-render tests and verify RED**

Expected: import failure because `plan_pointmaze.py` is absent.

- [ ] **Step 4: Implement target rendering**

Implement `render_target_observation(env)` using copied state and `try/finally`. Replace the first two qpos values by `env.unwrapped.goal`, zero qvel for the target image, validate `[224, 224, 3]` uint8 output, and restore state unconditionally.

- [ ] **Step 5: Write failing macro-execution tests**

Pass a `[10]` macro action to a fake environment. Assert exactly five ordered 2D calls to `step`, with early exit on success, termination, or truncation. Reject other shapes.

- [ ] **Step 6: Implement macro execution**

Add a frozen `MacroStepResult` dataclass and `execute_macro_action(env, macro_action)`. Convert each action pair to NumPy FP32 before `env.step`.

- [ ] **Step 7: Write failing history tests**

```python
latent_history, action_history = initialize_mpc_history(z0, 3, 10)
latent_history, action_history = update_mpc_history(
    latent_history, action_history, a0, z1
)
latent_history, action_history = update_mpc_history(
    latent_history, action_history, a1, z2
)
torch.testing.assert_close(latent_history, torch.stack([z0, z1, z2]))
torch.testing.assert_close(action_history, torch.stack([a0, a1]))
```

- [ ] **Step 8: Implement immutable history helpers**

Initialization repeats the first latent three times and creates two zero actions. Updates drop the oldest latent/action and append the new real latent/executed action without mutating callers.

- [ ] **Step 9: Write a failing `plan_action` integration test**

Use a toy world model, probe, history, target latent, and small deterministic CEM configuration. Assert the returned first macro action has shape `[10]`, remains within bounds, and its result exposes the complete optimized sequence and cost.

- [ ] **Step 10: Implement `plan_action`**

Build a score closure around `rollout_action_sequences` and `compute_terminal_cost`, then invoke `cem_optimize`. Keep raw candidate actions separate from normalized model inputs.

- [ ] **Step 11: Run Task 3 tests and commit**

Run `/Users/kmylonas/miniforge3/envs/ML/bin/python -m unittest tests.test_mylonas_ijepa_wm_pointmaze_planning` and expect PASS without importing Gymnasium or MuJoCo.

Commit `.gitignore`, `plan_pointmaze.py`, and its test with message `feat: add PointMaze planning runtime`.

---

### Task 4: Frozen encoder, live MPC, CLI, and results

**Files:**
- Modify: `src/mylonas_ijepa_wm/plan_pointmaze.py`
- Modify: `tests/test_mylonas_ijepa_wm_pointmaze_planning.py`

**Interfaces:**
- Consumes: Task 3 helpers, `embed_frames.preprocess_frames`, I-JEPA `vit_huge`, and existing checkpoint loaders.
- Produces: `load_frozen_ijepa_encoder(...)`, `encode_frame(...)`, `run_planning_episode(...)`, `summarize_results(...)`, `parse_args(...)`, and `main(...)`.

- [ ] **Step 1: Write failing frozen-encoder tests**

Use an injected tiny encoder to verify `encode_frame` converts one uint8 HWC frame into one `[patches, dim]` tensor without gradients. Patch the constructor in a loader test and verify loading consumes only the checkpoint `encoder` mapping, sets evaluation mode, and disables parameters.

- [ ] **Step 2: Run encoder tests and verify RED**

Expected: failure because encoder helpers are absent.

- [ ] **Step 3: Implement frozen encoder helpers**

Instantiate only `vit.__dict__[model_name]`, remove `module.` prefixes from checkpoint encoder keys, set `eval()` and `requires_grad_(False)`, and encode under `torch.inference_mode()` using existing I-JEPA preprocessing.

- [ ] **Step 4: Write a failing end-to-end fake episode test**

Use deterministic fake environment, encoder, world model, probe, and planner injection. Verify an episode creates and encodes one target image, replans after each macro action, stops on success or the macro-step limit, records actions and achieved positions, and never passes simulator state to the world model.

- [ ] **Step 5: Implement the MPC episode loop**

Reset by seed, hide the marker, encode initial and target images, initialize history, plan and execute one macro action per cycle, encode the new real frame, update history, and stop according to the spec. Use achieved and desired coordinates only for result metrics.

- [ ] **Step 6: Write failing parser and summary tests**

```python
args = plan_pointmaze.parse_args([])
self.assertEqual(args.cost, "probe")
self.assertEqual(args.planning_horizon, 3)
self.assertEqual(args.population, 256)
self.assertEqual(args.num_elites, 32)
self.assertEqual(args.cem_iterations, 5)
self.assertEqual(args.candidate_batch_size, 16)
self.assertEqual(args.max_macro_steps, 20)
```

Test that latent mode does not require a probe, while probe and oracle modes do. Test success rate, mean and median final distance, and mean macro steps on three fixed records.

- [ ] **Step 7: Implement the CLI and JSON output**

Import Gymnasium packages only inside `create_environment`. Add all paths and configuration from the spec. Load the K=3 world model, encoder-only I-JEPA checkpoint, and probe conditionally. Run sequential seeded episodes, print progress, summarize records, and atomically save JSON through a temporary file followed by `replace`.

- [ ] **Step 8: Run all relevant project tests**

Run these unittest modules together: `test_mylonas_ijepa_position_probe`, `test_mylonas_ijepa_wm_vit`, `test_mylonas_ijepa_wm_training`, `test_mylonas_ijepa_wm_rollout_eval`, `test_mylonas_ijepa_wm_rollout_compare`, `test_mylonas_ijepa_wm_planning`, and `test_mylonas_ijepa_wm_pointmaze_planning`.

Expected: all tests pass.

- [ ] **Step 9: Run a one-episode live smoke test**

Run:

```bash
/Users/kmylonas/miniforge3/envs/pointmaze/bin/python -u -m src.mylonas_ijepa_wm.plan_pointmaze --num-episodes 1 --max-macro-steps 1 --planning-horizon 1 --population 4 --num-elites 2 --cem-iterations 1 --candidate-batch-size 2 --device cpu --output /tmp/ijepa_pointmaze_planning_smoke.json
```

Expected: one episode completes, writes valid JSON, executes one macro action, and never loads the I-JEPA predictor.

- [ ] **Step 10: Verify and commit Task 4**

Validate the JSON with `python -m json.tool`, run `git diff --check`, and commit `plan_pointmaze.py` plus its test with message `feat: run CEM planning in PointMaze`.
