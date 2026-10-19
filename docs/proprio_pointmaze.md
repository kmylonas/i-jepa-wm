# Top-down PointMaze with proprioception

The proprioceptive world model uses the existing top-down I-JEPA cache. Each
sample supplies three consecutive patch-token frames, the corresponding
`(x, y, vx, vy)` states, and macro actions. It predicts the next patch tokens
**and** next state. During planning, the current state comes from the
environment observation; subsequent states inside each imagined rollout come
from the world model. CEM scores candidates by distance between the predicted
terminal patch tokens and the encoded goal image. It does not use a position
probe.

The original `wm.py`, `vit.py`, and `plan_pointmaze.py` remain available for
vision-only experiments. Proprioceptive runs use separate checkpoints and
results. These checkpoints cannot be loaded by the vision-only planner.

## Required data

From the repository root, the default training paths are:

```text
src/mylonas_ijepa_wm/topdown/ijepa_cache/latents/episode_XXXX.npy
src/mylonas_ijepa_wm/topdown/ijepa_cache/trajectories/episode_XXXX.pt
```

Each latent file must contain `[21, 256, 1280]` tokens. Its trajectory sidecar
must contain `sampled_states` with shape `[21, 4]` and `macro_actions` with
shape `[20, 10]`. The four state values are position x/y and velocity x/y.
The code reads these files directly; raw frames and re-embedding are not
needed. Position and velocity normalization statistics are computed from the
training split only, then saved with the checkpoint. The default split is
1,600 train, 200 validation, and 200 test episodes from 2,000 total.
The local 2,000-episode cache was checked for file pairing, tensor shapes,
finite state/action values, and frame indices `0, 5, ..., 100`.

## Train

From the repository root:

```bash
python -u -m src.mylonas_ijepa_wm.wm_proprio \
  --latent-dir src/mylonas_ijepa_wm/topdown/ijepa_cache/latents \
  --trajectory-dir src/mylonas_ijepa_wm/topdown/ijepa_cache/trajectories \
  --output-dir src/mylonas_ijepa_wm/topdown/world_models/proprio_v1 \
  --device cuda --precision auto
```

`best.pt`, `last.pt`, and `test_metrics.pt` are written to the output
directory. The default loss is one-step visual-token MSE plus normalized-state
MSE (weight `1.0`).
Validation selects `best.pt` using that combined loss;
the individual terms are also saved in the checkpoint history. The state-loss
weight is a tunable scale, not a guarantee that the model will accurately
predict position and velocity. Inspect both validation terms before planning.

To continue the same run, add `--resume --epochs NEW_TOTAL_EPOCHS`. To start
a separate three-step rollout fine-tuning run, use a new output directory:

```bash
python -u -m src.mylonas_ijepa_wm.wm_proprio \
  --latent-dir src/mylonas_ijepa_wm/topdown/ijepa_cache/latents \
  --trajectory-dir src/mylonas_ijepa_wm/topdown/ijepa_cache/trajectories \
  --output-dir src/mylonas_ijepa_wm/topdown/world_models/proprio_v1_k3 \
  --finetune-from src/mylonas_ijepa_wm/topdown/world_models/proprio_v1/best.pt \
  --rollout-steps 3 --rollout-loss-weight 1.0 \
  --device cuda --precision auto
```

You can pass other existing training flags, such as `--batch-size`,
`--epochs`, `--learning-rate`, and `--compile`; run
`python -m src.mylonas_ijepa_wm.wm_proprio --help` for the full list. Training
the full model is intended for a CUDA GPU. A small CPU smoke test works, but
the full 2,000-episode run is not practical on a laptop CPU.

## Plan with latent cost

Use the same frozen I-JEPA encoder checkpoint used to make the cache:

```bash
python -u -m src.mylonas_ijepa_wm.plan_pointmaze_proprio \
  --encoder-checkpoint /path/to/ijepa_encoder.pt \
  --world-model-checkpoint src/mylonas_ijepa_wm/topdown/world_models/proprio_v1/best.pt \
  --output src/mylonas_ijepa_wm/topdown/world_models/proprio_v1/planning_latent.json \
  --video-dir src/mylonas_ijepa_wm/topdown/world_models/proprio_v1/videos \
  --num-episodes 20 --base-seed 0 --max-macro-steps 20 \
  --planning-horizon 3 --population 256 --num-elites 32 \
  --cem-iterations 5 --candidate-batch-size 64 \
  --device cuda --precision auto --compile
```

The camera and cost are fixed to top-down and latent respectively. No probe
checkpoint is required. Omit `--video-dir` if videos are unnecessary. The
JSON records success, actions, achieved positions, and the measured
proprioceptive states at each replan.

Because the planner receives true position and velocity, its success rate is
not directly comparable to the previous vision-only setup. A comparison
should use the same top-down episodes, encoder, planning budget, and goal
cost, with proprioception as the only changed input.

## Evaluate visual and proprioceptive rollouts

The proprioceptive rollout evaluator plots three panels at each horizon:
visual latent MSE, normalized `(x, y, vx, vy)` MSE, and their weighted sum.
Each panel compares teacher-forced predictions (real visual and state history
at every step), autoregressive predictions (predicted visual and state
histories after the initial context), and persistence (both last observed
visual latent and last observed state held fixed). The combined value is
`latent MSE + proprio_loss_weight * state MSE`, with the weight read from the
checkpoint. All methods use the same episode windows. This per-horizon
diagnostic is distinct from the training validation loss, which also averages
predictions over the model's history positions.

The evaluator uses the checkpoint's saved split IDs and action/state
normalization. At each horizon, visual MSE is averaged over all windows,
patches, and embedding dimensions; state MSE is averaged over all windows and
four normalized state components. It does not require a position probe or the
I-JEPA encoder.

```bash
python -u -m src.mylonas_ijepa_wm.evaluate_rollouts_proprio \
  --checkpoint /path/to/proprio_v1/best.pt \
  --latent-dir /path/to/topdown/ijepa_cache/latents \
  --trajectory-dir /path/to/topdown/ijepa_cache/trajectories \
  --split test --horizon 10 --batch-size 4 \
  --output /path/to/proprio_v1/rollout_test_h10_metrics.json \
  --plot-output /path/to/proprio_v1/rollout_test_h10_metrics.png \
  --device cuda --precision auto
```

With the default three-frame history, the 21-frame cached episodes support
at most 18 future macro steps. A horizon of 10 uses nine overlapping windows
per test episode. The figure and JSON are saved separately so the numerical
values can be inspected or replotted later. On Colab, the cache may be read
from Drive directly; copying just the checkpoint's test episodes to local
storage is an optional speedup, and the full training cache is unnecessary.

## Inspect individual state rollouts

Use this separate command to plot only a few cached episodes. By default it
uses the first four IDs in the checkpoint's test split and the first window
of each episode, matching the older vision-only trajectory figure. The plot
shows real, teacher-forced, and autoregressive positions in maze coordinates;
velocity arrows appear at the middle and final horizons. The arrow scale is
labeled on the figure. The saved JSON contains all four unnormalized state
components at every plotted horizon.

```bash
python -u -m src.mylonas_ijepa_wm.visualize_proprio_rollouts \
  --checkpoint /path/to/proprio_v1/best.pt \
  --latent-dir /path/to/topdown/ijepa_cache/latents \
  --trajectory-dir /path/to/topdown/ijepa_cache/trajectories \
  --split test --horizon 10 --num-episodes 4 \
  --output /path/to/proprio_v1/rollout_test_h10_state_trajectories.png \
  --device mps --precision fp32
```

Use `--episode-ids ID1 ID2 ...` to inspect specific members of the selected
checkpoint split. Each prediction starts from the last of three real history
frames and uses the recorded macro actions. This is an open-loop diagnostic
on cached trajectories, not a closed-loop planning rollout.
