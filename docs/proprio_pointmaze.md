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
