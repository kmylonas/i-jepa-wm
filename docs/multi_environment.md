# Environment selected proprioceptive world models

The new commands use `--env pointmaze` or `--env wall` across collection,
embedding, training, and planning. PointMaze remains the default in the old
scripts. The new `plan_proprio` command shares the existing model and CEM
implementation but chooses a runtime adapter by environment.

## Configurations

Use [pointmaze_physics5.json](../configs/pointmaze_physics5.json) for the
existing top-down Physics5 setting. The default PointMaze configuration is
Physics1 for compatibility with older caches.

For Wall, copy [wall_example.json](../configs/wall_example.json) and set
`dino_wm_root` to a local checkout of
[DINO-WM](https://github.com/gaoyuezhou/dino_wm). Wall uses the original
`DotWall` motion and collision code; the adapter loads only that module to
avoid importing the unrelated D4RL environments. Install its Wall dependencies
(`gym`, `scipy`, `omegaconf`, `pyyaml`) in the runtime. Each episode selects one
`[wall_x, door_y]` pair from `wall_layouts` using its seed. Set a single pair
to use a fixed layout. Wall states are `(x, y)`; PointMaze states are
`(x, y, vx, vy)`. Both use 2D primitive actions. Wall images use the DINO-WM
RGB rendering resized to 224 × 224.
Planning videos draw a green goal ring for inspection; encoder inputs do not.

## Pipeline

Run these commands from the `ijepa` repository root, with your paths in place
of the shell variables. Use separate raw, cache, and run directories for each
environment/configuration.

```bash
python -m src.mylonas_ijepa_wm.generate_episodes \
  --env wall --env-config "$ENV_CONFIG" --output-dir "$RAW_DIR" \
  --num-episodes 2000 --steps-per-episode 100 --wall-policy mixed --resume

python -m src.mylonas_ijepa_wm.embed_frames \
  --env wall --env-config "$ENV_CONFIG" --input-dir "$RAW_DIR" \
  --output-dir "$CACHE_DIR" --checkpoint-path "$ENCODER_CHECKPOINT" \
  --start-index 0 --end-index 2000 --encoder-batch-size 4 --resume

python -m src.mylonas_ijepa_wm.wm_proprio \
  --env wall --env-config "$ENV_CONFIG" \
  --latent-dir "$CACHE_DIR/latents" --trajectory-dir "$CACHE_DIR/trajectories" \
  --output-dir "$RUN_DIR" --num-episodes 2000 \
  --train-episodes 1600 --val-episodes 200 --test-episodes 200 \
  --epochs 100 --batch-size 4 --rollout-steps 1 --device cuda --precision auto

python -m src.mylonas_ijepa_wm.plan_proprio \
  --env wall --env-config "$ENV_CONFIG" \
  --encoder-checkpoint "$ENCODER_CHECKPOINT" \
  --world-model-checkpoint "$RUN_DIR/best.pt" \
  --cost latent --output "$PLAN_JSON" --video-dir "$VIDEO_DIR" \
  --num-episodes 20 --planning-horizon 5 --actions-per-plan 5 \
  --population 300 --num-elites 30 --cem-iterations 10 --device cuda
```

Replace `wall` with `pointmaze` and use the PointMaze config for a Physics5
dataset. The embedding stage stores one I-JEPA latent per model step and
flattens each group of primitive actions. The trainer checks environment,
state fields, action dimensions, timing, and recorded configuration before
training. Checkpoints store the environment and the same dimensions; the
planner rejects a checkpoint from another environment. A Wall planning config
may specify a different layout for transfer testing, provided the state and
action dimensions still match.

### Training windows starting at every environment action

The default cache above embeds frames 0, 5, 10, ... for a five-action model
step. To train with windows starting at every action, embed the **same raw
episodes** into a separate cache using `--embedding-frame-stride 1`:

```bash
python -m src.mylonas_ijepa_wm.embed_frames \
  --env wall --env-config "$ENV_CONFIG" --input-dir "$RAW_DIR" \
  --output-dir "$DENSE_CACHE_DIR" --checkpoint-path "$ENCODER_CHECKPOINT" \
  --embedding-frame-stride 1 --start-index 0 --end-index 2000 \
  --encoder-batch-size 4 --resume
```

Then point `wm_proprio` (or the visual-only `wm`) at
`$DENSE_CACHE_DIR/latents` and `$DENSE_CACHE_DIR/trajectories`. No trainer flag
is needed: cache metadata selects the window spacing. For example, a Wall
episode with 50 actions and a history of 3 yields training windows starting
at frames 0 through 35. One such window reads latent frames 1, 6, 11, 16
and groups raw actions 1–5, 6–10, 11–15. The predictor still advances five
environment actions per model step. Rollout evaluation accepts the same cache.
Planning uses online images and the trained checkpoint, so its command does
not change.

This follows DINO-WM's one-action sliding-window principle; the current
loader keeps every window with a valid target, including the final few
windows that DINO-WM's slicer omits. The dense cache requires the raw frames
and roughly five times the embedding compute and latent storage of the
sparse cache. Existing sparse caches cannot supply the missing intermediate
frames. Keep the two cache directories separate; `--resume` checks their
embedding stride before skipping a file.

For fixed-layout Wall training, use `--wall-policy mixed`: even-indexed
episodes use random actions and odd-indexed episodes steer through the door,
then toward the opposite side. The doorway policy uses the true wall and door
coordinates only for **data collection**; the world model receives frames,
states, and actions, without waypoints. `--wall-policy random` retains the
original collector. In a 100-episode fixed-layout smoke check, only 1 random
episode crossed the wall; all 100 doorway-directed episodes did. Check your
own crossing coverage before starting a full embedding run. The Wall adapter
uses fixed `wall_action_limit` bounds, while DINO-WM's own dataset generator
uses a different action sampling policy.

The collector also accepts `--start-index` and `--end-index`. These ranges let
you collect and embed batches, save their cache files, and remove raw batch
files before collecting the next range. Keep `--num-episodes` and all settings
unchanged across batches, and pass `--resume` after the first batch.
