# DotWall layout generalization runbook

This experiment trains one proprioceptive I-JEPA world model on nine Wall
layouts, selects its checkpoint on two other layouts, and tests on four layouts
with wall and door positions absent from both training and validation. Wall
thickness and doorway width stay fixed. The [design](superpowers/specs/2026-10-24-wall-layout-generalization-design.md)
defines the scientific question and metrics.

Run commands from the `ijepa` repository root. Set these paths for your
machine or Colab session:

```bash
export DINO_ROOT=/absolute/path/to/pinned/dino_wm/source
export ENCODER=/absolute/path/to/IN1K-vit.h.14-300e.pth.tar
export FIXED_WM=/absolute/path/to/wall_fixed_3/world_models/k1/best.pt
export EXP_ROOT=/absolute/path/to/wall_layout_generalization
export RAW_DIR="$EXP_ROOT/raw"
export CACHE_DIR="$EXP_ROOT/cache_dense"
export RUN_DIR="$EXP_ROOT/world_model"
export MANIFEST="$EXP_ROOT/episode_manifest.json"
export ENV_CONFIG="$EXP_ROOT/environment.json"
```

The local reference checkout in this project is
`data/dino_wm_reference/source`, pinned at DINO-WM commit
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`. The I-JEPA checkpoint is
under `src/checkpoints/` on this machine. These data files are not in Git or
the isolated worktree; point the variables to their actual locations.

## Pilot before a large embedding run

Create a 10-episode-per-layout pilot (90 train, 20 validation, 40 test). The
manifest records each episode's seed, split, layout, and collection policy.
Each layout alternates wide correlated actions, uniformly covering lengths
0.2–3.5, with the doorway-directed policy. The original `dino` collector
retains its 0.2–1.8 range.

```bash
export PILOT_ROOT="$EXP_ROOT/pilot"
python -m src.mylonas_ijepa_wm.wall_layout_manifest \
  --output-dir "$PILOT_ROOT" --dino-wm-root "$DINO_ROOT" \
  --train-episodes 90 --val-episodes 20 --test-episodes 40
python -m src.mylonas_ijepa_wm.generate_episodes \
  --env wall --env-config "$PILOT_ROOT/environment.json" \
  --episode-manifest "$PILOT_ROOT/episode_manifest.json" \
  --output-dir "$PILOT_ROOT/raw" --steps-per-episode 50
python -m src.mylonas_ijepa_wm.audit_wall_dataset \
  --raw-dir "$PILOT_ROOT/raw" \
  --episode-manifest "$PILOT_ROOT/episode_manifest.json" \
  --output "$PILOT_ROOT/coverage.json"
```

Inspect each layout's solid contacts, doorway crossings in both directions,
and wide-policy actions above 1.8 and 3.0. Pilot counts are diagnostic; the
full-data thresholds below are not expected from ten episodes per layout.

### Local pilot result (October 24, 2026)

The local 150-episode pilot collected successfully and used about 1.1 GB for
raw files. Across the nine training layouts it recorded 360 solid-wall
contact transitions, 53 left-to-right and 66 right-to-left doorway crossings,
and 348 wide-policy actions longer than 3.0. The `[40, 42]` training layout
had zero solid contacts in its ten pilot episodes. A separate, unsaved
100-episode wide-policy replay at `[40, 42]` with seeds 1000–1099 found
contacts in 64 episodes (939 contact transitions), indicating the ten-episode
zero was sampling variance. The doorway policy contributed zero solid-wall
contacts in that replay, as expected from its steering objective. The full
dataset must still pass its own coverage audit before embedding.

A six-frame smoke sample from pilot episode 0 embedded on local MPS using
the frozen I-JEPA checkpoint. Its tokens were `(6, 256, 1280)` float16;
the trajectory sidecar contained six aligned `(x, y)` states, frame indices
`0..5`, the seed, and the manifest hash. This verifies alignment, not WM
training or held-out performance.

## Full collection and embedding

```bash
python -m src.mylonas_ijepa_wm.wall_layout_manifest \
  --output-dir "$EXP_ROOT" --dino-wm-root "$DINO_ROOT"
python -m src.mylonas_ijepa_wm.generate_episodes \
  --env wall --env-config "$ENV_CONFIG" --episode-manifest "$MANIFEST" \
  --output-dir "$RAW_DIR" --steps-per-episode 50 --resume
python -m src.mylonas_ijepa_wm.audit_wall_dataset \
  --raw-dir "$RAW_DIR" --episode-manifest "$MANIFEST" \
  --output "$EXP_ROOT/coverage.json" --require-coverage
```

The default manifest assigns 2,000 train, 200 validation, and 400 test
episodes. Do not embed until every training layout has at least 25 solid
contacts, 25 doorway crossings, five in each direction, and some wide-policy
actions above length 3.0. If the pilot indicates a shortfall, adjust the
collection design before generating the full manifest. If the full audit
fails, keep the raw data and revise the manifest and collection before
embedding; a manifest change requires regenerating or updating its recorded
identity in raw files and cache sidecars.

To collect or embed in batches, pass `--start-index A --end-index B --resume`
to the collector and embedder. Keep the same manifest and environment config
for all batches. The full raw set must exist when the audit runs.

```bash
python -m src.mylonas_ijepa_wm.embed_frames \
  --env wall --env-config "$ENV_CONFIG" --input-dir "$RAW_DIR" \
  --output-dir "$CACHE_DIR" --checkpoint-path "$ENCODER" \
  --embedding-frame-stride 1 --start-index 0 --end-index 2600 \
  --encoder-batch-size 4 --resume
```

At 2,600 episodes, the 51-frame float16 latent cache is roughly 87 GB before
filesystem overhead; raw RGB frames are roughly 20 GB. Ensure storage is
available. The embedder validates manifest identity and frame stride when it
resumes. Raw batches can be removed after a successful full audit and their
latent files and trajectory sidecars have been verified.

The local Mac has MPS but no CUDA, so the full training and planning matrix
were not run here. Local verification passed 222 tests and 16 subtests after
excluding four unrelated legacy test modules that need `torchvision` or a
different package import path in this Python environment. All nine relevant
CLI `--help` checks passed. The remaining test warning concerns unavailable
GLFW initialization in a camera test.

## Train and inspect predictions

Run the full training command on CUDA. It starts the world model from random
weights; the encoder stays frozen. The explicit manifest replaces random
episode splitting. Position and action normalization use train IDs only.

```bash
python -m src.mylonas_ijepa_wm.wm_proprio \
  --env wall --env-config "$ENV_CONFIG" --episode-manifest "$MANIFEST" \
  --latent-dir "$CACHE_DIR/latents" \
  --trajectory-dir "$CACHE_DIR/trajectories" \
  --output-dir "$RUN_DIR" --num-hist 1 --rollout-steps 1 \
  --epochs 100 --batch-size 4 --device cuda --precision auto
python -m src.mylonas_ijepa_wm.evaluate_rollouts_proprio \
  --checkpoint "$RUN_DIR/best.pt" --latent-dir "$CACHE_DIR/latents" \
  --trajectory-dir "$CACHE_DIR/trajectories" --split test \
  --horizon 5 --by-wall-layout --device cuda \
  --output "$RUN_DIR/heldout_rollouts.json"
python -m src.mylonas_ijepa_wm.evaluate_wall_collision_probes \
  --checkpoint "$RUN_DIR/best.pt" --encoder-checkpoint "$ENCODER" \
  --env-config "$ENV_CONFIG" --episode-manifest "$MANIFEST" \
  --output "$RUN_DIR/heldout_collision_probes.json" --device cuda
```

Read state and token error per horizon and layout, including rollouts whose
first predicted macro step contains a solid contact or doorway crossing.
Later horizons in those subsets measure prediction after that event. In
collision probes, compare predicted with
actual crossing at the new wall, through its door, and through an old
training-wall coordinate that is now open. Success in planning alone does
not establish that the model follows the visible geometry.

## Matched planning

```bash
python -m src.mylonas_ijepa_wm.run_wall_planning_matrix \
  --episode-manifest "$MANIFEST" --env-config "$ENV_CONFIG" \
  --encoder-checkpoint "$ENCODER" \
  --multi-checkpoint "$RUN_DIR/best.pt" \
  --fixed-checkpoint "$FIXED_WM" \
  --output-root "$EXP_ROOT/planning" --base-seed 10000 \
  --num-episodes 20 --device cuda --resume
```

The runner executes the multi-layout model and historical fixed-layout model
on each test layout with identical 20 start–goal seeds. Both use position
cost, horizon 5, five actions per plan, population 300, 30 elites, ten CEM
iterations, and at most 20 WM steps. It tests unbounded actions, Euclidean
primitive-action caps of 4.0, and the training maximum of 3.5. Capped modes
set `--unbounded-actions` before applying the norm cap, avoiding the adapter's
`[-1, 1]` component box. JSON output, videos, and per-layout Wilson intervals
are stored under `planning/`. Each result has an identity sidecar with hashes
of its model, encoder, and manifest; resume checks these before reusing a
result. Treat the fixed checkpoint as a historical
transfer reference because its collection policy and training layouts differ.

For a training-layout sanity check, run `plan_proprio` separately with a
single-layout config for `[24, 18]` and `[40, 42]`, using the same seeds and
the 3.5 cap. Keep these outputs separate from the held-out test matrix.

## Test whether the model uses the visual layout

The original collision probes change the agent's starting position with the
probe location. The matched visual test holds its starting position and five
actions fixed while changing the held-out wall or doorway layout. Four
predeclared comparisons cover doorway and wall-position changes in both
crossing directions. Their simulator outcomes must contrast; the command
fails if a pair stops contrasting. The model receives no layout label.

```bash
python -m src.mylonas_ijepa_wm.evaluate_wall_visual_pairs \
  --checkpoint "$RUN_DIR/best.pt" --encoder-checkpoint "$ENCODER" \
  --episode-manifest "$MANIFEST" --env-config "$ENV_CONFIG" \
  --output "$EXP_ROOT/results/visual_input_pairs.json" \
  --device cuda --precision auto
```

For Colab after cloning the `wall-layout-generalization` branch and defining
the same `RUN_DIR`, `ENCODER_LOCAL`, `MANIFEST`, `ENV_CONFIG`, `LOCAL_ROOT`, and
`RESULTS_DIR` paths used above, this cell updates the source, points the
saved environment configuration at the current pinned DINO-WM checkout, and
runs the test:

```python
import json
import subprocess
import sys
from pathlib import Path

repo = Path("/content/i-jepa-wm")
dino_root = Path("/content/dino_wm")
subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
assert (dino_root / "env/wall/envs/wall.py").is_file()
assert ENCODER_LOCAL.is_file() and (RUN_DIR / "best.pt").is_file()

local_config = LOCAL_ROOT / "environment_visual_pairs.json"
local_config.parent.mkdir(parents=True, exist_ok=True)
settings = json.loads(ENV_CONFIG.read_text())
settings["dino_wm_root"] = str(dino_root)
local_config.write_text(json.dumps(settings, indent=2))

output = RESULTS_DIR / "visual_input_pairs.json"
subprocess.run([
    sys.executable, "-u", "-m", "src.mylonas_ijepa_wm.evaluate_wall_visual_pairs",
    "--checkpoint", str(RUN_DIR / "best.pt"),
    "--encoder-checkpoint", str(ENCODER_LOCAL),
    "--episode-manifest", str(MANIFEST),
    "--env-config", str(local_config),
    "--output", str(output),
    "--device", "cuda", "--precision", "auto",
], cwd=repo, check=True)
```

This command reads no episode caches. It saves four PNGs under
`visual_input_pairs_figures/` next to the JSON. Each PNG has two panels with
identical full-frame 65×65 axes. The solid wall shows the current layout's
real width and door opening. Faint dashed rails show all nine *separate*
training layouts: wall centers x=24, 32, 40 and door centers y=18, 30, 42.
The three rails at each x are shifted slightly sideways so their distinct
door openings remain visible; those shifts are only a plotting device. The
blue path is the actual five-action ball trajectory, and the orange X is
the world model's single predicted endpoint. The purple dotted line is a
virtual scoring threshold, not a wall. A case counts as crossing when its
endpoint passes that x coordinate, which may differ from the current wall x.
The JSON reports crossing agreement, endpoint error, and whether both
contrasting cases in each pair were predicted correctly.

If only the figure style changes, the saved JSON contains everything needed
to redraw the PNGs without loading the encoder or world model. In Colab,
update the repository and run:

```python
subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"], check=True)
from src.mylonas_ijepa_wm.evaluate_wall_visual_pairs import save_pair_figure

results = json.loads((RESULTS_DIR / "visual_input_pairs.json").read_text())
for pair in results["pairs"]:
    save_pair_figure(
        pair, RESULTS_DIR / "visual_input_pairs_figures" / f"{pair['name']}.png"
    )
```

Correct contrasting predictions with the same position and action support
visual-input dependence in these cases. They do not establish that every
layout feature is encoded or that planning always depends on the visual input.
