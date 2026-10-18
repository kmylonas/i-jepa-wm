# PointMaze world modeling with frozen I-JEPA — project handoff

This file is the starting point for a new chat or collaborator. It describes the
custom PointMaze work in this repository, **not** the upstream I-JEPA pretraining
project described in `README.md`. The earlier MIT States and Something-Something
experiments are no longer part of the active project.

## Research question

Can a world model trained on **frozen I-JEPA image representations** predict the
visual consequences of actions well enough to plan in PointMaze, without using
the simulator's position or velocity as model inputs? This is inspired by
DINO-WM, but uses the official ImageNet-1K I-JEPA ViT-H/14 **encoder** in place
of DINOv2. The original I-JEPA predictor is not used; our action-conditioned
transformer is trained separately.

The simulator state is retained for supervision of a *diagnostic* position
probe and for evaluation. During world-model training, the model receives
I-JEPA tokens and actions, not state. During planning, the controller uses
rendered observations, actions, and a goal coordinate; the goal cost usually
uses positions decoded from predicted tokens by the probe. The environment's
true position is not passed into that cost in the default `--cost probe` mode.

## Repositories, code, and experiment files

- `../point-maze/random_episodes.py` generates raw PointMaze episodes. This is
  a separate sibling Git repository, not part of the I-JEPA repository.
- Shared experiment code lives directly in `src/mylonas_ijepa_wm/`. The
  `angled/` and `topdown/` subdirectories organize **artifacts by camera**;
  they are not Python packages or alternate implementations of the scripts.
- `src/mylonas_ijepa_wm/embed_frames.py` runs the frozen encoder and writes the
  latent cache and trajectory sidecars.
- `src/mylonas_ijepa_wm/vit.py` defines the action-conditioned transformer.
- `src/mylonas_ijepa_wm/wm.py` trains the world model, including one-step
  training and optional multi-step rollout fine-tuning.
- `src/mylonas_ijepa_wm/position_probe.py` trains a spatial-softmax position
  probe on frozen patch tokens; evaluation also applies it to WM predictions.
- `src/mylonas_ijepa_wm/evaluate_rollouts.py` evaluates teacher-forced and
  autoregressive rollouts. `compare_rollout_models.py` compares checkpoints.
- `src/mylonas_ijepa_wm/visualize_position_probe.py` makes parity, trajectory,
  and attention figures. `evaluate_position_probe_grid.py` maps position error
  across the maze.
- `src/mylonas_ijepa_wm/planning.py` implements CEM; `plan_pointmaze.py` runs
  closed-loop planning, videos, and failure diagnostics.
- The Colab notebooks used for the most recent dataset and training runs were
  supplied separately as `point_maze_random_episodes.ipynb`, `ijepa_cache.ipynb`,
  and `pointmaze_wm.ipynb` (currently in the user's Downloads directory, **not
  tracked by this repository**).

The current local artifact layout, relative to the I-JEPA repository root, is:

```text
src/mylonas_ijepa_wm/
├── angled/
│   ├── raw/
│   ├── ijepa_cache/{latents,trajectories}/
│   ├── position_probe_runs/{spatial_softmax,spatial_softmax_mlp}/
│   └── world_models/{full_default_v1,full_deafult_v1_k3}/
└── topdown/
    ├── raw/
    ├── ijepa_cache/{latents,trajectories}/
    ├── position_probe_runs/
    │   ├── spatial_softmax_linear/
    │   ├── spatial_softmax_mlp/
    │   ├── spatial_softmax_mlp 2/
    │   ├── spatial_softmax_mlp 3/
    │   └── spatial_softmax_mlp 4/
    └── world_models/{full_deafult_v1,full_deafult_v1_k3}/
```

The `full_deafult_*` names really are misspelled on disk. The top-down
`world_models/` directories currently contain no checkpoint files locally;
the trained top-down WM checkpoints have been used from Google Drive. The
latest locally copied top-down probe is
`topdown/position_probe_runs/spatial_softmax_mlp 4/best.pt`. Directory names
ending in ` 2`, ` 3`, or ` 4` contain spaces and must be quoted in shell
commands. Only a small subset of raw/cache files is present locally; the full
2,000-episode cache and top-down WM runs are on Drive/Colab. These artifacts
are untracked, so do not assume a fresh clone contains them.

Run the following module entry points from the **I-JEPA repository root**;
each exposes its current CLI arguments with `--help`:

```bash
python -m src.mylonas_ijepa_wm.embed_frames --help
python -m src.mylonas_ijepa_wm.wm --help
python -m src.mylonas_ijepa_wm.position_probe --help
python -m src.mylonas_ijepa_wm.evaluate_rollouts --help
python -m src.mylonas_ijepa_wm.plan_pointmaze --help
```

Generate raw episodes from the separate `point-maze` repository with
`python random_episodes.py --help`. The actual training notebooks pass
explicit Drive and Colab-local paths rather than relying on CLI defaults.
The CLI defaults do **not** automatically select `angled/` or `topdown/`:
always pass the appropriate latent, trajectory, output, and checkpoint paths
for a camera-specific run. Do not mix a top-down encoder/cache with an angled
world model or position probe.

## Data lineage and alignment

The environment is `PointMaze_UMaze-v3`. The newer dataset uses a 224 × 224
RGB **top-down** camera (`distance=7`, `azimuth=90`, `elevation=-90`); the
earlier dataset uses the environment's angled camera. In the newer dataset,
the red goal marker is hidden in frames used for encoding, but the goal
coordinate is still stored. Each of 2,000 random episodes has 100 environment
actions:

| Raw episode field | Shape | Meaning |
| --- | --- | --- |
| `frames` | `[101, 224, 224, 3]`, `uint8` | Initial image, then image after each raw action |
| `actions` | `[100, 2]`, `float32` | Force commands in x/y |
| `states` | `[101, 4]`, `float32` | Position x/y and velocity x/y; diagnostic labels only |
| `desired_goal` | `[2]` | Goal position |

The cache takes frames `0, 5, 10, ..., 100`, converts pixels from `uint8`
`[0,255]` to floating point `[0,1]`, applies ImageNet channel normalization,
and encodes them with the frozen I-JEPA ViT-H/14. One episode produces:

| Cached file/field | Shape | Meaning |
| --- | --- | --- |
| `latents/episode_XXXX.npy` | `[21, 256, 1280]`, `float16` | 256 patch tokens per sampled image |
| `trajectories/episode_XXXX.pt`: `sampled_states` | `[21, 4]` | `states[0], states[5], ..., states[100]` |
| `macro_actions` | `[20, 10]` | Five consecutive 2-D raw actions, flattened |
| `frame_indices` | `[21]` | `0, 5, ..., 100` |

At cached index `k`, latent `k` represents raw frame `5k`, the probe target is
raw state `5k`, and macro action `k` contains raw actions `5k` through `5k+4`
and leads to latent `k+1`. Raw frames and states are both captured after each
environment step. Both training datasets pair files by the same episode ID.

The standard split is a seeded (`42`) random permutation of episode IDs:
1,600 train, 200 validation, 200 test. A dataset sample is a short window
*within one episode*, not a batch of whole episodes. The default world-model
history has three cached frames spaced five raw steps apart, spanning ten raw
steps from the first observed frame to the last.

### Alignment audit already performed

Code inspection found no off-by-one error in frame/state/action indexing. The
top-down Colab embedding run processed all 2,000 episodes with zero skipped
outputs, and the training notebook copied 2,000 latent files plus 2,000
trajectory sidecars. The probe and WM used the same copied cache. On Drive,
raw episodes 0000, 0999, and 1999 were directly compared with their trajectory
sidecars; frame/state/action sampling and alignment checks passed. Saved raw
RGB frames from those episodes were displayed and were visibly top-down—not
redrawn maze diagrams.

Raw frames 0, 50, and 100 from each of those three episodes were independently
re-encoded with the frozen checkpoint and compared with the cached tokens.
Each re-encoding's nearest cached frame was the expected index 0, 10, or 20;
mean absolute token differences were about `1.6e-5`–`2.1e-5`, much smaller
than the second-best match. Float16 storage and runtime arithmetic are
consistent with the small differences; their precise contribution was not
isolated. These spot checks strongly support top-down
cache provenance and alignment, but do not constitute a bytewise audit of
every episode.

## World model and probe

The world model consumes three frames of tokens `[B, 3, 256, 1280]` and
three associated macro actions `[B, 3, 10]`. It embeds each action, broadcasts
it across that frame's patches, projects into a smaller transformer width,
uses spatial/temporal positional embeddings and a block-causal attention mask,
then projects predictions back to 1,280-dimensional patch tokens. A common
configuration is width 512, four blocks, eight heads, and MLP width 1,024.

The initial model is trained for one-step token prediction. A later run starts
from that checkpoint and adds a three-step autoregressive rollout loss
(`--rollout-steps 3 --rollout-loss-weight 1.0`). `wm.py` saves `last.pt` for
epoch-level resume and `best.pt` for the lowest validation loss. `--resume`
continues training; `--finetune-from` copies weights into a fresh run.

The diagnostic probe predicts `(x,y)` from a single frame's 256 frozen tokens.
Its patch scorer is either linear or a small MLP. A softmax over patch scores
produces a weighted image-grid location, which a final 2-to-2 layer maps to
maze coordinates. This is a *constrained localizer*, not a general MLP over
all 256 × 1,280 input features. The scorer's attention heatmap is informative
but not definitive: final I-JEPA patch tokens contain global context, so a
patch away from the ball can still carry information about its position.

## Planning

`plan_pointmaze.py` now configures the same top-down camera as the generator.
It loads only the pretrained I-JEPA **encoder**, plus the trained world model
and position probe. CEM samples candidate sequences of macro actions; each
macro action has five raw 2-D controls. The default planner uses a population
of 256, 32 elites, five CEM iterations, and a horizon of three macro actions.
It predicts future tokens, decodes positions, and scores them by distance to
`desired_goal` (`--cost probe`). It executes **only the first macro action**
(five raw controls), observes a new real image, and replans. Optional videos
and final-plan diagnostic figures help inspect failures. Videos may show the
red goal for human visualization even though encoded observations hide it.

## Results so far — keep camera conditions separate

**Earlier angled-camera dataset and models:**

- The spatial-softmax MLP probe's best checkpoint had test mean Euclidean
  position error **0.0682** maze units versus a mean-position baseline of
  **1.2793**. Test R² was about **0.988 x / 0.998 y**.
- The K=3 fine-tuned WM still accumulated error autoregressively. At a
  10-macro-step rollout horizon, probe-decoded position error was **0.2434**
  for autoregressive predictions, **0.0927** for teacher-forced predictions,
  and **0.3956** for persistence. These are from the saved angled-camera
  rollout evaluation, not the newer top-down model.
- An initial 20-episode closed-loop planning trial was reported as **65%**
  successful. Inspection of failed videos showed attempts to approach the
  goal directly despite an intervening wall. The WM's plotted selected-plan
  endpoints did not obviously pass through walls; this does not, by itself,
  establish whether the main failure is model error or short-horizon planning.
- The angled camera could hide the ball behind a central wall. This motivated
  changing both raw-data rendering and live planning observations to top-down.

**New top-down dataset and models:**

- The same spatial-softmax MLP probe was resumed repeatedly on the top-down
  tokens. Locally copied `spatial_softmax_mlp/`, `spatial_softmax_mlp 2/`,
  `spatial_softmax_mlp 3/`, and `spatial_softmax_mlp 4/` have best-checkpoint
  test mean position errors of **0.1398**, **0.0775**, **0.0646**, and
  **0.0513** maze units respectively. The latest checkpoint is one-based
  epoch **200**, with test R² approximately **0.9979 x / 0.9980 y**. It
  outperforms the angled probe's **0.0682** global mean error, although some
  individual positions and trajectories still show visible error. The
  earlier claim that the top-down features are not position-decodable is not
  supported by this result.
- On an inspected episode, the angled probe concentrated attention near the
  ball, while the top-down probe's strongest attention stayed on background
  patches even after further training. These are scores over *contextualized*
  I-JEPA tokens, not a literal pixel segmentation; off-ball attention alone
  does not show that the encoder ignored the ball.
- The top-down WM's reported one-step and K=3 training losses were higher in
  absolute value than the angled run's. Raw token MSE is not directly
  comparable between camera views because the target embeddings change.
  The latest reported top-down K=3 test rollout used 200 episodes and 1,800
  windows, with the epoch-30 WM checkpoint and epoch-200 position probe.
  At horizon 1, decoded position error was **0.0813** for the WM versus
  **0.0493** from true latents and **0.0730** for persistence. At horizon 3,
  the corresponding values were **0.1185 / 0.0515 / 0.1476**. At horizon 10,
  autoregressive / teacher-forced / true-latent / persistence position errors
  were **0.2280 / 0.0941 / 0.0560 / 0.3943**. Thus the WM beats persistence
  beyond the first step, but predicted-latent position error compounds.
  At horizon 10, latent MSE divided by persistence MSE was approximately
  **0.42**, close to the angled model's approximately **0.43**.
- The later 150-epoch command shared in chat resumed the **one-step** top-down
  run under Drive's `full_default_v2/`, not the K=3 run: it omitted
  `--rollout-steps 3` and `--rollout-loss-weight 1.0`. At epochs 59–73,
  training loss kept falling while validation loss flattened or rose. The
  script saves the lowest-validation-loss checkpoint as `best.pt`; do not
  assume the last epoch is the model used for evaluation.

The top-down Colab WM runs used Drive paths under
`datasets/pointmaze/world_model_runs/full_default_v2/` and
`full_default_v2_k3/`; their actual files are not mirrored under the local
`topdown/world_models/` placeholders. Probe checkpoints were copied locally
under `topdown/position_probe_runs/`. Confirm the current Drive paths in Colab
before running, and pass paths explicitly rather than relying on CLI defaults.

## Current open question and useful next experiments

The top-down probe ultimately surpassed the angled probe on global test mean
position error. The remaining question is whether the **top-down WM's predicted
tokens** retain position information well enough for reliable planning. The
probe is trained on *real* I-JEPA tokens, so a higher decoded error on WM
outputs could reflect incorrect dynamics, off-manifold predicted tokens that
the probe handles poorly, or both. The one-step position error being worse
than persistence even while latent MSE is better suggests that uniform token
MSE is imperfectly aligned with ball position; it does not establish a unique
root cause.

Suggested order:

1. On the **validation** split, compare `best.pt` and `last.pt` for the
   top-down WM using the latest top-down probe, especially at horizons 1–3.
   A flat latent validation loss does not guarantee a flat decoded-position
   metric. Keep the test split out of checkpoint selection.
2. Run matched closed-loop top-down planning trials with the same top-down
   encoder and WM, changing only the position probe to
   `spatial_softmax_mlp 4/best.pt`. Keep episode seeds, CEM settings, and
   step limits fixed. The planner's horizon is three macro actions and it
   replans after executing one, so horizons 1–3 are more directly relevant
   than a 10-step open-loop rollout.
3. If improving the WM, first determine whether the short-horizon position
   error comes from insufficient training, the token-MSE objective, or poor
   probe behavior on predicted tokens. Do not assume that more epochs or a
   larger encoder will fix it. A less constrained patch-preserving probe on
   the same frozen tokens remains a possible diagnostic, not a completed run.
4. For formal comparisons, avoid repeatedly choosing changes based on the
   same test episodes; reserve fresh held-out episodes or report that the
   existing test set was consulted during development.

## Starting a new chat

Point the new assistant to this file and the active code under
`src/mylonas_ijepa_wm/`. State whether the question concerns the **angled** or
**top-down** dataset and provide the exact checkpoint/cache paths being used.
The active script location is shared, but its CLI defaults do not point to a
camera-specific artifact directory. In particular, the latest local top-down
probe is in a directory whose name contains a space; quote that path.
For new experiments, record the Git commit, CLI arguments, camera metadata,
dataset path, and best-checkpoint epoch together; the existing notebooks do
not fully pin Git revisions or artifact provenance.
