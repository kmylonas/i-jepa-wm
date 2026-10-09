# PointMaze CEM Planning Design

## Objective

Add live model-predictive control for `PointMaze_UMaze-v3` using the
frozen I-JEPA encoder and the trained action-conditioned latent world
model. The planner receives RGB observations and a target RGB
observation, searches over continuous actions with the cross-entropy
method (CEM), executes one macro action, observes the real environment,
and replans.

The first experiment is diagnostic. It must establish whether the
current world model can support closed-loop control before introducing
larger models or longer rollout training.

## Scope

The implementation will provide:

- a testable, environment-independent CEM optimizer;
- batched and chunked world-model evaluation of candidate action
  sequences;
- image-goal costs based on either the position probe or direct latent
  distance;
- an explicit privileged oracle-goal cost for diagnosis only;
- a live PointMaze MPC evaluation command;
- per-episode and aggregate JSON results.

The implementation will not train or modify I-JEPA, the world model, or
the position probe. It will not add proprioception to the world-model
input. It will not attempt to reproduce every DINO-WM environment or
planner setting.

## Components

### Pure planning module

`src/mylonas_ijepa_wm/planning.py` will contain code that does not import
Gymnasium or MuJoCo:

- CEM optimization over bounded continuous macro actions;
- construction and normalization of the action sequence expected by the
  world model;
- chunked autoregressive rollout of candidate sequences;
- terminal probe, latent, and oracle-goal costs;
- validation of tensor shapes and planning arguments.

Keeping this module independent from the simulator allows its behavior
to be tested with small deterministic models.

### PointMaze runner

`src/mylonas_ijepa_wm/plan_pointmaze.py` will contain the live evaluation
CLI and simulator integration:

- create `PointMaze_UMaze-v3` with 224-by-224 RGB rendering;
- hide the red goal marker to match the training observations;
- load the frozen I-JEPA encoder, world-model checkpoint, and position
  probe when required by the selected cost;
- construct a target observation;
- run CEM in an MPC loop;
- execute actions and update the observation history;
- report and save evaluation metrics.

Gymnasium and Gymnasium-Robotics imports will be lazy so that unit tests
for the planning core do not require MuJoCo.

## Goal Observations and Costs

The planner's goal is an RGB target observation, not the numeric
`desired_goal` returned by the simulator. For evaluation setup, the
runner will create this target image by temporarily placing the green
ball at the episode's sampled goal, setting its velocity to zero,
rendering, and restoring the original simulator state. The red target
marker remains hidden in both current and target images.

The goal image is encoded once by I-JEPA at the start of an episode.

The CLI will support three costs:

1. `probe` (default): decode the final predicted latent and target-image
   latent with the trained position probe, then minimize their Euclidean
   distance. The planner never receives the simulator goal coordinate,
   although the probe itself was trained with position supervision.
2. `latent`: minimize mean squared distance between the final predicted
   patch tokens and the target-image patch tokens. This is the fully
   representation-based objective.
3. `oracle`: decode the predicted latent with the probe and compare it
   directly with the simulator's numeric `desired_goal`. This privileged
   mode is only a diagnostic upper-bound-style baseline and will be
   identified as such in saved results.

All three modes score the final predicted state of each candidate action
sequence. Simulator state may be used after action execution to measure
success and final distance, but not as world-model input.

## Temporal and Action Alignment

The world model uses three latent observations. A macro action contains
five consecutive 2D PointMaze controls, flattened to dimension 10,
matching the training cache.

For a planning horizon `H`, each candidate has shape `[H, 10]`. The world
model rollout receives the two previously executed macro actions followed
by the `H` candidate macro actions. This produces the required
`num_hist + H - 1` aligned actions for autoregressive prediction.

At reset, only one real observation exists. History is initialized as
three copies of the initial latent with two zero macro actions. After
each real macro action, the oldest latent and action are discarded and
the newly encoded real observation and executed action are appended.
Consequently, after two planning cycles the complete history comes from
real environment transitions.

Raw candidate actions are bounded by the PointMaze action limits. If the
checkpoint contains action-normalization statistics, history and
candidate actions are normalized immediately before world-model input.
The raw actions remain available for simulator execution.

## CEM Optimizer

CEM maintains a diagonal Gaussian over action sequences with shape
`[H, 10]`.

For each iteration it will:

1. sample a configurable population;
2. clamp samples to the environment action bounds;
3. score candidates through chunked world-model rollouts;
4. select the lowest-cost elite candidates;
5. replace the Gaussian mean and standard deviation with elite
   statistics, clamping the standard deviation to a small positive
   minimum.

The lowest-cost sequence from the final iteration is returned. A seeded
`torch.Generator` makes planning repeatable. Candidate evaluation is
chunked because repeating `[3, 256, 1280]` history for the full population
would otherwise consume excessive accelerator memory.

Initial defaults are:

- planning horizon: 3 macro actions;
- population: 256;
- elites: 32;
- CEM iterations: 5;
- candidate evaluation chunk size: 16;
- initial Gaussian mean: 0;
- initial Gaussian standard deviation: 1;
- minimum standard deviation: 0.05.

## MPC Execution

At each planning cycle, the runner will optimize a complete macro-action
sequence but execute only its first macro action. That macro action is
reshaped to five 2D controls and sent to the environment through five
consecutive `env.step` calls. The resulting RGB frame is encoded by
I-JEPA and used to update the real history before replanning.

An episode ends when the environment reports success, termination, or
truncation, or after 20 executed macro actions. The default therefore
uses at most 100 underlying 10 Hz controls, matching the duration of the
training trajectories.

## Command-Line Interface

The runner will accept paths for:

- the I-JEPA encoder checkpoint;
- the world-model checkpoint;
- the position-probe checkpoint;
- the output JSON file.

It will also expose the cost, number of evaluation episodes, base seed,
maximum macro steps, planning horizon, CEM population, elite count,
iterations, candidate chunk size, device, numerical precision, and an
optional compilation flag.

The position-probe checkpoint is required for `probe` and `oracle` costs
and unused for `latent`. Defaults will point to the repository's current
K=3 world model, encoder-only I-JEPA checkpoint, and MLP position probe.

## Results

The JSON output will include the full planner configuration and one
record per episode containing:

- seed;
- cost mode;
- success;
- executed macro and micro steps;
- initial and final simulator goal distances for evaluation only;
- final planner cost;
- executed macro actions;
- achieved-position trajectory.

Aggregate fields will include success rate, mean and median final goal
distance, and mean macro steps. Terminal logs will show progress and the
same aggregate result.

## Error Handling

The CLI will reject incompatible checkpoint architecture, invalid CEM
settings, missing probe checkpoints for probe-based modes, action bounds
that do not match the 2D-by-five macro representation, and environments
whose rendered images do not have shape `[224, 224, 3]`.

Target-image rendering must restore the simulator position and velocity
even if rendering fails. Environment cleanup will occur in a `finally`
block.

## Testing

Automated tests will cover:

- convergence of CEM on a bounded quadratic objective;
- reproducibility from a fixed seed;
- candidate action/history alignment and normalization;
- chunked rollout equivalence to unchunked evaluation;
- probe, latent, and oracle cost calculations;
- execution of exactly five underlying actions per macro action;
- restoration of simulator state after target-image rendering;
- MPC history updates;
- argument validation and CLI defaults;
- a lightweight end-to-end episode using deterministic fake encoder,
  world model, probe, and environment objects.

The existing world-model, rollout, and position-probe test suites must
remain green.

## Success Criteria

The implementation is complete when a user can run one command against
the saved K=3 checkpoint and obtain reproducible PointMaze episodes and
aggregate metrics for `probe`, `latent`, or `oracle` goal costs. During
planning, the world model receives only I-JEPA latents and actions; no
simulator state or proprioception enters its input.
