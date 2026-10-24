# Wall layout generalization experiment

## Research question

Can one action-conditioned, proprioceptive I-JEPA world model use its image
tokens to predict and plan in DotWall layouts whose wall and doorway positions
were absent from training? The primary outcome is performance on held-out
geometry, supported by collision and rollout diagnostics. Closed-loop success
alone is insufficient because the planner also receives the current true
position after each execution batch.

## Experimental boundaries

- Keep the frozen I-JEPA ViT-H/14 encoder, DotWall simulator, 224-pixel RGB
  rendering, two-dimensional state, five primitive actions per WM step,
  `num_hist=1`, and one-step WM training used by the strongest fixed-layout
  baseline.
- Vary the wall's horizontal position (`wall_x`) and the doorway's vertical
  position (`door_y`) independently. DotWall supports both through its fixed
  layout parameters. Keep `wall_width=6` and `door_space=4` constant; the
  current inclusive renderer makes their visible spans seven and nine
  simulator-grid cells, respectively. This tests position generalization,
  not a change in wall thickness or opening size.
- Use 50 primitive actions per collected episode and embed every frame
  (`--embedding-frame-stride 1`). Train from scratch; do not initialize from
  the fixed-layout checkpoint.
- Use one deterministic dataset manifest with explicit episode ID, seed,
  `[wall_x, door_y]`, and split. The configuration and manifest must be saved
  with the run. All stages validate them. Train, validation, and test layouts
  are disjoint. Normalization statistics use training episodes only.
- Use a collector alternating a correlated random-heading policy with
  primitive action lengths distributed across `[0.2, 3.5]` and the existing
  doorway-directed policy within each layout. The wide random policy must
  actually sample the upper part of that interval, rather than merely raise
  a truncation bound on a narrow distribution. The doorway policy uses
  geometry only to collect data; no layout coordinates enter the world
  model. Record each policy and its action distribution. Preserve the
  existing DINO-like `[0.2, 1.8]` policy for older runs.
- Preserve the existing episode-random split as the default for older runs;
  require an explicit layout manifest for this experiment.

### Proposed layouts

The training grid is the Cartesian product of wall x positions
`[24, 32, 40]` and door y positions `[18, 30, 42]` (nine layouts).
Moving both coordinates is deliberate: keeping wall x fixed would leave
the model free to rely on the familiar wall position exposed by the
PointMaze transfer failure.
Validation uses `[[28, 24], [36, 36]]`. Test uses
`[[26, 22], [26, 38], [38, 22], [38, 38]]`. Every validation and test
wall coordinate and door coordinate is absent from training; test coordinates
are also absent from validation. All lie within the training coordinate
ranges, so this first experiment tests interpolation. The exact pairs are a
design choice, not a result or an assumption that they are equally hard.

Target counts are 2,000 training episodes, 200 validation episodes, and 400
test episodes, balanced across the layouts within each split. A small pilot
uses 10 episodes per layout before the full collection. The collector should
accept explicit layout assignments rather than deriving layout from seed
modulo a list length. Seeds remain deterministic and are reused across
models at planning time.

## Data and coverage gate

Before embedding the full dataset, a read-only audit of raw episodes reports,
per layout and policy: episode count, action-length distribution (including
counts above 1.8, above 3.0, and up to 3.5), attempts
to push through solid wall, successful doorway crossings, direction of
crossings, and contacts with outer borders. The audit compares observed
motion with DotWall's `position + 2 * action` proposal and checks the known
wall/door geometry, distinguishing wall collisions from border collisions.
Include representative start/action/end triples for manual inspection.

The pilot checks that both types of transition occur and that each layout is
reachable under the assigned policy. The full audit must show at least 25
solid-wall contacts and 25 doorway crossings per training layout, with at
least five crossings in each direction, before embedding. If a layout misses
the target, collect more episodes for that layout, then re-audit; never fill
the gap with validation or test episodes. Counts are coverage requirements,
not evidence of model quality.

Raw frames and dense embeddings are large. Support resumable collection and
embedding in batches, while retaining the manifest and trajectory sidecars.
No full GPU training is assumed to run on this Mac; provide Colab/CUDA-ready
commands and a local smoke test. Do not claim a trained multi-layout model
until its checkpoint and evaluation outputs exist.

## Training and model selection

Extend `wm_proprio` with a manifest-based split option. It verifies that
every requested episode exists, that each sidecar's saved layout and seed
match the manifest, and that no layout appears in multiple splits. Save the
manifest identity, layout sets, and exact IDs in checkpoint metadata. A
resumed or fine-tuned run must reject a mismatched manifest. Validation loss
selects the best checkpoint; test data never selects epochs or hyperparameters.

Keep the fixed-layout model and its saved 18/20 planning run as historical
reference. Do not present different collection policies and training sets as
an isolated causal comparison. The primary new model starts from random WM
weights with the same architecture and one-step objective as `wall_fixed_3`.

## Evaluation

1. **Rollouts:** Report one-step and autoregressive 2–5 WM-step state and
   token MSE, per layout and split, alongside persistence. Include contact
   and doorway-transition subsets. Use test results only after model and
   planning settings are fixed.
2. **Counterfactual collision probes:** For each test layout, place the dot
   near its new solid wall and push toward it; place it at the doorway and
   cross; place it near an old training-wall coordinate that is now open and
   move through. Replay the same starting state and action sequence in the
   real simulator. Record predicted and actual final positions, crossing
   classification, progress error, and token error. Probe both travel
   directions where feasible. These cases test whether the prediction
   changes with visible geometry rather than memorized coordinates.
3. **Planning:** Run position-cost CEM on each test layout with the same 20
   start–goal seeds for the multi-layout model and the fixed-layout
   checkpoint. Save each layout's JSON and videos for failures. Use horizon 5,
   five macro actions per plan, population 300, 30 elites, ten CEM
   iterations, and at most 20 macro steps. Run three action-search modes for
   each model/layout/seed combination: unbounded, Euclidean primitive-action
   length capped at 4.0, and length capped at the maximum training length of
   3.5. The latter two modes must remove the adapter's per-component box
   bound before applying the Euclidean norm cap. Report success count,
   confidence interval, final distance, and failure type per layout and
   action-search mode, not only a pooled percentage. Clearly label the
   4.0 and unbounded modes as beyond the training action support.
4. **Sanity checks:** Plan on two training layouts with the same seed set,
   and verify that planning records carry the actual `[wall_x, door_y]`.

The primary claim requires credible blocking at new walls, free passage at
old wall coordinates, and useful held-out planning. A high planning success
rate without the collision probes supports control performance but not the
mechanism claim.

## Deliverables

- Versioned layout/split manifest and environment configuration, plus
  commands to collect, audit, embed, train, evaluate rollouts and probes, and
  plan.
- Minimal additions to the existing collector, trainer, planner output, and
  evaluation tools; focused tests for split leakage, metadata checks, and
  coverage classification.
- A small local pilot and smoke-test report. Full embedding/training and
  final held-out results depend on CUDA compute and are separate run outputs.

## Implementation calls to review

- The layout grid and interpolation-only test positions above are proposed
  defaults.
- The 2,000/200/400 episode budget keeps training volume near the fixed
  baseline while giving 100 held-out trajectories per layout.
- The three requested planning modes share all settings and start–goal seeds;
  3.5 is the in-support reference and 4.0/unbounded probe extrapolation.
