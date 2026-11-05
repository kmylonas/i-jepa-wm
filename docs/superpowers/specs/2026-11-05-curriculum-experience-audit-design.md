# Read-only audit of curriculum collection experience

User approved the proposed audit after the first run: inspect wandering,
efficiency, reversals, visited-state coverage, and correctness of the temporal
training pairs. The original results are near 20/20 unchanged and farther
18/20 to 16/20. This audit does not retrain, collect or replay simulator actions.

## Inputs and execution

Read the existing pilot manifest, runtime identity, complete collection records,
and `raw/run_0/*.pt` on Drive. Verify matching case/runtime identity and aligned
frames, states, actions and primitive timestamps. Read existing training metadata
and evaluation/audit records when available. Do not instantiate PilotRuntime or
load encoder/world-model checkpoints. New audit source must be usable against
the original run without changing its provenance or stage files.

Provide a separate CPU Colab notebook that clones the feature branch in a
separate source directory and reads the original pilot output. It writes only
derived files under `experience_audit/run_0/`. Keep the executed user notebook
and all original pilot artifacts intact. Push the branch under the user's
established GitHub/Colab workflow.

## Measurements

For every collection episode, compute actual path length, shortest straight
distance to the success region, successful path efficiency, distance-to-goal
change at each primitive action, cumulative regression, stationary movement,
action variation and action/movement reversals (>90 degrees). Report strict
progress signs and a configurable 0.25-unit tolerance to separate small changes.
These geometry diagnostics apply to this unobstructed same-side setup; do not
treat Euclidean progress as a universal label for doorway detours.

Reproduce the fitter's exact revisit filter (0.5 units) and pair constraints
(>=2 primitive steps, >=2 units endpoint separation). Compare pair progress
before and after filtering. Average per-trajectory fractions across the usable
successful trajectories, matching uniform trajectory then uniform pair sampling.
Also report pooled-pair fractions with a different explicit label. Verify saved
training retained/pair counts and usable IDs when metadata exists. Raw data is
never filtered in place. Identify the most contradictory retained pairs.

Plot visited positions against near/far start rectangles and the goal; quantify
visits to the farther test rectangle. Exiting the near *start rectangle* toward
the goal is expected and must not itself be labeled poor experience.

Read paired evaluation results, identify previously successful cases lost after
adaptation and plot their recorded macro-boundary paths/distances with actual
timestamps from macro durations. Label this resolution explicitly. Show numeric
candidate audit summaries already saved, including that those cases can differ
from lost cases. Summarize existing training loss history without interpreting
it as accuracy. Exact adapter ordering is not computed: the original goal
embedding was not persisted, so this CPU audit must not substitute a successful
terminal image or silently load the huge encoder.

## Outputs

Self-contained `summary.json`, per-episode CSV, plain-language `README.md`,
progress overview, pair-quality figure, coverage figure, worst/best episode
paths and real-frame examples, surviving contradictory-pair image examples,
and paired lost-case plots when evaluation exists. Save plot-ready state/action
traces in a small JSON file, so the report can be shared without the full raw
image dataset. Print headline measurements and clear definitions in the notebook.

Tests use hand-checkable trajectories, verify sampler weighting, rejection of
misaligned/mixed artifacts, no mutation of original inputs, and a synthetic CLI
audit with figures. Do not present synthetic results as findings about the user's
40 episodes. Validate notebook code and request a fresh review before delivery.
