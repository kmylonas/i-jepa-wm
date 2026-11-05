# Ranking-only adapter curriculum

The user explicitly requests setting aside the goal-reference investigation
and removing the anchor loss. They selected retraining both initial adapters
and all subsequent curriculum updates without anchoring. Continue inline in
the existing curriculum-cost-adapter worktree and publish a standalone Colab.

## Experiment

- Both original and clean recipes start from identity-initialized residual
  adapters, fitted on the same saved successful near-band replay. Never load
  the old anchored adapter as the initial weights for this experiment.
- Every fit uses anchor_weight=0 and only the existing softplus ranking loss.
  Keep the raw replay median as the ranking normalization scale; it is not a
  penalty. Reserve the old anchor-sampling RNG draw to preserve pair sampling.
- Initial fitting uses seed 0, 500 updates, batch 16, lr 1e-4, bottleneck 64.
  Later stages warm-start their own last checkpoint and reset Adam, retaining
  the existing stage seeds, replay balancing and pair rules. Original uses
  temporal pairs; clean additionally requires >=0.5 actual goal progress.
- Retain the current shared saved clean goal unchanged. Do not fix or replace
  it in this ablation. The existing target mismatch remains part of the setup.
- Reuse deterministic cases and geometry: x [40,54], near y [30,36], then
  [22,30], [14,22], [8,14], goal (50,42), layout (32,30). Twenty evaluation
  cases per band; forty collection attempts per new band and recipe.
- Use the same evaluate/collect/success-only cumulative replay/update/retention
  loop, zero-new-pair stop, fixed budget and provenance/resume checks. Policies
  collect their own experience; changing the objective can change that data.
- Save to multiband_no_anchor_v1, separate from all prior artifacts. Existing
  source and anchored result directories remain read-only.
- Produce initial and per-update raw/adapted/difference maps, then expanded
  shared-scale maps and vertical profiles across both recipes and all saved
  versions. Compare success/action matrices against the historical anchored
  run on identical evaluation cases, marking missing cells unavailable.

## Interpretation

This tests whether allowing the ranking objective to reshape costs without
the raw-cost penalty improves geometry and planning. Cost magnitudes may grow;
report numeric maxima and unclipped maps instead of clipping or secretly
regularizing the fit. No change to the architecture or other optimizer terms.
The historical run is a reference, not a paired-data offline loss experiment:
policy-dependent collections differ, and historical original initial fitting
did not persist its goal tokens. New initial fits both use the shared saved
target. Do not attribute every difference uniquely to anchoring.

The maximum remains 840 planning/collection episodes; fitting two initial
adapters adds 1,000 optimizer updates but no simulator actions. GPU outcomes
are measured by the user's Colab execution, not by local synthetic tests.
