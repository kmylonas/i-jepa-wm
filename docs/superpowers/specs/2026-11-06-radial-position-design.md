# Cumulative radial position readout

User approved predicting ball coordinates from frozen I-JEPA tokens, training
in a limited region and testing outside it, with the same increasing-radius
stages. Preserve ranking/distance experiments and source pins; same worktree,
branch and authorized Colab publication workflow. No planning/actions.

## Model and fitting

Builder default: a small spatial MLP. Apply Linear(D,32)+GELU to every patch
token, concatenate patches in their original order, then Linear(P*32,64),
GELU and Linear(64,2). No spatial averaging, coordinate input, goal-token
input, output activation or output clipping. Start the last layer at zero,
so initial prediction is the fixed label center (50,42).

Fit mean coordinate squared residual to (true_xy-[50,42])/34. Center and
scale stay fixed at all stages; report physical positions as center+34*output.
Actual coordinate labels reach fitting only for approved training IDs.
Same frozen encoder, fresh static renders, right-side domain,24rays/radii,
stage limits6,14,22,34 and cumulative48,90,105,115training positions.
500Adam updates per stage, batch32uniform positions with replacement, lr1e-4,
stage seeds0,1,2,3, gradient clip1. Fresh model initially, then cloned previous
weights with fresh Adam moments. No ranking/anchor term or test-based selection.
Training metadata includes actual sample counts and before/after/full-pool
coordinate loss and position error every50updates plus noisy batch loss.

## Fixed evaluation

Reuse all4,668coordinates, withheld directions, interpolation/offset families,
full-right grid and fresh encoder/renderer protocol. Exclude offered coordinate
and rendered-image overlaps from heldout position metrics. The goal frame is
not an input to the position fitter and must not be counted as fitting exposure.
Primary metric: mean Euclidean position error ||pred_xy-true_xy||, position
RMSE, x/yMAE and signed bias in physical simulation units. Also compute
distance-to-known-goal MAE from predicted coordinates. Report training,
heldout directional inside/outside, unseen-direction inside/outside, and fixed
unseen-ray R<=6 groups with deduplicated IDs and empty groups N/A. Report
near-boundary outside groups (0,2],(2,6],(6,12],(12,infinity) beyond stageR.
These are descriptive spatial comparisons, not independent statistical trials.

Baseline: constant position equal to the mean of offered training coordinates,
with no test-label fitting. Show its position error on identical test groups.
Derived-distance directional/interpolation/offset checks reuse established
matched cohorts/tolerances. Label the new cost as position-derived distance,
never adapted latent MSE. AtR34 directional probes beyond34 are unavailable;
grid corners still extend beyond34 and are visibly tagged as grid-only probes.

Figures after each stage: full-right true/predicted goal-distance and position-
error maps; true-vs-predicted x/y and spatial arrows; distance-vs-radius curves;
matched interpolation/offset diagnostics; training loss. Fixed-radius masks and
training points overlaid; no clipped prediction values or heatmap colors.
Store all predictedxy and numerical errors in reports so later fixed-cohort
analysis does not require re-encoding. Failure of this particular readout does
not establish that encoder features lack position information. Predicting
coordinates uses richer supervision than the earlier distance-only experiment.

## Artifacts and compatibility

Separate radial_position_colab.ipynb, timestamped Drive root radial_position_v1,
local token cache and Colab clone. Only the frozen encoder checkpoint is needed.
Reuse streaming cached preparation with explicit signed objective position,
fixed label center/scale and model configuration. Existing ranking/distance
defaults, fitters, notebooks and pins remain unchanged. Position checkpoints
have their own model format, provenance and complete-chain validation; validate
source/config/input/cache/checkpoint/report hashes and aliases before fitting
or admitting completed results. Missing figures can regenerate from validated
results without retraining. No silent reuse across experiment kinds.

Verify meaningful RED->GREEN tests, legacy focused tests, all four synthetic
CLI stages/plots and actual notebook report cells; full suite; one fresh final
review, one fix pass if needed, production-code pin and verified push. Actual
GPU execution and empirical generalization remain for user Colab execution.
