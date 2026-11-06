# Increasing-radius ball-position readout

Open `notebooks/radial_position_colab.ipynb`, select GPU and Run all. Only the
existing frozen I-JEPA encoder checkpoint in Drive is needed. Existing ranking
and Euclidean-MSE notebooks/pins and their results remain unchanged.
Outputs use a new timestamped `radial_position_v1` directory.

## Model and supervision

Frozen patch tokens pass through a shared Linear(D,32)+GELU projection.
Concatenate patches in their existing order, then Linear(P*32,64)+GELU and
Linear(64,2). There is no spatial averaging, coordinate input, goal-token input,
output activation or clipping. Last-layer zeros initialize prediction to the
fixed label center (50,42). Coordinate target is `(true_xy-[50,42])/34`, with
one fixed center/scale across stages; physical prediction is `center+34*output`.
Loss is mean squared coordinate residual. Position labels are richer supervision
than the previous distance-only target; this is a diagnostic of one readout.

Same hand-positioned domain/rays/radii and cumulative stages6,14,22,34, with
48,90,105,115training positions. Each stage keeps earlier labels/features,
uses500updates,batch32uniform positions with replacement,Adam1e-4,clip1 and
stage seed0,1,2,3. Later fits copy previous weights and reset optimizer moments.
No ranking/anchor term, planning/actions, old adapters or world-model loads.

## Evaluation

Primary metric is mean Euclidean position error in physical simulation units.
Position RMSE is `sqrt(mean(||pred_xy-true_xy||^2))`, not coordinate RMSE.
Also report x/yMAE, signed bias and the distance-to-known-goal MAE calculated
from predicted coordinates. Correct radius alone can hide incorrect direction.
The baseline always predicts the mean of the currently offered training labels.

Training, heldout directional inside/outside, unseen-ray inside/outside and a
fixed unseen-ray R<=6 group use deduplicated IDs. Exact coordinate and rendered-
image overlaps with offered fitting inputs are excluded from heldout groups.
The reference goal frame is not a fitting input. Outside bins use true radius
relative to the current boundary: (0,2],(2,6],(6,12],(12,infinity). Empty groups
report N/A. Final-stage directional extrapolation is unavailable beyond34;
full-right heatmaps still show grid corners beyond34. Cohorts are descriptive
and spatially correlated, not independent trials.

Derived-distance interpolation/offset/uphill diagnostics retain the fixed
matched families/tolerances used earlier. The new cost is Euclidean distance
from predictedxy to the known goal, not adapted token MSE. Raw MSE remains a
directional reference. All probes are fixed before fitting; test labels and
features never enter optimization or model selection.

## Figures and artifacts

Each stage saves five PNG/PDF figures: true/predicted goal-distance and position-
error maps, coordinate scatter/arrows/errors, distance-vs-radius curves,
matched local interpolation/offset tests, and noisy/full-training loss curves.
Maps extend to physical x≈60. Prediction values and colors are not clipped.
True/predicted distance maps share a scale; position-error maps have their own.
Reports persist every predicted coordinate for later fixed-cohort analysis.

Encode the same4,668coordinates once with pinned renderer/frozen encoder,
fp32batch8. The roughly5.7GiB feature cache stays on local disk; compact
features/checkpoints/reports/figures go to Drive. Objective/normalization/model
config and input/cache/source/checkpoint hashes guard reuse. Position models
have a separate checkpoint format and whole-chain validation. Completed stages
can regenerate missing figures without fitting. Losing local tokens requires a
new timestamped run; earlier Drive outputs remain available.

Local verification uses synthetic features and real CLI/notebook reporting
cells. GPU execution and scientific position extrapolation await the Colab run.
A failure would concern this architecture/training setup and would not establish
that encoder features lack usable position information.

## Verification

Published production-code pin:
`7a839fb66e5db25fcb1b39df98d4e3bffa070d28`. The full suite passed447tests and
16subtests with4existing warnings. All four synthetic CLI stages exported
20PNG/20PDF figures, and actual notebook report/display/summary cells ran.
A fresh read-only review independently passed5position tests and found no
actionable issues. GPU execution and scientific results await the Colab run.
