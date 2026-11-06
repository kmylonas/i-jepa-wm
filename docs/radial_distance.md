# Radial Euclidean-distance side experiment

Open `notebooks/radial_distance_colab.ipynb`, select a GPU runtime and Run all.
The existing encoder checkpoint in Drive is the only prerequisite. The ranking
notebook and its published source pin remain unchanged. This experiment writes
new timestamped results under `curriculum_cost_adapter/radial_distance_v1`.

## Fitting

The adapter's symmetric token MSE is fitted to **Euclidean distance / 34**:

`mean((adapted_MSE - radius / 34)**2)`

The scale is always 34, including the initial radius-6 stage. The target is
linear distance, not squared distance, and predictions are not clipped.
Multiplying adapted MSE by 34 gives an estimated distance in simulation units.
There is no ranking or anchoring term.

Same goal (50,42), layout, physical right-side bounds, 24 rays every 15 degrees,
radii 2,6,...34 and cumulative stages 6,14,22,34 (48,90,105,115 training points).
The first adapter starts at identity; later stages retain all prior positions
and warm start the preceding weights. Each stage resets Adam moments and uses
500 updates, learning rate 1e-4, bottleneck 64 and stage seeds 0,1,2,3.
Uniformly sample 32 positions per update, with replacement. This matches the
32 observation evaluations in the old 16-pair batches, but uses a different
sampling distribution. It is a fitting-recipe comparison, not a perfectly
matched-exposure isolation of the loss function.

## Evaluation

The original fixed interpolation, offsets, unseen directions and full-right
heatmaps are retained. Directional errors distinguish inside, outside and
crossing steps using both endpoints. Distance errors use deduplicated heldout
directional positions, excluding coordinate/rendered-image overlap with any
offered training input. No probe reaches fitting or model selection.

Report distance MAE and RMSE in physical simulation units. A diagnostic raw-MSE
baseline fits one nonnegative zero-intercept scalar on the current training
positions only, then predicts distance as `scalar * raw_MSE`. Both methods use
the same heldout positions. The scalar baseline is never a fitting target.
The fixed unseen-direction radius<=6 group also runs after every update, so
changes to earlier generalization are visible. Inside/outside cohorts otherwise
change as the radius expands. At radius34 the directional outside group is
empty and reported N/A; the heatmap still includes physical corners beyond34.

Each stage displays and saves six PNG/PDF figures: raw/adapted heatmaps,
distance-calibration curves/scatter/errors, radial costs, matched offset errors,
interpolation examples and training loss. Calibration plots include the true
distance diagonal. Full-training regression loss is distinguished from noisy
minibatch loss. Counts are descriptive and spatially correlated.

## Artifacts

Fresh goal pixels/tokens and all 4,668 unique coordinates are encoded once,
float32 in batches of eight. A roughly 5.7 GiB local Colab cache is separate
from Drive's persistent training features, checkpoints, figures and reports.
Only stage-approved training tensors enter the optimizer. Objective kind,
fixed scale, configuration, source, input/cache hashes and checkpoint lineage
are validated before fitting or reuse; unsafe output aliases are rejected.
The new repository/cache/output locations are separate from the ranking run.
Lost local tokens require a new run; earlier Drive results remain available.

Local verification uses synthetic tokens and the real CLI/notebook reporting
cells. Actual GPU/Colab behavior and scientific outcomes remain pending user
execution. Other-goal transfer and planning performance are separate questions.
