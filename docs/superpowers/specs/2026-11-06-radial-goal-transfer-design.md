# Frozen distance-adapter goal transfer

User pauses expansion/architecture experiments and requests moving the goal
inside the fitting region for the Euclidean-MSE adapter, with heatmaps.
Use existing distance adapter (not ranking/position model), read-only; no fitting,
optimizer, new labels for calibration, planning or actions. Same worktree/branch
and authorized Colab publication workflow, all earlier notebooks/results intact.

Defaults: distance source run20261106T101715235909Z under radial_distance_v1,
stage4 (R34), original goal50,42 plus new goal46,38. Source path, stage and goal
editable. Validate new goals inside source fitting disk and physical right-side
domain, and safe reset bounds. The encoder-facing renderer contains no goal
marker. Render/reset each goal to establish correct reference image; verify a
fixed-position render stays identical when changing environment goal. Encode
each full-right grid position and goal once, batch8 fp32, then compare frozen
adapter MSE to each freshly encoded goal. Same frozen encoder/pinned renderer.

Verify persistent source root input hashes/config kind distance/scale34/design,
completed stage1..selected chain hashes and checkpoint provenance, without
requiring the lost local token cache. Historical source fingerprint is recorded
but need not equal diagnostic code; require encoder, renderer and precision
compatibility. Original-goal fresh control separates goal transfer from runtime
changes. Record fresh-vs-cached original goal raw-token MSE and matched grid
cost differences; do not silently calibrate them away. Freeze adapter parameters;
verify source checkpoint hash and in-memory state unchanged after evaluation.

Fixed grid from existing radial design spans x35..60,y4..60 (406coordinates),
plus unique goal coordinates. Main metrics use the same ball positions for all
goals: inside the original fitting disk and no farther than fittingR from any
tested goal. Also show full-domain, goal-local R<=6 and coordinate/image heldout
subsets. Preserve empty denominators N/A. Direction tests use horizontal/vertical
adjacent grid edges oriented toward each goal by true distance, equal-distance
edges excluded, tolerance max(1e-7,.001*costrange). Compare methods on same edges.
Distance estimates34*adaptedMSE; raw baseline k*rawMSE with k from original
training report, never re-fit for new goal. Cost targets trueEuclideanDistance/34.

Per goal: rawMSE/adaptedMSE/trueDistanceOver34/error heatmaps, shared full-range
adapted/target scale, separate raw/error scales, original fitting circle and
training points, old/newgoal markers; no clipped values. Calibration scatter
for common region. Save fresh goal images/tokens, numeric grid report, signatures,
source checkpoint lineage, PNG/PDF figures in separate timestamped
radial_goal_transfer_v1 folder. No modifications to source run. Reject output
overlap/descendant aliases and mismatched/corrupt completed diagnostic reuse.
Completed output can regenerate missing figures without encoding or fitting.

Separate pinned radial_goal_transfer_colab.ipynb, frozen encoder only plus
source Drive results. Unit/integration RED->GREEN, full suite, synthetic CLI/
notebook workflow, fresh final review and verified publication. Actual GPU/
scientific goal transfer awaits user run. Success applies to selected goals,
geometry and checkpoint; it does not establish arbitrary-goal/planning transfer.
