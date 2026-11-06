# Radial Euclidean-distance regression side experiment

User requests a separate increasing-radius experiment fitting adapted latent
MSE to Euclidean distance, then measuring farther-radius generalization.
Existing ranking notebooks, output folders and published source pins remain
unchanged. Same worktree/branch and existing Colab publication workflow.

## Target and fitting

Builder choice: cost C=mean((A(z_x)-A(z_goal))^2) should equal r/34, where r is
physical Euclidean distance to the fixed goal (50,42). Loss is mean((C-r/34)^2).
This matches distance itself, not squared distance. The scale34 is a protocol
constant throughout all stages; it is never recomputed from stage radius or
test scores. Prediction in simulation units is34*C; values are not clipped.
No ranking/anchor term, no planner, no actions, no world-model checkpoint.

Reuse the exact24rays, radii2,6,...34, physical bounds, stage limits6,14,22,34,
fresh goal reference, frozen encoder and residual token adapter bottleneck64.
Identity start; later stages warm start with all earlier training positions.
500Adam updates, lr1e-4, fresh moments, stage seeds0,1,2,3. Uniformly sample
32positions/update with replacement (matches32observation evaluations in the
old16pair batches). This sampler differs from uniform pair sampling; describe
the comparison as replacing the fitting objective and point sampler, not an
exact isolation of the objective under an identical exposure distribution.
Only currently offered training tensors/radii may reach the optimizer.
Record actual sample counts and full-training loss/MAE/RMSE before/after and
every50updates; fixed schedule, no probe-based stopping or model selection.

## Evaluation

Reuse all fixed interpolation/offset/withheld-direction probes and full-right
heatmaps x35..60,y4..60; preserve coordinate/rendered-image exposure masks.
Keep the existing uphill/tie/endpoint/monotonicity metrics and denominators.
Additionally report distance MAE/RMSE (simulation units) on training positions
and genuinely heldout directional points, split both inside and outside the
current radius. Deduplicate point IDs. Final-stage outside-direction metrics
are N/A because these rays end at34. FixedR<=6 heldout-direction metrics at
every stage permit comparison of the same near region across updates.

Builder choice: compare distance errors to a calibrated raw-MSE baseline.
Fit one nonnegative zero-intercept scalar k on offered training points only,
minimizing mean((k*C_raw-r)^2). Evaluate k*C_raw on the same heldout points.
This distinguishes learning a spatial shape from rescaling the raw cost;
the scalar is a diagnostic baseline, never an adapter input or training loss.

Show all existing figures plus a distance-calibration figure with raw-baseline
and adapted radial curves against true r, training limit, scatter and fixed
near-region error. Show regression loss with correct labels; retain noisy
minibatch and full-training measurements separately. Save numerical reports.

## Artifacts and compatibility

New notebook radial_distance_colab.ipynb; new timestamped Drive root
radial_distance_v1 and separate local cache/repository paths. Reuse frozen
encoder checkpoint only; no old adapters or results as fitting inputs.
Shared runner can dispatch a validated objective kind, defaulting to legacy
ranking. No change to ranking fitter, geometry, notebook or old source pins.
Kind/scale are signed in configuration and checkpoint lineage; mismatched
objectives/scales/data/source, corruption and output aliases must reject reuse.
Use the same bounded-memory encode-once cache and fresh goal as radial ranking.
All five notebook code cells compile; synthetic CPU stages/plots and notebook
result cells run; full suite green before source pin/push. Actual GPU/Colab
behavior remains explicitly pending user execution. No claims about other-goal
or planning transfer follow from this experiment.
