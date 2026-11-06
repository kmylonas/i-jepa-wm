# Frozen distance-adapter goal transfer

Open `notebooks/radial_goal_transfer_colab.ipynb`, select GPU and Run all.
Defaults use the existing distance-fitting run20261106T101715235909Z, final
stage4/R34, original goal(50,42) and new goal(46,38). Source run, stage and new
goal are editable. No fitting, optimizer, planning or recalibration runs.

The adapter is frozen and loaded from the saved Euclidean-MSE experiment,
not the ranking or coordinate experiment. Root input hashes and completed
stage1..selected checkpoint lineage/provenance are verified without requiring
the previous local feature cache. Historical fitting source is preserved and
may differ from this diagnostic source. Encoder, renderer and precision must
match; current execution settings are recorded. Old notebooks/results stay intact.

Encoder images contain walls and one ball, no goal marker. A moved goal uses a
fresh reference image with the ball at the new location. Both original/new
reference coordinates and the406point full-right grid are encoded once, batch8
fp32, with identical tokens reused for self-comparisons. A fixed-position render
is checked before/after resetting the goal. Original-goal cached/fresh token MSE
and grid-score changes are reported as a runtime/reference consistency control.
Source files and adapter parameters must remain unchanged after evaluation.

Costs remain adapted MSE, with distance estimate34*cost. The raw baseline uses
the scalar saved in the source training report, unchanged for the new goal.
No target/test labels fit either scalar or adapter. Main metrics use exactly the
same ball positions for both goals, inside the original fitting disk and within
the source trained distance range relative to both goals. The heldout subset
excludes fitting coordinate/image overlaps and query references. Also report
common, per-goal localR<=6 and full-domain metrics; full-domain includes spatial/
distance extrapolation. Empty denominators are N/A. Directional tests use
horizontal/vertical adjacent grid edges, oriented toward each goal by true
distance, omit equal-distance edges and report uphill/tie fractions with counts.
Tolerance is max(1e-7,0.001*common-region cost range), separately per method/goal.

Each goal has raw-MSE, adapted-MSE, true-distance/34 and physical distance-error
heatmaps plus a calibration scatter. Adapted/true maps share a full-range scale;
raw/error use separate scales. All maps span physicalx35..60,y4..60 and show the
original fitting region/training positions plus original/evaluated goal markers.
No values are clipped. The fitting circle stays at the old center when the goal
moves. Main metrics are descriptive spatial comparisons, not independent trials.

New timestamped outputs live under `radial_goal_transfer_v1`, separate from the
source run. Persist fresh goal pixels/tokens, numeric grid scores, signatures,
source hashes, reports and4PNG/4PDF figures. Validated completed results can be
displayed and missing figures regenerated without re-encoding or fitting.
Changing goals/settings under an existing output rejects reuse; rerun setup
for a new timestamp. Actual GPU/scientific transfer awaits user execution;
the local workflow checks use synthetic features. Success is specific to the
selected goal shift/checkpoint and does not establish general planning transfer.

## Verification

Production-code pin: `2cdafe088d46b4843a5147928c049025e3ce33fe`.
Full local suite:452tests and16subtests passed with4existing warnings.
Synthetic frozen evaluation, validated report CLI, four PNG/PDF figure pairs
and actual notebook reporting cells passed. Fresh review identified one query-
reference image-alias exclusion issue; a failing regression and full suite
verified its fix. Actual GPU/goal-transfer outcomes await the Colab run.
