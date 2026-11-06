# Frozen adapter: eight equal-distance goal shifts

Open `notebooks/radial_goal_sweep_colab.ipynb`, select GPU and Run all. Same
saved distance-fitting run20261106T101715235909Z, stage4/R34 and frozen encoder.
No optimizer, adapter updates, planning or recalibration occurs. Existing
single-goal, distance, ranking and position notebooks/pins/results are unchanged.

Defaults are original goal(50,42) plus eight shifts of sqrt32≈5.66 units at
angles0,45,...315degrees, rounded to6decimal coordinates. This includes the
previous goal(46,38) and tests axial/diagonal directions at the same magnitude.
Source/stage/new-goal list are editable. All goals must be distinct, physically
valid and inside the source fitting region. Source lineage/backbone/renderer/
precision checks and the original cached/fresh-goal consistency control remain.

## Fair comparisons

Use one intersection across ALL goals: ball positions within the original
fitting disk and within sourceR of every goal. Exclude all original fitting
coordinate/image exposure and every query-reference coordinate/image alias.
The heldout ball-position IDs are identical in each goal row. Local/full-domain
metrics remain secondary: local populations vary, full-domain includes farther
spatial/distance extrapolation. Small-stage intersections may have no heldout
grid points; report N/A without broadening the cohort. The original control may
differ from the earlier two-goal study because the shared cohort is stricter.

Distance estimate remains34*adaptedMSE, and raw baseline keeps its original
training scalar. Report heldout MAE/RMSE, deltaMAE versus original control,
uphill/tie fractions/counts. Directional tests retain horizontal/vertical grid
edges oriented toward each evaluated goal, skipping equal-distance edges;
edge counts can vary by goal. Goal cases are descriptive, correlated spatial
tests, not independent seeds or planning trials. Zero self-cost is automatic;
transfer evidence comes from the other positions.

## Figures and artifacts

The summary plot places goal shifts around the original center and colors them
by heldout distance MAE and uphill fraction; numbers match table/figure IDs.
All-N/A metrics render safely. Each of nine goals gets full-right raw/adapted/
true/error heatmaps and a calibration scatter as PNG/PDF. Default notebook
display is summary plus nine heatmaps; `SHOW_CALIBRATION=True` also displays
the scatters. Goal reference images/tokens and full numeric reports are saved.

Freshly encode each unique coordinate once in batches8. Reference tokens can
span multiple batches, are reused for self-comparison and are retained while
remaining points stream through bounded buffers. The old token cache is not
required. New timestamped outputs go to `radial_goal_sweep_v1`, separate from
the source and two-goal runs. Dynamic artifact hashes and source state checks
guard completed reuse; changed goal order/count/coordinates reject reuse.
Missing figures regenerate from verified results without encoding or fitting.

Local tests use synthetic features; actual GPU/goal-transfer results await
user execution. Outcomes concern this checkpoint, displacement and goal set.
