# Nearby and distant frozen-goal transfer

Open `notebooks/radial_goal_sweep_far_colab.ipynb`, select GPU and Run all.
Same final stage 4 / R34 distance adapter and frozen evaluator source pin
`8fc35399ea9578512f3016e6e6aad228298d2ad9`; no fitting or recalibration.

Retain original goal (50,42) and eight nearby controls. Add seven distant goals:
(40,18), (50,18), (56,18), (40,10), (50,10), (56,10), (50,8). They are 24–34 units from
the original goal, inside its fitting disk and within valid reset bounds. There
are 16 goals total. Default source checkpoint must remain stage 4 for this set.

All 16 rows share the same eligible heldout ball positions. The intersection is
stricter than earlier sweeps: within the original fitting region and within 34
units of every goal. Compare control and changed goals within this run rather
than comparing unpaired means across notebooks. Full-right per-goal heatmaps
include outside-common-region points; the main table excludes them.

Summary plot/table and heatmaps are printed for all 16 goals. Calibration plots,
goal images/tokens, numerical reports and provenance are saved; optional
`SHOW_CALIBRATION` displays the scatters. Displacement and direction now both
vary. Keep conclusions descriptive and specific to these goals/checkpoint.

Separate timestamped output root `radial_goal_sweep_far_v1`. Earlier notebooks,
source adapter and results remain unchanged. This adds goal coverage within the
already fitted region, not further adapter training or region expansion.
