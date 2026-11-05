# Ranking-only curriculum ablation

Open `notebooks/curriculum_no_anchor_colab.ipynb`, select a GPU and Run all.
Use the same Drive sources as the previous curriculum. New artifacts live in
`datasets/curriculum_cost_adapter/multiband_no_anchor_v1`; the prior
`multiband_curriculum_v1` remains the historical anchored reference.

Both original and clean adapters are newly initialized to identity and fitted
on the saved successful near-band replay. The original uses temporal pairs;
clean also requires >=0.5 actual goal progress. Every fit has anchor_weight=0,
so the objective is only `softplus(0.2-(early-late)/scale)`. The median raw
replay cost still normalizes the ranking margin. No penalty pulls adapted
costs toward raw MSE, and anchor observations are excluded from the forward
pass. The reserved sampling draw keeps pair RNG aligned with anchored fits.

Initial fitting uses seed 0, 500 updates, batch16, lr1e-4, bottleneck64 and
fresh Adam. Later fits use the same cumulative replay balancing, stage seeds,
warm starting and fresh optimizer as the existing curriculum. The two branches
collect their own experience; only successes enter their ranking replay.

Starts keep x [40,54], near y [30,36], then y [22,30], [14,22], [8,14]. The
goal (50,42), layout, frozen encoder/world model and planning settings are
unchanged. The shared saved goal is deliberately retained; this ablation does
not change or resolve the goal-reference inconsistency.

The maximum remains 840 episodes across raw plus both curricula. Two initial
fits add 1,000 optimizer updates but no simulator actions. Source/goal/replay/
runtime/checkpoint/journal provenance checks and per-branch stopping remain.
Initial checkpoints are written under each branch's `checkpoints/v0.pt`, with
their inputs and fitting metadata in `updates/v0/`. Completed initial fits
resume without refitting; a partial fit can restart from identity.

Immediate heatmaps use a full range for the current checkpoint. Final maps
and vertical profiles use one expanded range across all available versions
and both recipes. Cost growth is visible rather than hidden by the old raw
color ceiling. The report stores actual numeric costs, outcome matrices,
case paths, replay sizes, loss histories and actual interaction budgets.
Final plots admit maps only after validating completed checkpoint lineage,
the saved map hashes, and their binding to the current checkpoint and grid.
An invalid recipe is omitted while valid results from the other remain visible.

The `compare` CLI stage reads the old anchored cases without running episodes
or rewriting that source. It checks identical cases, planning/fitting settings
apart from anchor weight, source artifacts and shared goal. Missing cells are
unavailable, not zeros. `anchor_comparison.json` records paired wins/losses
and action counts. A damaged historical recipe does not hide the other's
comparisons. Execution settings, including precision and encoder batch size,
must also match; an incompatible historical run is reported as unavailable.

The historical original near fit did not persist its target tokens, while
both new initial fits use the shared saved target. Policy-dependent collection
also changes when the adapter changes. Treat historical comparison as an
experiment reference, not a fixed-data causal estimate of anchoring alone.

Return the executed notebook to compare planning, ranking loss, cost growth
and the lower-band cost hill. Empirical GPU results remain pending execution.
