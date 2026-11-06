# Manual multi-goal ranking

User requests returning to ranking supervision in the multi-goal setting,
using manually positioned observations without planning or experience
collection. Keep the existing worktree, original experiments and publication
workflow. Create a separate `radial_multigoal_ranking_colab.ipynb`.

Reuse the final R34 manual fitting candidates and the same 15 reserved
near/far goals. Use the established marker-free, float32/batch8 data preparation
and coordinate/image exclusions. A new run keeps this prepared dataset in
`data/` under its own output; no old preparation/training semantics change.

An example is a triple (G,F,N): G is a training reference, F and N are other
training nodes, both within34 of G, and d(F,G)-d(N,G)>=0.5 simulation units.
Geometry determines only ordering and eligibility. There is no numerical
distance regression. Single-goal control uses original G=(50,42); multi-goal
uses every admitted fitting node as a reference. Reserve every shifted goal
and matching image from all three roles. No simulated trajectory is implied:
these are manually constructed closer/farther comparisons.

Use the previous ranking objective:
`mean(softplus(0.2-(C(F,G)-C(N,G))/s))`, where C is shared-adapter token MSE.
One positive fixed s, common to both fresh arms, is the median positive raw
MSE over multi-goal training pairs. All three inputs receive gradients. No
anchoring, distance target, distance calibration, clipping or early stopping.
Both fresh models start at identity (seed0), use bottleneck64, Adam lr1e-4,
gradient clip1, 2,000 updates and32 triples/update. Uniformly sample a valid
training reference, then one of its valid F,N comparisons. Persist all offered
triples, sampled triple IDs and per-role exposure counts. Log fixed train-only
probe loss/ranking accuracy and mean costs; do not confuse those with test
metrics. Ranking triples require96 endpoint computations/update, unlike the
previous regression's64; describe regression comparison as a recipe comparison.

Freeze and compare the successful multi-goal distance adapter from
`radial_multigoal_v1/20261106T184946814711Z` (multi branch), checking persistent
input hashes, source lineage, checkpoint hashes and provenance. Its historical
code fingerprint need not match new diagnostics; no old cache is needed.
Require matching source geometry, reserved-goal list and admitted fitting
images. Score all three models and raw features on the same freshly encoded
references/grid, verify reference/source hashes unchanged and original cached
vs fresh goal consistency.

Primary evaluation: common heldout queries and directed adjacent grid edges,
same cohort as existing transfer. Report uphill, ties and correctly decreasing
fractions. Additionally compare all heldout query pairs whose true goal-distance
gap>=0.5: correctly ranked, reversed and tied fractions with denominators.
Equal/small physical gaps are excluded; cost tolerance is the existing
max(1e-7,0.001*common cost range). Original goal is a seen control; summaries
over shifted goals exclude it. Ranking scores have arbitrary units: never
multiply them by34 or report distance MAE/RMSE for ranking models.

Print coverage, fixed training loss/ranking accuracy, per-goal ordinal summary,
and16 full-right heatmaps x35..60,y4..60: raw MSE, single-goal ranking MSE,
multi-goal ranking MSE, frozen multi-goal distance MSE, true Euclidean distance.
Separate full-range color scales per method, clearly labelled; compare shapes
and ordering, not raw color magnitudes. Show all fitting points and goal markers.
No physical-error map for uncalibrated ranking costs.

Separate timestamped `radial_multigoal_ranking_v1` Drive outputs/local cache.
Signed data, triples, checkpoints/reports validate on reuse; missing figures
regenerate after cache deletion. Old source runs remain untouched. Verify with
meaningful unit/integration tests, full suite, actual notebook reporting and
16-goal synthetic workflow. Fresh final review, pinned source and publication.
Actual GPU/scientific results await user execution.

Reference-balanced sampling, fixed final region, matched fresh ranking control,
budget and distance-model comparator are implementation choices in the
requested multi-goal ranking experiment.
