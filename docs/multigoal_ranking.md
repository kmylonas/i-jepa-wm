# Manual multi-goal ranking

Open `notebooks/radial_multigoal_ranking_colab.ipynb`, select GPU, Run all.
Keep the existing original radial distance run and the completed multi-goal
distance run on Drive. New results use `radial_multigoal_ranking_v1`;
previous runs/notebooks/weights remain unchanged.

Use the same manually positioned candidates in the final R34 region.
Reserve the same 15 shifted goal coordinates and image aliases from all
fitting roles. No planner, trajectories, actions or collected experience.

Each training triple (G,F,N) has distinct admitted nodes, both queries at
most 34 units from G, and F at least 0.5 farther from G than N. Geometry
determines ordering/eligibility only. Fit:

`mean(softplus(0.2 - (MSE(adapter(F),adapter(G)) - MSE(adapter(N),adapter(G)))/s))`

One fixed training-only positive raw-MSE median `s` is shared by both arms.
It sets optimization units; it is not a calibration to physical distance.
All three inputs receive gradients. There is no anchor or distance target.

Single-goal ranking uses original G=(50,42). Multi-goal ranking samples all
admitted training references uniformly, then a valid farther/nearer comparison
uniformly for that reference. Fresh identity initialization, seed 0, bottleneck
64, Adam lr 1e-4, gradient clip 1, 2,000 updates, 32 triples per update. Save
offered triples, sampled IDs and reference/endpoint exposure counts. Fixed
training probes log loss, ordering accuracy and cost magnitudes.

Compare both fresh ranking adapters, raw MSE and the frozen multi branch of
`radial_multigoal_v1/20261106T184946814711Z`. Frozen baseline loading verifies
source/data/checkpoint hashes and matching fitting images; historical code
fingerprints can differ. Old local tokens are unnecessary. Ranking updates
compute 96 endpoints versus regression's 64; this is a fitting-recipe
comparison, not a pure compute-matched loss ablation.

All models/goals share heldout query coordinates/images and common region:
inside original R34 and within 34 of all test goals. Primary metrics are
uphill/tied adjacent approaching moves and correctly/reversely/tied heldout
query pairs with physical distance gap >=0.5. Equal/small gaps are excluded.
Tolerance is max(1e-7,0.001*common cost range), matching prior diagnostics.
No MAE/RMSE in physical units is assigned to uncalibrated ranking scores.
Seen original reference is excluded from summaries over 15 shifted goals.

Print coverage, fitting loss/accuracy/magnitudes, ordinal summary and 16
full-right x35..60/y4..60 maps. Panels: raw, single-ranking, multi-ranking,
frozen multi-distance MSE, true physical distance. Each has its own unclipped
scale: compare shapes and ordering, not color magnitudes across panels.

Input data live in the new run's `data/`, reusing the established preparation
pipeline without changing its semantics. Ranking configuration, triples and
checkpoints are separately signed at the run root. Encoded tokens are reused
locally; persistent reports and missing figures work after cache deletion.
Completed artifacts validate on reuse; partial steps require a new timestamp.
Actual GPU/scientific results await the user's Colab execution.

## Verification

The notebook pins source to `a917f166301e40a027a0d904d4c8035afde16a1c`.
The final full suite passed 484 tests and16 subtests, with four existing
warnings. Synthetic verification exercised all16 goals, real train/evaluate/
report CLI operations, actual notebook reporting cells and19 PNG/PDF figures,
including regeneration after deleting local tokens. The synthetic split had
5,394 single-goal and642,038 multi-goal triples; actual image exclusions may
change counts. Fresh review found no scientific correctness blockers; its
two reporting completeness notes were fixed with RED→GREEN tests for
per-role exposure and directional tie/decreasing fractions. No old source
files or earlier experiment results changed.
