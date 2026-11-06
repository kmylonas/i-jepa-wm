# Multi-goal distance supervision within the fitted region

The user approved the recommended next experiment: keep the same region and
adapter, supervise distances between positions, and reserve goals and query
positions from fitting. Work in the existing curriculum-cost-adapter worktree;
preserve all previous experiments and publish a separate runnable Colab.

## Experiment

Use the final R34 radial distance source run and its validated geometry and
historical adapter, without changing that run. Freshly encode the original
116 candidate positions (115 radial positions plus original goal), the full
right-side grid, and the 15 shifted goals from the far sweep. Reuse these tokens
within this run. Same frozen encoder, renderer, float32, batch size 8. Changing
the environment goal must leave fixed-position images unchanged.

Reserve all 15 shifted goal coordinates AND matching rendered images before
constructing training pairs. Remove any candidate fitting node that aliases
one of those goals. Reserve grid queries by coordinate and image against all
offered fitting nodes and every evaluation reference. Same heldout mask and
same all-goal common region for every method, as in the far-goal sweep. The
original goal (50,42) is an explicitly seen-reference control; only shifted
goals count as goal-transfer tests.

Two fresh identity-initialized ResidualTokenAdapters, same seed 0, bottleneck
64, Adam lr 1e-4, 2,000 updates, batch size 32 pairs, gradient norm clip 1.
No anchor loss. Single-goal control samples pairs from candidate radial nodes
to the original goal. Multi-goal model samples unordered distinct node pairs
uniformly with replacement from all available pairs of Euclidean distance
0 < d <= 34. The same symmetric adapted token MSE is fitted to d/34 with
squared regression error. Both endpoints are adapted with gradients. Both
arms compute 64 endpoint embeddings/update; endpoint exposure distributions
necessarily differ and are reported. Entire final region is offered at once:
this isolates goal supervision with a matched control, not the old curriculum
schedule. The 2,000-update budget matches the old four-stage total, without
claiming the historical run is a controlled comparison.

Freeze train-only probes (up to 256 pair examples) before fitting, log their
loss/physical MAE/RMSE at update 0 and every 100 updates. Label those as fixed
training probes, distinct from stochastic minibatch loss and heldout metrics.
Never use test goals/queries for fitting, calibration, early stopping or model
selection. Save pair populations and sampled endpoint exposure counts.

## Evaluation

Evaluate raw MSE, fresh single-goal adapter, multi-goal adapter and historical
adapter on the same reference/grid tokens. Estimate distance as 34 * adapted
MSE; fit one nonnegative zero-intercept raw scalar on multi-goal TRAINING
pairs only, shared across goals. Save the historical raw scalar separately;
primary raw baseline uses the new training-pair scalar and is labelled so.
Use existing goal-transfer measure() for identical heldout common MAE/RMSE
and directed adjacent-edge uphill/tie counts. Main distances remain <=34
from every goal and inside original R34 disk. Full-domain heatmaps include
outside-common-region probes and are labelled accordingly.

Per goal print a full-right x35..60, y4..60 heatmap with raw MSE, fresh
single-goal MSE, multi-goal MSE, true distance/34, multi-goal physical error.
Single/multi/true share a full-range scale without clipping; raw/error have
separate scales. Overlay offered fitting nodes (white), original fitting
circle, original goal and tested goal. Also print fitting-node/test-goal
coverage, fixed training losses and common-cohort per-goal MAE/uphill summary.
Print all 16 goal rows with seen/withheld status and method names.

## Artifacts and execution

Separate timestamped radial_multigoal_v1 Drive output and local token cache.
Signed preparation, checkpoint and report artifacts include source lineage,
code/runtime signatures, configuration, pair and split populations, source
consistency, exposure counts and hashes. Completed steps validate hashes and
inputs before reuse; changed settings or partial outputs require a new run.
Missing plots regenerate without encoding or fitting. Persistent training
tokens and goal reference tokens survive local cache removal; report/plots
do not depend on the cache. Validate source hashes before/after execution.

New focused modules for data/fitting, orchestration and plotting; no changes
to previous radial or transfer semantics. Unit tests prove pair exclusion,
distance targets/gradients, improvement on synthetic tokens, matched budgets,
reuse/corruption handling and heldout metrics. Synthetic full workflow and
actual notebook report cells are verified locally. Actual GPU results await
user execution; no claim of arbitrary-goal or planning success.

Implementation choices above (matched fresh control, one final region,
distance cap, budget and plots) are agent choices within the approved scope.
