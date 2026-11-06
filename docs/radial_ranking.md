# Hand-positioned radial ranking

Open `notebooks/radial_ranking_colab.ipynb`, select GPU and Run all. Only the
existing encoder checkpoint in Drive is needed. Existing trajectory experiments
are independent. New timestamped outputs live under `radial_ranking_v1`.

The fixed goal is (50,42), wall32/door30. Training uses24rays every15degrees
and radii2,6,10,14,18,22,26,30,34. Cumulative stage limits6,14,22,34 yield
48,90,105,115 valid images. Outside-boundary/cross-wall positions are omitted.
The physical right boundary is about60, not the reset-safe limit56.6.

The adapter starts at identity and later warm starts from its preceding
checkpoint. Each stage uses500updates,batch16,bottleneck64,Adam lr1e-4, fresh
moments and stage seeds0,1,2,3. Farther/nearer pairs include different rays and
exclude equal radii. Softplus ranking is normalized by each stage's positive
raw-cost median; anchoring is absent. There is no test-based model selection.
Full-training-pair accuracy/loss is measured before/after fitting and every50
updates, separately from sampled-batch losses.

A fresh goal image and float32 tokens are saved and reused. All4,668 unique
coordinates are encoded once in batches8. Float32 shards (~5.7GiB) stay on
local Colab disk; compact numeric results, goal pixels/tokens, training features
and checkpoints persist in Drive. Only stage-allowed training rows enter the
optimizer. Pre-encoded future points are not training inputs until their stage.

The fixed probes include integer-spaced radial holdouts (multiple unseen
points between fitted radii), rays at7.5+15k degrees,50consecutive
radius segment families with11samples and offsets0,+/-0.5,+/-1,+/-2. Of these,
45families have all seven signed profiles; offset summaries use that same
cohort at each offset, averaging signs within families. An offered-training-
endpoint subset measures local offset behavior separately from future radii.
All-profile counts, exact/image exposure and nearest fitting distances remain
available. Flat curves count separately. Tolerances reuse local interpolation
metrics: max(1e-7,1e-3 times profile range) for local steps, endpoint-only
contrast for endpoint comparisons.

Directional summaries distinguish both endpoints inside the trained radius,
both outside, and crossing it. Genuinely heldout comparisons exclude coordinate
or fresh-render identity with any offered training position. At stage4 there
are no directional probes beyond34, so outside results are N/A, not perfect.
The full-right heatmap still shows valid corners beyond34 without training.

Each stage exports independent-scale full-right heatmaps, radial curves,
matched offset errors, interpolation examples and training loss, as PNG/PDF.
Radius plots rise in the desired direction; interpolation plots fall toward
the goal. Counts are descriptive, not independent statistical trials.

Configuration, source/model/renderer signatures, input hashes, token shards
and checkpoint lineage are validated before fitting/reuse. Complete stages
resume with the same local cache; interrupted uncommitted fits can restart
deterministically from their incoming checkpoint. A lost Colab runtime loses
the local feature cache; rerun setup for a fresh timestamped experiment while
preserving previous Drive results. No planning, actions or world-model loads.
