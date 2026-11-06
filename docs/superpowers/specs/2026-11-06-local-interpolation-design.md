# Local spatial generalization diagnostic

Approved in chat: test the clean no-anchor v0 adapter on 50 actual fitting
pairs, 11 rendered coordinates between endpoints, and parallel offsets
0, +/-0.5, +/-1, +/-2 units. Keep the existing goal/layout/encoder and compare
raw MSE on the same fresh tokens. No fits, world-model loads or episodes.

Reconstruct actual sampled pairs from the fitter seed, replay order, eligible
pair rules and reserved anchor RNG draw. Select up to 50 distinct pairs,
spread across episodes and distance bins, without consulting adapter costs.
Require decreasing actual goal distance throughout each unobstructed segment.
Do not hide endpoint ranking failures. Report candidate/filter/selection counts.

Track nearest distance to ALL fitting input positions (including the fixed
goal), exact rendered-image identity, and conservative replay coverage
separately. Distinguish previously covered from unseen observations and count
unseen adjacent comparisons separately. New positions are physical coordinate
interpolations, never latent interpolation. Label smooth flat costs as ties.

Metrics: endpoint accuracy, increasing/decreasing/tied local steps, magnitude
of uphill motion normalized by endpoint contrast, reversal frequency by
offset and nearest-training-distance bins. Use tolerance max(1e-7,1e-3 times
the cost range of that method/profile). Flat curves cannot pass as informative.
Endpoint order uses max(1e-7,1e-3 times absolute endpoint contrast),
independent of interior ridges, for consistent replay/fresh checks.
Show raw values, pair-relative curves, worst cases and a geometry-selected
sample; results are descriptive, pairs within an episode are dependent.

Write timestamped standalone outputs outside all source trees. Validate v0
checkpoint, goal, replay, encoder and pinned renderer; never construct a
training runtime or modify source reports/caches. Chunk fresh encoding in
batches of eight and persist numeric costs plus provenance, not huge tokens.
Check freshly re-encoded A/B against replay-token scores and flag reversals.

Heatmaps cover x=40 to just inside the physical right border (about x=60),
y=8..50, with unclipped numbers and independent raw/adapted color scales.
Draw the conservative reset-safe edge separately: it is narrower than the
physical rollout domain. New curriculum grids default to this full right
domain; legacy caches retain their exact old coordinates and remain readable.

Deliver a GPU Colab on the same GitHub branch and worktree, pinned to verified
code. Print endpoint/offset/coverage summaries, curves, local zooms and the
full map. Empirical GPU outcomes await user execution. Fixed-goal transfer,
new fitting supervision and world-model action audits remain later questions.
