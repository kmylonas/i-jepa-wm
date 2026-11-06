# Controlled radial ranking experiment

Approved in chat: replace planning-collected experience with hand-positioned
radial observations around goal (50,42), layout wall32/door30. Use 24 training
rays every15degrees; radii2,6,10,14,18,22,26,30,34. Four cumulative stages have
outer radii6,14,22,34 and about48,90,105,115 valid training positions. Omit
positions outside the physical rectangle or goal-side wall boundary x>35;
never clamp. No planner, simulator actions, old replay or world-model loads.

Fresh identity ResidualTokenAdapter(dim1280,bottleneck64) initially; later
stages warm start the previous adapter. Each fit uses500updates,batch16,Adam
lr1e-4 with fresh moments and a recorded stage seed. Pair all offered training
positions with distinct radii and margin>=0.5, including cross-ray pairs;
farther must cost more. Existing softplus ranking objective, normalized by
that stage's positive raw-cost median, anchor_weight0. No equal-radius losses.

Freshly render/encode a single goal image, save pixels and float32 tokens,
and reuse that exact target at all stages. Frozen encoder and pinned renderer.
Only approved training-point tensors enter fitting; heldout probes never do.
Record actual sampled point counts, eligible pair counts, rank accuracy/ties
and fixed-batch learning curves. Do not stop/select models on test results.

Build one fixed probe design before fitting: radial midpoints, withheld rays
at7.5+15k degrees, and up to50 geometry-selected consecutive-radius families
with11coordinate samples and offsets0,+/-0.5,+/-1,+/-2. Use actual renders, not
latent interpolation. Include farther radii before training reaches them.
Stratify tests as within current radius, fully outside, or crossing it; final
stage extrapolation is unavailable, not zero error. Match offset cohorts
requiring all seven signed profiles and average both sides within each pair.
Keep full-data counts too. Track coordinate and rendered-image overlap with
all offered fitting points and actual sampled inputs, plus nearest distances.
Flat costs are not useful monotonicity. Reuse established curve tolerances.

Every stage prints full-right heatmaps, radial cost curves on training and
withheld directions, interpolation/offset metrics, seen/extrapolation results,
loss histories and examples. Cover the whole right compartment, to just
inside physical x60; independent unclipped raw/adapted scales with numeric
maxima, training overlay and current-radius ring. Test positions are fixed
across stages. Fractions are descriptive, not independent trials.

Encode each unique coordinate once, batch8 fp32; bounded-memory float32 token
shards live on Colab local disk, not Drive. Persist compact numeric results,
training tensors, target pixels/tokens, configuration, checkpoint lineage and
hashes to a separate timestamped radial_ranking_v1 Drive output. Validate cache
hashes/config/source signatures before fitting or reusing completed stages.
Interruptions may restart an uncommitted deterministic fit; never silently
reuse mismatched data. Existing experiments stay read-only and independent.
GPU Colab is pinned, same worktree/branch, source publication authorized.
