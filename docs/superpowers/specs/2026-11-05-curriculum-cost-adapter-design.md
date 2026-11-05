# Curriculum cost adapter: first mechanism pilot

Approved scope: the experiment recommendation in the chat, followed by the
user's request to implement it in an isolated worktree. No results exist yet.

## Question and limits

Can successes collected near one image goal reshape a frozen I-JEPA planning
metric enough to improve planning on farther, previously untrained starts?
This tests the prerequisite for outward bootstrapping, not the benefit of a
complete curriculum. Doorway crossing, multiple goals/layouts, and the fixed
mixture control follow a positive first result. Do not claim novelty here.

## Fixed experiment

- Start from `wall-layout-generalization` in branch `curriculum-cost-adapter`.
- Frozen matched-v3 fixed world model: SHA256
  `b0f3ccfefe0534d9f51af2d0767619913f48d0c323ebb323f0bcd4b800206c3f`.
- Frozen IN1K I-JEPA encoder: SHA256
  `3f37d8a7572e25c2dce9a3072c2fd8cd55ce32dc6edb92a8996b0e4cb140bf98`.
- Pinned DINO-WM commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`.
- Layout `(32,30)`, goal `(50,42)`, success distance strictly below 4.5.
- Near starts: x in [40,54], y in [30,36]. Farther starts: same x range,
  y in [14,24]. Geometry is used for resetting and supervision filtering;
  the learned metric consumes only image tokens. World-model planning also
  uses proprioception. This is not a vision-only or supervision-free system.
- Manifest generated once from a fixed seed: 20 near and 20 farther evaluation
  cases, 40 separate near collection cases per run seed (initially one run,
  then two independent replications if promising). Training starts must stay
  at least 0.25 units from evaluation starts. Same evaluation CEM seeds in all
  conditions. Save explicit case IDs, starts, goals, layout, and RNG seeds.
- CEM horizon 5 macros, execute up to 5 macros before replanning, population
  300, elites 30, iterations 10, candidate batch 64, min std 0.05, final
  `elite_mean` selection, no component bounds, primitive norm cap 3.5.
  Five primitive actions per macro; maximum 20 macros per episode.

## Sequence and gates

1. Identity/raw evaluation on all 40 fixed cases; persist per-episode results.
2. Continue only if near success >= 12/20 (scale threshold with case count for
   explicit smoke configurations). Otherwise position-cost evaluation of the
   same cases is a diagnostic, not adaptation. Never tune on farther results.
3. Collect exactly 40 fresh near attempts with identity cost, saving failures
   and every executed primitive action, state, duration, and real frame.
4. Encode successful trajectories; batch-fit the adapter for a fixed schedule.
   No farther evaluation observations or baseline episodes enter training.
5. Freeze adapter and evaluate all 40 paired cases. Report near retention and
   farther transfer, paired wins/losses, final distances, primitive actions,
   and counts with Wilson intervals. A +4 farther success / <=2 near loss
   first-run screen motivates replication; it is not statistical proof.
6. Reuse the unchanged baseline for run seeds 1 and 2; use independent near
   collection and training seeds. Report each run separately, since reused
   evaluation starts are not independent additional tasks.

The first run is 120 episodes; three runs share baseline and total 280.
Diagnostic candidate replays are counted separately from adaptation attempts.
All stages resume completed per-episode artifacts only when configuration and
model/source hashes match. Mismatched artifacts must fail rather than mix runs.

## Cost adapter

Shared per-token `A(z) = z + Up(GELU(Down(LayerNorm(z))))`, width 1280,
bottleneck 64. Zero-initialize Up weight and bias. Skip path is unnormalized.
Compute costs in float32 as the mean square over patches and channels, applying
the same adapter to predicted terminal and goal tokens. At initialization it
must exactly equal the existing cost, including when batched by CEM. Do not
feed adapted tokens into encoder/world-model histories. Cache adapted goal
tokens for a frozen episode. Encoder and world model never receive gradients.

## Builder decisions for the first training schedule

These are implementation defaults, not user-approved scientific findings.

- Adam, lr 1e-4, 500 updates, 16 pairs/update, clip gradient norm at 1.
  Fixed last iterate; no best checkpoint selection using evaluation.
- Chronological loop erasure within each successful trajectory: when a new
  state is within 0.5 units of an earlier retained state, discard that loop
  and retain the latest visit. Use simulator state only for this filtering.
- Eligible retained pairs have >=2 actual primitive steps between observations
  and >=2 units of endpoint separation. Sample trajectories uniformly, then
  eligible pairs uniformly. Log rejected loops/pairs. Remaining temporal order
  is noisy behavioral supervision, not optimal travel time.
- Let s be the median positive raw goal cost across successful training frames,
  fixed for the run. Loss is softplus(0.2 - (c_early-c_late)/s), plus coefficient
  1.0 times mean squared normalized deviation from raw costs on sampled real
  replay frames. No zero-cost label on successful nonidentical terminal frames.
  A smooth ranking loss retains training signal even if initial ranks agree;
  uniform cost rescaling alone will not improve CEM and does not count as a win.
- Empty successes, zero scale, or no eligible pairs produce an explicit stop
  artifact; do not report a trained adapter or select easier test cases.

## Outputs and execution

Provide a staged command-line runner and Colab notebook with Drive paths for
the matched-v3 checkpoint and IN1K encoder. No historical latent cache restore.
Save protocol/config hashes, source commit and dirty status, model hashes,
manifest, all episode records, successful token replay, final adapter, full
optimizer state, training loss history, stage status, paired evaluation report,
real-frame cost maps, trajectory plots, and optional videos. Candidate audits
compare predicted/real costs on initial samples and final elites, with their
additional simulator interactions recorded separately. Outputs live under a
user-selected output directory, never tracked as repository data.

Tests use small real tensors and a tiny synthetic dynamics model; integration
tests exercise pinned DotWall when available. A local short real-model smoke
check may verify loading/identity/collection plumbing, but full CEM measurements
belong on a GPU in Colab. Clearly distinguish verified software from empirical
pilot results in delivery.
