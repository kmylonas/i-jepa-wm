# Frozen multi-goal ranking adapter in world-model planning

User requests world-model planning with the latest multi-goal ranking
adapter versus raw latent MSE, multiple starts/goals on the trained side of
the wall. Keep the existing worktree and published experiments. New Colab
evaluates only; no fitting, replay collection, or parameter updates.

## Fixed experiment

Load the multi branch of
`radial_multigoal_ranking_v1/20261106T195304628603Z` read-only. Verify persistent
ranking/dataset inputs, old source lineage, frozen distance-reference lineage,
checkpoint hashes and provenance. Historical diagnostic source fingerprints
may differ; they are provenance, not a reason to refit. No local fitting
cache is required. A new read-only loader must not relax the training loaders.

Default goals: (50,42) seen original control; (46,38), (40,18), (50,18),
(56,10), (50,8) withheld references from the latest experiment. Eight starts
per goal: two at distance[6,10), three[10,18), three[18,30]. Fixed geometry-only
seed20261106; fixed per-case CEM/reset seeds. Both endpoints must have x in
[38,56], y in[8,55], lie inside original R34 disk about(50,42), and remain
strictly right of x35. Pair distance<=30. Starts must be >=0.5 from all
offered fitting nodes/references and other sampled starts. Goal0 is labelled
seen; others must match source reserved goals. No outcome-based task selection.

All48 cases are paired across raw/adapted, total96 episodes. Same encoder,
world model, CEM random seeds, reset and success criterion. Later candidate
values may differ as the costs select different elites. Use existing fixed
world model SHA b0f3ccfefe0534d9f51af2d0767619913f48d0c323ebb323f0bcd4b800206c3f,
encoder SHA3f37d8a7572e25c2dce9a3072c2fd8cd55ce32dc6edb92a8996b0e4cb140bf98,
renderer commit0a9492fa12044b852ae9e001cc74604b79c8bb0c, fp32.

Reuse existing PLANNING: horizon5 macros, execute5 before replanning,
population300, elites30,10 CEM iterations, candidate batch64, min_std0.05,
elite_mean selection, unbounded components with primitive norm cap3.5,
max20 macros (100 primitive steps). Success is simulator distance<4.5.
Proprioception conditions the frozen world model; it is not a position cost.
Cost alone changes: raw terminal-token MSE versus MSE after applying the
frozen ranking adapter to predicted terminal and goal tokens. Histories and
world-model inputs remain raw. No oracle costs or candidate position masks.

Fresh marker-free goal references are prepared per actual goal, encoded once
and shared bit-for-bit across cost conditions. Record image/token hashes and
original cached/fresh consistency. The runtime must never fall back to the
old single hard-coded goal cache. Verify each reset renders the intended goal.
Reference/runtime encoder/renderer/source/precision/device signatures match.

Endpoints are constrained; physics is unchanged. Record every primitive
position/action and flag any path leaving the original fitting disk, physical
right-side domain or crossing the wall. Main success includes all attempted
cases; also report successes achieved without leaving the fitting region.
Do not discard excursions or apply asymmetric action clipping.

## Outputs and reproducibility

Separate timestamped `ranking_planning_v1` Drive output. Prepare fixed manifest
and source snapshots, then shared goal references, then raw/adapted evaluation,
then report. Persist each completed case and a primitive-action journal so
interrupted evaluation resumes only missing cases and counts repeated attempts.
Validate source/checkpoint/model/runtime/manifest and case/result hashes before
reuse. Report works after local caches vanish and does not load GPU models.
Completed reports can regenerate figures. Reject mismatches and overlapping
source/output/encoder/world-model paths. Keep frozen adapter/source hashes
unchanged before/after evaluation; freeze model/encoder and avoid gradients.

Per goal, distance range, overall and withheld-only: success counts/rates,
paired wins/losses/both succeed/both fail, final distance, primitive steps,
path length, region excursions and in-region successes. Efficiency comparisons
must also use the subset where BOTH succeed. Include all per-case outcomes,
fixed settings, seeds, journals and elapsed times. No operational improvement
gate or significance claim from48 spatially correlated cases.

Print case coverage, comparison summary, per-goal trajectory panels (all8
paired paths), and paired final-distance/step plots. Use x30..60/y4..60 plots
to show wall and excursions. Save complete trajectories numerically; optional
videos use goal markers only in visualization frames, never encoder images.
Default videos off to keep the benchmark compact. An initial matched pair in
the notebook measures actual runtime before finishing the full evaluation;
its completed episodes are reused, not repeated.

Verify tiny real-world-model/CEM integration, per-goal scorer/seed pairing,
domain/holdout case generation, interruption/reuse/corruption, accounting,
report plots and actual notebook cells. Full local suite and fresh review,
then pinned source publication. Actual GPU planning results await user Colab.

Six goals, eight stratified starts, reused planner settings, excursion reporting
and default videos off are implementation choices within the user's request.
