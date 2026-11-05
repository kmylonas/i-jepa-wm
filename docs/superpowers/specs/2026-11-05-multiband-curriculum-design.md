# Sequential multi-band cost-adapter curriculum

## Intent and evidence

The user requests a real curriculum: evaluate, collect new experience in the
current band, update the adapter, then move to the next band lower in y. Keep
x in [40,54]. Display raw versus adapted latent-MSE heatmaps after every band
update. The user explicitly selected two separate curriculum experiments,
starting respectively from the existing original-pair and clean-pair adapters.
Use the existing curriculum-cost-adapter worktree and GitHub/Colab workflow.

The executed adjacent-band notebook reports 20/20 successes for raw, original
and clean costs. Total actions are 550, 478 and 504 respectively; mean final
goal distances are 3.912, 3.957 and 3.609. These are fixed-case observations,
not proof that subsequent updates will improve the adapters. Previous far-band
results remain valid for their original starts.

## Geometry and fixed system

Goal (50,42), wall layout (32,30), frozen world model and encoder, success
threshold 4.5 and existing CEM/MPC settings remain inherited from pilot_v1.
Existing near band: y [30,36]. New collection/update stages, in order:

| Stage | x range | y range |
| --- | --- | --- |
| 1 | [40,54] | [22,30] |
| 2 | [40,54] | [14,22] |
| 3 | [40,54] | [8,14] |

The last tile is shorter to stay above the valid reset boundary. The current
environment config gives minimum valid center coordinate 7.4 (border location
5 minus 1 plus twice dot standard deviation 1.7). Eight is the proposed safe
lower endpoint. Confirm this against the pinned runtime before executing.
The padding formula comes from the pinned DotWall implementation:
https://github.com/gaoyuezhou/dino_wm/blob/0a9492fa12044b852ae9e001cc74604b79c8bb0c/env/wall/envs/wall.py#L33

Predeclare 20 fixed evaluation cases per band, including the original near
cases. Reuse the exact 20 adjacent evaluation case payloads from
adjacent_band_v1 for stage 1, read-only. New lower bands use deterministic
starts and CEM seeds. All conditions and checkpoints use identical evaluation
cases. Training collection cases are separate, at least 0.25 units from every
evaluation start, with 40 predeclared attempts per new band and branch. Share
those collection case payloads across branches for comparability, but run each
policy independently and never share its resulting experience with the other.

## The two branches

Original branch begins from pilot_v1/run_0/adapter.pt. It retains ordinary
revisit-filtered temporal ranking pairs. Clean branch begins from
clean_pair_v1/run_0/adapter.pt. It retains the additional oracle progress
requirement distance(early,goal)-distance(later,goal) >=0.5. This preserves the
two existing training recipes; it does not isolate pair quality independently
of the policy-dependent experience distributions.

Seed both replay buffers with the same 40 saved successful near episodes from
pilot_v1. Thereafter each branch adds only its own successful collection
episodes. Save failures and interaction budgets, but do not add failed episodes
to ranking replay. Evaluation trajectories never enter training. Do not use
the raw planner as a rescue collector if the current adapter cannot succeed.
If no current-band success has usable pairs, stop that branch with a clear
insufficient-experience result rather than silently repeating only old data.

## Per-band loop

Keep a raw baseline for every evaluation band, computed once in the new
runtime. It provides a fixed reference for both branches. At each stage:

1. Freeze the incoming adapter and evaluate 20 cases in the current band
   before training there. Report transfer before seeing that band's training
   experience. Evaluate the initial checkpoints on near cases too.
2. Execute the 40 current-band collection attempts with this frozen adapter.
   Save primitive observations, states, actions, actual timestamps and success.
   Encode new successful frames once using the frozen encoder. Do not alter
   the original parent replay.
3. Warm-start from the incoming adapter weights. Add new successes to the
   branch's cumulative replay. Run 500 optimizer updates, batch size 16,
   learning rate 1e-4, anchor weight 1, same residual architecture and ranking
   objective. Use a fresh Adam optimizer at each band, with deterministic stage
   seeds; persist this choice in provenance. Never reinitialize the adapter.
4. Sample a replay band uniformly, then a usable episode uniformly,
   then a pair uniformly. Include all bands through the current one. This is a
   proposed explicit replay-balancing choice to prevent older large buffers
   from dominating. Anchor observations come from all frames in that sampled
   episode, preserving the raw-MSE anchor target. Keep oracle-margin selection
   only on the clean branch. Report per-band/episode pair counts and sample
   counts, old/new episode counts, loss histories and checkpoint lineage.
5. Freeze and save the new checkpoint. Evaluate it on the same 20 current-band
   cases and all prior bands, including near, to measure improvement and
   forgetting. Reuse the incoming checkpoint's most recent evaluations on prior
   bands for before/after comparisons when identities match exactly.
6. Print/save heatmaps for this checkpoint. Then advance to the next stage.

Evaluate all scheduled stages even if performance worsens: the report marks
degradation and retention losses; there is no hidden best-checkpoint selection,
retry, threshold tuning or rollback. Only missing usable current experience,
nonfinite training, incompatible artifacts or execution failures stop a branch.
Heatmaps are diagnostics after a complete 500-update fit, not after every
individual optimizer step. Add initial-checkpoint maps for context.

## Heatmaps

Use one predeclared real-image grid for both branches and every checkpoint:
x = 40,42,...,54; y = 8,10,...,50 (176 positions). This includes all start
bands and the goal (50,42). Render the same fixed layout with the ball at each
position and encode once. Cache full float32 tokens and their source/model,
preprocessing, precision, point-list and content hashes. No grid tokens enter
replay, training, anchoring, checkpoint choice or progression decisions.

At each saved checkpoint display three aligned panels:

- Raw MSE(z,z_goal).
- Adapted MSE(A(z),A(z_goal)).
- Adapted minus raw MSE.

Use exactly the saved clean goal tokens for both branches and raw scoring.
Raw and adapted panels use the same fixed numeric scale across all updates,
from zero to the maximum raw grid cost. Mark values exceeding this scale with
the colorbar extension and print their count and actual maxima so clipping
cannot conceal changes. The difference panel uses fixed symmetric limits at
plus/minus the same maximum raw cost. Save all un-clipped values as data.
Show the goal, all band boundaries, the current band and replay-visited
positions. The raw map must be numerically unchanged between updates. Label
these as costs on real rendered images; they are not predicted-terminal cost
maps and cannot alone establish that the world model ranks plans correctly.
Rendering grid probes executes zero primitive actions and is accounted for
separately from interaction budgets.

## Reporting and experiment size

For every band/checkpoint report successes, paired wins/losses and IDs relative
to raw and the incoming checkpoint, mean final goal distance, primitive actions,
and trajectories. Show an adapter-version by evaluation-band success/actions
matrix to make earlier-band forgetting visible. Mark future-band cells not yet
evaluated explicitly. Plot training losses and actual experience paths per band.

Keep the prior operational 18/20 competence and at-most-two paired-loss screens
as annotations, never as automatic proof of improvement. A checkpoint can be
more efficient at a 20/20 success ceiling; report both actions and distances.
Repeated held-out cases across updates are monitoring cases, not independent
new tasks or a confirmatory benchmark. This is one curriculum run per recipe.

Maximum without interruption: 80 raw-baseline episodes (near plus three new
bands). Per branch, 20 initial near evaluations, 60 pre-update current-band
evaluations, 120 collection attempts and 180 post-update evaluations across
current/prior bands: 380 episodes. Both branches total 840 episodes including
raw. Cached complete records resume without additional actions. Print actual
interaction budgets and interruptions; do not promise runtime from the small
adjacent evaluation alone.

## Execution and provenance

Create a standalone staged GPU Colab in the same Git worktree/branch. Default
Drive output: datasets/curriculum_cost_adapter/multiband_curriculum_v1, a sibling
of pilot_v1, clean_pair_v1 and adjacent_band_v1. Read all three existing runs
without modification. Validate source hashes and training/runtime identities;
new evaluation/collection records name their actual runtime and checkpoint.
Bind each update to its incoming checkpoint hash, exact replay subset, fitting
schedule, band and pair rule. Freeze adapters throughout each entire stage.

Use independent branch/stage directories and immutable checkpoint files.
Preflight all existing case records and journals before taking more actions.
Retain interrupted primitive actions; retry incomplete cases only. Prevent
source writes through path aliases. Completed training resumes only after
checking checkpoint, goal/cache and replay hashes. An interrupted fit may
restart its fixed 500 updates from the incoming checkpoint; count no simulator
interactions for fitting. A stopped branch must not block the other's results.
Persist goal/grid tokens and numeric heatmaps for later audit. Display progress
and each stage's elapsed time. Pin the notebook source revision before delivery.

## Proposed implementation choices awaiting design review

The user fixed downward tiling, x range, both independent curricula, repeated
experience-driven updates and heatmaps. The exact bottom endpoint, collection
count, replay balancing, optimizer reset, heatmap cadence/grid and diagnostic
progression are the concrete proposed defaults above. Architecture, world model,
encoder and training loss are inherited; no separate architecture experiment.

A fresh builder review found nothing important left to resolve before planning.
The design is ready for user approval; implementation has not started.
