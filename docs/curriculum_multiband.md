# Sequential curricula with raw/adapted MSE maps

Open `notebooks/curriculum_multiband_colab.ipynb` in GPU Colab and Run all.
Use the same Drive artifacts from the pilot, clean-pair ablation and adjacent
evaluation. Results are written to a sibling directory:
`MyDrive/datasets/curriculum_cost_adapter/multiband_curriculum_v1`.

The two curricula start from the original-pair and clean-pair checkpoints.
They have independent experience buffers and checkpoint chains. Both keep x
in [40,54], goal (50,42) and layout (32,30). They progress through y bands
[22,30], [14,22], [8,14] beneath the original near band [30,36]. The final band
ends at y=8 because the pinned environment's valid center coordinates start
at 7.4. The grid stage checks these environment bounds before evaluation.

## What happens in each band

1. Evaluate the incoming frozen adapter on the band's 20 monitoring cases.
2. Run 40 separate collection attempts with that adapter. Original and clean
   policies use identical starts/seeds but collect their own outcomes.
3. Add successful episodes to that branch's replay, including the original
   successful near episodes and its own earlier successes. Save failures too,
   but do not train from them. Evaluation/grid observations never enter replay.
4. Copy the incoming adapter weights and start fresh Adam moments. Run 500
   updates, batch size 16, learning rate 1e-4, anchor weight 1. Sample a usable
   band uniformly, then an episode and eligible pair. This prevents a large
   earlier buffer from overwhelming current experience.
5. Freeze/save the new checkpoint and evaluate current plus earlier bands.
6. Display raw MSE, adapted MSE and their difference on the shared grid.

Original pairs use the existing revisit/time/separation filters. Clean pairs
also require the later state to be >=0.5 units closer to the goal. Earlier
examples are anchored to raw MSE as in the original fitting objective.
Training copies, rather than edits, the incoming checkpoint. It never
reinitializes the adapter or restores old optimizer moments.

If current-band successes provide no usable pairs, that branch stops. Its
actual failures and journal remain visible; the other branch continues. Poor
evaluation does not change the schedule or secretly select another checkpoint.

## Reading the results

The success/actions matrices have adapter versions as rows and bands as
columns. A version's next-band result is its performance before collecting
there. Same-band and earlier-band results after fitting measure learning and
forgetting. N/A means no completed evaluation, rather than zero successes.

Each comparison prints paired wins and losses relative to raw and the incoming
checkpoint, with case IDs. Offsetting wins do not conceal losses. The inherited
18/20 reference competence and <=2-loss retention screens are annotations.
Repeated cases across versions are monitoring tasks, not independent new
samples or a significance test. At 20/20, inspect actions and final distance.

One raw baseline covers all 80 cases. Each branch schedules 20 initial near
evaluations, 60 incoming current-band evaluations, 120 collection attempts and
180 current/earlier post-update evaluations. This is at most 840 episodes
total without interruptions; stopped branches execute fewer. Actual primitive
actions and interrupted attempts are journaled. Stage durations are printed.

## Reading the heatmaps

The grid contains 176 real rendered positions: x=40,42,...,54 and
y=8,10,...,50. Full float32 tokens are encoded once and cached. All costs use
the saved clean goal tokens, including raw planning. Heatmaps describe real
image costs; they do not evaluate world-model terminal prediction quality.

Raw and adapted MSE share fixed limits [0, maximum raw grid cost] for every
checkpoint. The difference uses fixed symmetric limits around zero. A dark or
bright region keeps the same numeric meaning across updates. Values above the
display range are exposed through colorbar extensions, actual maxima and
saturation counts. All unclipped values are saved in JSON. White points show
training positions; the orange outline marks the most recently learned band;
the star is the goal. Initial checkpoint maps precede the six learned maps.
Rendering/encoding the grid executes zero primitive actions and never supplies
training pairs, anchors or progression decisions.

## Artifacts and resumption

`manifest.json` fixes cases and fitting settings; `experiment.json` references
the immutable source artifacts; `execution.json` binds common source/model and
runtime settings. `cache/goal.*` and `cache/grid.*` hold shared diagnostic
inputs. `raw/` holds the baseline runtime, cases and journal.

Each `branches/original/` or `branches/clean/` has its own runtime, journal,
raw collection episodes, successful replay, immutable `checkpoints/vN.pt`,
fitting metadata under `updates/vN/`, evaluation records under `episodes/eval/`,
and numeric/visual maps under `heatmaps/`. `report.json` and
`curriculum_matrix.png` summarize both branches, including partial/stopped runs.
The report also stores each monitoring case's path, outcome, final distance and
actions. After reporting, `updates/vN/evaluation_paths.png` compares raw,
incoming and updated trajectories on current and earlier bands. The collection
paths and fitting losses appear in `updates/vN/training_and_experience.png`.

Stages validate all cached cases in a stage before new actions. Source hashes,
replay lineage, checkpoint/goal/grid identities and shared runtime settings
are checked; output aliases and descendant symlinks are rejected. Completed
updates are reused; an interrupted fit may restart its 500 updates from the
incoming checkpoint. Interrupted simulator actions remain counted and only
incomplete cases are retried. Use the notebook's pinned source revision to
resume. A branch stopped for lack of usable experience remains stopped.

Every committed episode must match a complete journal attempt with the same
stage, case and primitive action count. A missing or inconsistent journal
blocks further actions. Interrupted attempts and truncated crash lines remain
supported; truncated lines make reported counts lower bounds. A damaged branch
is marked invalid in the combined report while the other branch remains
visible. Valid interaction accounting is retained independently of checkpoint
validation; unavailable accounting produces an unavailable total and an
explicit known subtotal. Existing grid bytes are validated before reporting
or baseline actions as well as before updates.

Download the executed notebook and return it for interpretation. Local tests
exercise the staged loop with synthetic episodes; live GPU outcomes remain
pending until that execution.

## Implementation verification

The final review's three important findings have regression coverage: branch
report isolation, journal reconciliation and baseline/report grid validation.
The full local suite passes 389 tests (plus 16 subtests); the notebook's five
code cells compile. Its runtime is pinned to
`a854e170c114d4977b91171891201111be16a221`.
These checks do not measure live GPU outcomes, full-size replay memory or
Colab/Drive reliability. Failures there can interrupt the run, and successful
execution still may not improve planning. No minor findings were deferred.

## Inspecting saturated maps and vertical profiles

Open `notebooks/curriculum_cost_profiles_colab.ipynb` on a CPU runtime and Run
all. This standalone viewer reads the existing `heatmaps/vN.json` files from
both branches, checks their completed-map hashes and shared grid/raw costs,
and plots every available version. Its common color ceiling covers the
largest saved raw or adapted cost, so high values remain distinguishable.

The vertical profiles fix x at 40, 46, 50 and 54 and plot cost against y from
8 to goal y=42. Read left to right: rising sections assign a higher cost to
positions closer to the goal; falling sections assign a lower cost. Samples
are joined without estimating unmeasured grid points. Real-image profiles do
not establish the ordering of world-model predictions or feasible actions.
PNG/PDF figures are saved in `diagnostics/cost_profiles`; no simulation,
training, checkpoint loading or changes to existing artifacts are needed.

## Checking the nonzero cost at the physical goal

`notebooks/curriculum_goal_consistency_colab.ipynb` runs a separate GPU
diagnostic using the same frozen image encoder and pinned Wall renderer.
It reads the existing multi-band and clean-pair caches, and writes a new
timestamped folder under `goal_consistency_audit_v1`. It executes no actions,
fits no adapter and loads no world-model checkpoint.

First it checks the clean source goal, its copied target, the grid's embedded
target and the exact grid point (50,42). It validates cache hashes and
recomputes the saved raw goal MSE. Then it compares fresh goal/grid/reset
images at the pixel level. Identical pixels are encoded singly and in
eight-image batches, including the original eight-position grid row. Fresh
tokens are compared with both cached representations. All comparisons repeat
under strict FP32 arithmetic with TF32 disabled, recording execution settings.

The diagnostic distinguishes measurements from historical conclusions:
the original goal image and every arithmetic flag were not saved. Fresh
pixel equality cannot prove historical pixel equality, and current arithmetic
differences alone cannot establish the previous runtime's settings. The
executed notebook and `report.json` should be inspected before replacing a
target or changing the experiment. Source hashes are rechecked after probing.

Diagnostic verification: the full suite passes 400 tests (plus 16 subtests).
Regression tests cover mismatched target copies, batch-dependent encoding,
render-route differences, a difference isolated to the goal's batch slot,
and arithmetic restoration for all three precision modes on normal and
exception exits. A local smoke run used the real pinned renderer and a small
encoder to verify report/tensor/figure output with unchanged source hashes.
The real Drive goal/grid tensors and CUDA results remain to be inspected.
The diagnostic notebook pins runtime revision
`7367ee2b92be83e85e49ca696c4051cdce6ecec5`.
