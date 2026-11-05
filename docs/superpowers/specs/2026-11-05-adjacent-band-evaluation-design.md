# Adjacent band evaluation

The user approved evaluating the existing raw cost, original temporal-pair
adapter, and clean-pair adapter on 20 new fixed starts. Work stays in the
existing curriculum-cost-adapter worktree and branch.

The new start rectangle is x in [40,54], y in [22,30], touching the near
rectangle at y=30. Goal (50,42), layout (32,30), planning configuration,
success criterion and frozen model checkpoints are inherited from pilot_v1.
Generate deterministic held-out case IDs and CEM seeds once. All three
conditions receive exactly the same full case payloads. This is evaluation
only, with 60 total episodes and no adapter fitting or experience collection.

Read the original adapter from pilot_v1 and the clean adapter from
clean_pair_v1, validating their original training runtime IDs and hashes.
Record their identities as source provenance, distinct from the new evaluation
runtimes. Validate source integrity before resumed execution and reporting.
Never write into either source directory, including through symlink aliases.
Use the saved clean goal tokens as the shared goal for every condition.

Each condition gets its own output directory and interaction journal under
adjacent_band_v1. Existing atomic per-case stage logic skips complete cases
and retains interrupted attempts in the journal. New evaluation runtime
signatures bind the cases and source identities; source models must match.

Report successes, paired wins/losses and IDs relative to raw, final distances,
primitive actions, failed IDs, and individual trajectories with goal and band
boundaries. Raw competence is >=18/20; each adapter retains competence when it
loses at most two raw successes, interpreted only when raw is competent.
These are operational screens, not statistical claims or guarantees of unseen
state transfer; near trajectories may already visit the adjacent band.

Deliver a standalone Run-all GPU Colab on the existing branch. Preserve
pilot_v1, clean_pair_v1 and prior notebooks. Push after tests and fresh review.
Empirical performance remains pending the user's Colab execution.
