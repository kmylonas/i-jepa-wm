# Adjacent band evaluation

This evaluates the frozen raw cost, original adapter and clean-pair adapter
on one set of 20 held-out starts: x [40,54], y [22,30]. It touches the near
training band, y [30,36], at y=30. The goal is (50,42), layout (32,30).
All three conditions use identical case payloads, CEM seeds and planning
settings inherited from the parent. There are 60 evaluation episodes.

Open `notebooks/curriculum_adjacent_band_colab.ipynb` in GPU Colab and Run all.
It reads `MyDrive/datasets/curriculum_cost_adapter/pilot_v1` and
`clean_pair_v1`, and writes a sibling `adjacent_band_v1` directory. It reuses
both trained adapters and the saved clean goal tokens. Evaluation encodes
current images as usual; there is no fitting, collection or replay encoding.

`prepare` freezes the 20 cases and source hashes. `evaluate --condition raw`,
`original`, and `clean` write independently under `conditions/<name>`, using
the existing atomic case records and primitive-action journals. Completed
cases resume; interrupted actions remain counted. `report` checks all source
and case provenance before comparing. Changed source files or runtime settings
are rejected rather than mixed. The notebook pins its verified source commit
for exact resumption even if the GitHub branch later changes. Raw cost uses an
explicit MSE scorer against the same saved target as both adapters. Cached
records and journal identities are checked before additional actions. Output
trees with descendant symlinks are rejected, including dangling links.

The report includes successes, failed IDs, paired wins/losses and their IDs,
mean final distance, primitive actions, journals, and every case's results.
Raw competence requires at least 18/20 successes. Each adapter retains
competence when it loses at most two raw successes; offsetting wins cannot
hide losses. When raw is below 18/20, retention is reported as null. These
are operational screens, not significance claims. At 20/20 success, use
actions, distance and trajectories to assess efficiency.

Figures show success/distance/actions and 20 separate paired trajectory
panels, with starts, goal, success circle and the two adjacent rectangles.
They connect macro-boundary positions, not every primitive position. The new
starts are held out, but training paths could have visited this band: this is
an adjacent-band evaluation, not proof of transfer to entirely unseen states.
Earlier far-band findings remain valid for their original cases.

The source adapters are validated against their original training runtime
IDs. Each new evaluation runtime has its own ID bound to this manifest and
source snapshot. A new source digest does not relabel the training artifacts.

Download the executed notebook and return it for interpretation. Live GPU
performance is established by that run; local tests use synthetic episodes.

Final review found three integrity gaps: raw target forwarding, descendant
links and cached-record preflight. All were fixed with failing-then-passing
regression tests. The full suite passed 364 tests with four existing warnings.
Live Colab installation/Drive access, GPU outcomes and bitwise reproducibility
across hardware remain unverified locally. A remote setup difference may need
correction. The source training procedures and 18/20 and two-loss thresholds
are inherited approved choices; interpret counts, distances and actions as
well as the screen if those choices prove too coarse.
