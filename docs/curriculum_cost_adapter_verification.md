# Pilot software verification — 2026-11-05

This records software checks, not results of the scientific pilot.

## Tests

Final command, using the existing pointmaze environment and optional dependencies
installed into the ignored worktree-local `.superpowers/deps` directory:

```bash
PYTHONPATH=.superpowers/deps:.:src \
  /Users/kmylonas/miniforge3/envs/pointmaze/bin/python -m pytest -q
```

Result: **331 passed, 16 subtests passed**, four upstream warnings (unavailable
xFormers and a GLFW initialization warning). All 22 new adapter/pilot tests pass.
The suite includes the historical released DINO-WM model tests using shared
local read-only input artifacts. No test modifies those checkpoint files.

The notebook's nine Python code cells compile. CLI help and a tiny modified
`prepare` configuration run successfully. The delivered source archive also
receives a separate extraction/import/prepare/provenance check.

## Actual frozen-model smoke check

A bounded local MPS check used the selected SHA-verified matched-v3 fixed world
model, IN1K encoder, pinned DotWall, and explicit 256x1280 tokens. Four candidates,
one iteration, horizon/execution one macro and a one-macro episode limit were
used solely to exercise the software path. This is not the pilot's search budget.

- Raw and identity-adapter CEM returned exactly identical actions and cost.
- Identity-adapter cost exactly matched raw MSE on real encoded frames.
- A short *directed* simulator trajectory supplied four real smoke frames for
  three adapter optimization steps; it is not planner-collected pilot experience.
- Adapter checkpoint save/restore worked; restored weights were frozen.
- Encoder and world model remained frozen.

No full near-goal competence rate, adaptation benefit, outward transfer, or
curriculum result has been measured. Those require the supplied GPU protocol.

## Independent review

A fresh reviewer checked the whole branch. Four substantive findings were each
reproduced by a failing test and corrected before the final full suite:

1. Transfer CPU checkpoint normalization statistics to the diagnostics GPU.
2. Apply retention screening to paired near-case losses, not net success change.
3. Journal audit primitive actions durably and include interrupted/repeated
   diagnostic attempts in their separate budget.
4. Give shared evaluation cases distinct video directories for each adapter run.

No review findings were deferred.

## Implementation decisions

- The native worktree tool could not resolve the nested repository; Git created
  the explicitly requested worktree instead.
- Initial baseline tests lacked optional dependencies. Implementation continued
  with relevant tests; task-local dependencies and artifact links subsequently
  enabled a green full suite without altering the shared Python environment.
- The ranking/anchor design was instantiated with 500 Adam updates at 1e-4,
  batch 16, anchor coefficient 1, two-step/two-unit pair separation, and a
  0.5-unit revisit filter. These defaults are fixed and logged, not tuned on
  farther evaluation; a negative result may motivate a separate loss ablation.
- The branch and worktree are preserved for running/reviewing this pilot.
  The source archive provides Colab execution without publishing Git history.
