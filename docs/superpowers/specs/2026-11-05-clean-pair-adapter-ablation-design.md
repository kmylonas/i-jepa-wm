# Clean-pair adapter ablation

## Intent

Test whether contradictory temporal labels caused the original planning-cost
adapter to degrade performance. Reuse the exact 40 saved successful collection
episodes and change only the rule that selects ranking pairs. This is a
diagnostic oracle ablation, not yet the final curriculum-learning method.

## Experimental comparison

The parent pilot remains `datasets/curriculum_cost_adapter/pilot_v1`. It is
read-only. The clean condition writes to
`datasets/curriculum_cost_adapter/clean_pair_v1` and records the parent runtime,
source artifacts, pair rule, adapter checkpoint and evaluation condition.

Keep the original adapter architecture, initialization seed, 500 updates,
batch size, learning rate, anchoring, successful episodes, goal, world model,
encoder, CEM settings and 20 near plus 20 far evaluation cases. Do not collect
new trajectories or re-encode the 730 replay frames.

Start from the original revisit-filtered temporal pairs. For each pair `(i,j)`,
compute true Euclidean distances from the saved states to the case goal and
retain it only when:

`distance(state_i, goal) - distance(state_j, goal) >= 0.5`.

Continue to sample an episode uniformly, then a retained pair uniformly. Keep
all successful episodes that have at least one clean pair. Report before/after
pair counts and episode IDs. Invalid or nonfinite states, goals or thresholds
must stop the run. The default temporal behavior must remain byte-for-byte
compatible when no oracle margin is requested.

## Execution and provenance

Provide a separate staged GPU Colab notebook. `prepare` validates and references
the complete parent pilot without modifying it. `train` reads the existing
float16 replay tokens and saved states, computes and persists the actual goal
tokens, and saves a new adapter plus training metadata. `evaluate` freezes that
adapter and runs only the same fixed 40 evaluation cases. `report` compares the
new records against the parent raw baseline and prints paired wins and losses,
final distances and primitive-step budgets.

The new runtime may have a different source digest from the parent. Its
provenance must therefore name both runtime IDs rather than pretending that new
code produced the original replay. Resuming a completed stage must validate
the parent runtime, clean-pair rule and checkpoint hash. Original pilot files,
records, adapter and audit outputs remain unchanged.

## Interpretation and screen

The operational screen is defined before evaluation:

- at most two raw-success/clean-failure losses among near cases;
- clean near successes no lower than raw near successes minus two;
- clean far successes no lower than raw far successes.

For the observed parent baseline, these aggregate thresholds are 18/20 near
and 18/20 far. Report paired wins and losses separately so offsetting changes
cannot be hidden by the aggregate count. Passing means the clean condition
avoided the previous degradation; it is not a significance claim. Far results
above 18/20 are improvement evidence for this fixed set, while unchanged far
results are consistent with the audit's zero far-region training coverage.

Save goal tokens so exact raw-versus-clean adapter cost ordering can be audited
later. Do not run the expensive candidate-replay diagnostics automatically in
this ablation.

## Deliverables and verification

Add the optional oracle-progress pair filter to replay fitting, a clean ablation
runner, focused provenance and pairing tests, plain-language documentation, and
a Colab notebook that safely supports Run all. Tests must prove the default
pair behavior is unchanged, wrong/ambiguous pairs are excluded at the 0.5
margin, parent artifacts are not mutated, and report comparisons use identical
cases despite different parent and child runtime IDs.

