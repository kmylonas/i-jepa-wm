# Clean-pair adapter ablation

This experiment tests one explanation for the original adapter degradation:
some chronological training pairs said that a later frame was preferable even
when the ball had moved farther from the goal.

The ablation reuses the original run-0 replay. It begins with the same revisit,
time-gap and position-separation rules, then keeps a pair only when the later
saved state is at least 0.5 simulator units closer to the case goal. State
distance is a diagnostic oracle for this ablation. It is not yet proposed as
the final curriculum method.

Open `notebooks/curriculum_clean_pair_ablation_colab.ipynb` on the
`curriculum-cost-adapter` branch and select a GPU runtime. The notebook reads:

`MyDrive/datasets/curriculum_cost_adapter/pilot_v1`

and writes:

`MyDrive/datasets/curriculum_cost_adapter/clean_pair_v1`

The parent directory remains unchanged. No collection episodes or candidate
diagnostics run. Training reads the existing float16 replay tokens, computes
the frozen goal tokens once, fits one clean-pair adapter for 500 updates, and
evaluates the same 20 near plus 20 far cases. The new evaluation has its own
durable interaction journal and runtime identity.

The report compares three conditions: raw cost, the original temporal-pair
adapter, and the clean-pair adapter. Passing the operational screen requires no
more than two paired near losses, near success no lower than raw minus two, and
far success no lower than raw. For the observed pilot this means at least 18/20
near and 18/20 far. Passing means label cleaning avoided the earlier damage; it
is not a significance test. Far performance above 18/20 is improvement on the
fixed cases. Unchanged far performance remains compatible with the audit's
finding that the replay never visited the far-start region.

The child directory records the parent runtime and adapter hashes, every replay
file hash, the exact pair rule, the newly computed goal-token hash, the child
runtime and the clean adapter hash. Rerunning completed stages validates these
identities and resumes complete artifacts instead of mixing conditions.
