# Near-to-far planning-cost adapter pilot

The first experiment measures raw-cost competence near one image goal. The
second fits an identity-initialized residual metric using new near-goal
successes, then measures transfer to farther starts before collecting training
data from those starts. This is a mechanism pilot, not yet an experiment showing
that a complete curriculum beats a fixed mixture of starts.

Open `notebooks/curriculum_cost_adapter_colab.ipynb` from the GitHub branch in
Colab and select a GPU. Its setup cell clones `kmylonas/i-jepa-wm` on branch
`curriculum-cost-adapter`, prints the exact source commit, and records that
commit in experiment provenance. Run cells in order. The encoder and matched-v3
fixed checkpoint are the only large input artifacts. No source archive transfer
or historical training cache is required.

On a fresh runtime, the notebook clones the current branch. Rerunning setup
fetches the branch and checks out the current remote commit. Set `SOURCE_COMMIT`
to the commit recorded in an earlier run's `runtime.json` to reproduce that
revision. Local source edits stop checkout rather than being overwritten.
Changed Python source requires a new experiment output directory.

## Frozen setup

| Item | Value |
|---|---|
| World model | `wall_matched_fixed_multi_v3/fixed/world_model/best.pt` |
| World-model SHA256 | `b0f3ccfefe0534d9f51af2d0767619913f48d0c323ebb323f0bcd4b800206c3f` |
| Encoder | IN1K I-JEPA ViT-H/14 encoder-only checkpoint |
| Encoder SHA256 | `3f37d8a7572e25c2dce9a3072c2fd8cd55ce32dc6edb92a8996b0e4cb140bf98` |
| DINO-WM commit | `0a9492fa12044b852ae9e001cc74604b79c8bb0c` |
| Layout / goal | wall `(32,30)`, goal `(50,42)`, success distance `<4.5` |
| Near starts | x `[40,54]`, y `[30,36]` |
| Farther starts | x `[40,54]`, y `[14,24]` |
| Evaluation | 20 starts per band, same starts and CEM seeds for every condition |
| Collection | 40 separate near attempts per run seed, including failures |
| Planner | horizon 5, execute 5 macros, 300 candidates, 30 elites, 10 iterations |
| Actions | 5 primitives/macro, norm cap 3.5, no component bounds |
| Episode limit | 20 macros / at most 100 primitives |
| Selection / precision | `elite_mean`, explicit float32 by default |

The manifest is generated once. Training starts stay at least 0.25 units from
evaluation starts. This is a fixed-goal experiment with held-out starts; it does
not establish goal generality. The controller uses proprioceptive dynamics.
Simulator coordinates support reset selection, success detection and revisit
filtering. The adapter receives only image tokens.

## Commands

From the source root, define paths once (these examples use Colab paths):

```bash
python -m src.mylonas_ijepa_wm.curriculum_cost_pilot prepare \
  --output-dir /content/drive/MyDrive/datasets/curriculum_cost_adapter/pilot_v1
```

For execution stages, add these common arguments:

```text
--output-dir /content/drive/MyDrive/datasets/curriculum_cost_adapter/pilot_v1
--world-model-checkpoint /content/curriculum-cost-adapter-artifacts/fixed-best.pt
--encoder-checkpoint /content/curriculum-cost-adapter-artifacts/encoder.pth.tar
--dino-wm-root /content/dino_wm
--device cuda --precision fp32 --encoder-batch-size 8
```

Run stages in this order:

1. `baseline`: 40 raw-cost evaluation episodes. Writes `gate.json`. Fewer than
   12/20 near successes returns exit code 2 and stops adaptation. The near
   criterion is an operational collection gate, not a significance test.
2. If that gate fails, `position` evaluates the same cases with predicted
   position cost as a diagnostic. Inspect the results before redesigning the
   pilot; do not silently change starts, planner, or checkpoint.
3. `collect --run-seed 0`: exactly 40 new near attempts using identity cost.
   Adapter updates happen after this batch, not within collection episodes.
4. `train --run-seed 0`: fit only on successful collection episodes. No baseline
   or evaluation frames enter replay. No successes or eligible pairs stops with
   a reason in `stages/run_0/train.json` and no trained checkpoint.
5. `evaluate --run-seed 0`: freeze the final adapter and use the same 40 cases.
6. `report`: needs only `--output-dir`, no checkpoint paths or GPU. Writes paired
   comparisons and plots. Reports each run separately.
7. `diagnostics --run-seed 0`: optional, after training. Saves real-frame maps
   and candidate audits on the first near and farther cases for raw and adapted
   cost. Eight initial samples, eight final elites, and the returned elite mean
   are replayed per audit. At horizon 5 this adds 1,700 primitive interactions,
   recorded separately from the adaptation budget. Replay continues after
   success to preserve the full terminal horizon used in prediction.

A first-run screen of at least four extra farther successes and no more than
two lost near successes motivates independent runs 1 and 2. Repeat `collect`,
`train`, `evaluate` for each, then `report`. This screen is not a p-value or a
claim of statistical reliability. Baseline evaluation is reused across runs;
60 adapted evaluations of a band still reuse the same 20 tasks.

The first run is 120 episodes. Three adapter runs sharing one baseline total
280 episodes. Collection is at most 4,000 primitive actions per run; actual
durations, including unsuccessful attempts, are reported. `interactions.jsonl`
also records primitive actions from interrupted attempts that must be repeated.
The completed-episode summaries and durable interaction budget are both shown.
An incomplete journal line marks that count as a lower bound. Optional position
and candidate diagnostics have additional, explicitly separate budgets.

## Adapter and fixed training defaults

The shared MLP maps `1280 -> 64 -> 1280` through LayerNorm, linear, GELU, linear,
with an unnormalized residual skip. The final linear layer begins at zero. Its
approximately 168k parameters change terminal cost only. Both predicted and goal
tokens use the adapter, with float32 mean square over all patches and channels.
The exact initial cost and the existing CEM history inputs are covered by tests.

Successful observations retain actual primitive timestamps, including partial
terminal macros. Revisits within 0.5 units are chronologically loop-erased.
Training pairs need at least two primitive steps and two units of separation.
This reduces obvious contradictions but does not make the paths optimal.

The fixed first schedule uses Adam at 1e-4, 500 updates, 16 pairs/update, gradient
clip 1. Let `s` be the median positive raw goal cost on successful training frames.
The objective is:

```text
softplus(0.2 - (cost(earlier) - cost(later)) / s)
  + mean(((cost(anchor_frame) - raw_cost(anchor_frame)) / s)^2)
```

Sample a usable trajectory uniformly, then a pair uniformly; anchor frames are
sampled from the same training replay. Terminal success images receive no
explicit zero-cost target. Save the final iterate and all optimizer state;
there is no checkpoint selection or tuning on farther evaluation performance.
These are implementation defaults for this pilot, not established best values.
Uniformly scaling costs may reduce a loss while leaving CEM decisions unchanged;
only held-out planning results support the mechanism.

## Resuming and outputs

Stages save each completed episode atomically and skip it on rerun. Do not run
two writers against the same output directory. Re-run `prepare` with identical
arguments and re-run the interrupted stage after a session reset. Training
restarts its fixed 500 updates if interrupted before the final adapter commit;
completed collection is reused. A final adapter whose summary was interrupted
restores that summary from its checkpoint without retraining.

Resuming rejects changed model hashes, source code, manifest, Torch version,
device type, precision or encoder batch size. Candidate audits also reject
changed sampling counts. Use a new directory for a changed protocol. Explicit
`prepare` overrides for tiny smoke runs mark the manifest/report `modified`;
they are not pilot results. Hash overrides select a different model artifact
and therefore define a separate experiment.

Expected files:

```text
manifest.json, runtime.json, gate.json, interactions.jsonl
episodes/baseline/*.json
episodes/run_0/{collect,evaluate}/*.json
raw/run_0/*.pt                  # real frames/states/actions, failures included
replay/run_0/*.pt               # successful training tokens only
run_0/adapter.pt                # final weights, optimizer, training provenance
run_0/training.json, training_loss.png, trajectories.png
stages/**.json, report.json, success_by_band.png
diagnostics/run_0/{real_cost_map.*,audit_*.json,summary.json}
videos/**                      # when --videos is explicitly requested
```

Inspect paired wins/losses and videos as well as aggregate success. Map/rank
improvements without planning gains are insufficient. Candidate audits separate
real metric geometry from predicted-versus-real scoring error, including elite
subsets where CEM selects its actions.

## Local verification

Use the existing `pointmaze` Python environment. The full test command is
`PYTHONPATH=.:src python -m pytest -q`. The optional historical DINO comparison
also needs torchvision, einops, accelerate and its released model artifacts.
Task-local optional dependencies may be supplied through an extra PYTHONPATH.

New tests are `tests/test_curriculum_cost_adapter.py`,
`tests/test_curriculum_replay.py`, and `tests/test_curriculum_cost_pilot.py`.
Small synthetic workflows verify software behavior; only the GPU experiment
can establish whether near successes improve farther-start planning.
