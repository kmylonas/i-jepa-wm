# Multi-goal distance fitting

Open `notebooks/radial_multigoal_colab.ipynb`, select GPU and Run all. The
existing `radial_distance_v1/20261106T101715235909Z` Drive run and encoder
checkpoint are prerequisites. Outputs use a new `radial_multigoal_v1` folder;
previous notebooks, results and source weights remain intact.

Two fresh identity adapters fit the same final R34 region at once, using the
original 115 radial candidate positions plus original goal (50,42). Before
either arm fits, remove candidates matching any of the 15 reserved near/far
goals by coordinate or rendered image, and deduplicate identical fitting
images. Print the admitted nodes and pair counts.

- Single-goal control: each radial position paired with the original goal.
- Multi-goal: unordered distinct node pairs, distance <=34, uniform sampling.

Both minimize `mean((adapted_pair_MSE - Euclidean_distance/34)**2)` with
2,000 updates, 32 pairs/update, Adam lr 1e-4, bottleneck 64, seed 0, no anchoring.
Both endpoints receive gradients. Each arm computes 64 endpoint embeddings
per update; exposure distributions differ intentionally and are saved.
Fixed train-only probe losses are logged separately from minibatch loss.
The historical four-stage adapter is an additional descriptive comparator,
not a matched-budget/schedule control; it may have seen current test-goal
images during its original fitting. The new adapters have not.

Reserve the same eight nearby and seven distant goals as the far sweep:
(40,18), (50,18), (56,18), (40,10), (50,10), (56,10), (50,8) are the distant
ones. Goal 0 is the seen original reference; the other 15 are withheld. Queries
exclude all fitting/reference coordinate and image aliases. Main metrics
use one shared cohort across every goal/method, inside original R34 and
within 34 of every goal. Report MAE, RMSE, uphill and ties. The raw baseline
fits one shared scalar on multi-goal training pairs only; old raw calibration
is recorded separately. No test calibration, early stopping or model choice.

Print coverage, fitting losses, per-goal error/direction summary and 16
full-right heatmaps (x35..60,y4..60). Each heatmap shows raw, matched single,
multi, true distance/34 and multi physical error. Single/multi/true share a
full-range scale; raw/error have separate scales. Full maps include probes
outside the main cohort; white points are admitted fitting nodes.

New preparation encodes roughly 500 positions once in batches of 8, float32, rather
than needing the old local token cache. Frozen encoder preparation and fitting
run in separate processes to release encoder GPU memory. Drive retains
training/reference tokens, input/split/pair populations, exposure counts,
checkpoints, provenance, numerical reports and PNG/PDF figures. Reports and
missing figures work after local cache deletion. Completed artifacts validate
hashes/source/config before reuse; partial steps require a fresh run.

Local verification uses synthetic encoding and actual fitting/evaluation/
notebook report code. Scientific transfer and GPU runtime await the user run.

## Verification

The notebook pins production source to
`b1f3e57a68a78530be775879b085772b48ff1050`. The final full suite passed:
472 tests, 16 subtests and four existing warnings. The 16-goal synthetic
workflow exercised the real training/evaluation/report CLI, actual notebook
report cells, all 19 PNG/PDF figures, and regeneration after cache deletion.
Synthetic counts were 115 admitted nodes, 114 single-reference pairs and
6,232 multi-goal pairs; actual rendering may change image-alias exclusions.
A fresh review found one empty-directory handling defect; two regression
tests reproduced it and passed after the fix, followed by the green full
suite. No scientific leakage or metric correctness findings remained.
