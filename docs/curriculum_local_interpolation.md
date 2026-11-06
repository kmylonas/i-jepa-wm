# Local spatial generalization

Open `notebooks/curriculum_local_interpolation_colab.ipynb`, select a GPU and
Run all. It reads the completed `multiband_no_anchor_v1` and its three source
folders from the same Drive locations as before. New timestamped outputs are
written under `local_interpolation_v1`. Only the image encoder is loaded;
there are no fitting updates, world-model loads or simulator actions.

The clean no-anchor v0 adapter is the first, simplest checkpoint to inspect.
The diagnostic reconstructs actual sampled fitting pairs, including reserved
anchor RNG draws, from the saved seed and replay order. It selects up to 50
pairs across episode/separation groups before inspecting adapted costs. Each
pair has a valid original straight segment approaching the goal; retained
parallel offsets are 0, +/-0.5, +/-1 and +/-2 units, with 11 positions each.

The ball coordinates are interpolated and freshly rendered/encoded. Raw and
adapted MSE use identical fresh tokens and the retained saved goal. Endpoint
scores on replay tokens are also compared with fresh rendering to expose a
possible replay/rendering/encoding confound. The goal-reference inconsistency
is deliberately not corrected.

The diagnostic distinguishes exact fitting input positions, conservative full
replay coverage, nearest distances, and fresh images identical to a render at
any fitted position. These image checks do not establish identity with unknown
historical target pixels. The fixed goal counts as a fitting reference.

Metrics distinguish uphill, downhill and tied steps, endpoint ordering,
informative monotonicity and flat curves. Tolerance is max(1e-7, 1e-3 times that
method/profile's cost range). Endpoint ordering uses its own tolerance,
max(1e-7, 1e-3 times absolute endpoint contrast), independent of interior
ridges; replay/fresh comparisons therefore share one endpoint rule.
Both endpoints must be unseen for the unseen
comparison summary. Uphill magnitudes are saved in raw cost units and divided
by absolute endpoint contrast when nonzero. Counts are descriptive, not
independent statistical trials.

Plots include full-right maps with independent raw/adapted full-range scales,
offset summaries, distance-to-fitting error summaries, raw-valued pair curves,
relative center curves and local measured-point zooms. The first three example
pairs come from geometry selection; additional worst cases are labeled.

New curriculum grids extend to just inside the physical right border (about
x=60), rather than stopping at x=54. The conservative reset-safe edge around
x=56.6 is marked separately. Legacy caches preserve their original coordinates
and remain readable; the diagnostic writes its own expanded maps.

Existing checkpoints alone cannot prove a deliberate spatial holdout. If most
probes are covered by training, a later sparse refit with withheld neighborhoods
will be needed. Transfer to other goals is a separate experiment.

## Verification

425 tests passed (four existing warnings, 16 subtests). Synthetic integration
exercised the real CLI, source immutability checks, fresh scoring, notebook
result cells and six PNG/PDF plot exports; GPU/encoder/renderer boundaries and
local display were substituted for that workflow. The actual pinned renderer
also rendered x=59.999 and neighboring coordinates successfully. All six
notebook code cells compile. Real GPU encoding, Colab setup and empirical
results remain pending user execution.
