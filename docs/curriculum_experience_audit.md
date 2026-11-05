# Auditing the saved curriculum experience

This audit answers whether the 40 successful collection episodes are useful
training experience. It reads the saved primitive states, actions and RGB
frames. It does not load either learned model, interact with the simulator, or
train the adapter.

For each episode it measures:

- how often one primitive action clearly reduced or increased goal distance;
- path efficiency, comparing the travelled path with the shortest path to the
  4.5-unit success region;
- stationary steps, consecutive action reversals and cumulative movement away;
- which regions were visited;
- whether the exact temporal pairs admitted by the training filter agree with
  the intended label that the later image should be closer to the goal.

The pair report first computes a fraction inside each usable successful episode
and then averages those fractions. This matches the fitter, which samples an
episode uniformly and then a pair uniformly. A separate pooled-pair result is
included in JSON because it answers a different question and gives long
episodes more weight.

The default 0.25-unit tolerance calls very small distance changes unclear. A
later frame farther by more than that tolerance is a contradictory pair for
this same-side, unobstructed collection setup. A detour through a doorway would
need a geometry-aware progress measure and should not be judged this way.

Run `notebooks/curriculum_experience_audit_colab.ipynb` in a CPU Colab runtime.
It reads `MyDrive/datasets/curriculum_cost_adapter/pilot_v1` and writes derived
files under `experience_audit/run_0`. Existing pilot files remain unchanged.

The report includes every evaluation case that changed from success before
adaptation to failure after adaptation when complete evaluation records exist.
Those paths contain observations only at replanning boundaries, with their
actual primitive timestamps. Exact learned-cost ordering is not recomputed:
the original goal embedding was not saved, and the audit does not substitute a
terminal observation for it.

