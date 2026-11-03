# Matched fixed versus multi-layout Wall comparison

## Purpose

Isolate the effect of training-layout diversity on geometry-sensitive world-model prediction and position-cost planning. Train two new proprioceptive I-JEPA world models from scratch. They differ in their training layouts only: the fixed model uses wall/door `[32, 30]`; the multi model uses the existing nine-layout grid. The historical fixed checkpoint is context, not part of the causal comparison.

## Data and controls

- Use the same frozen ImageNet-1K I-JEPA ViT-H/14 encoder checkpoint for both models. Use the pinned DINO-WM DotWall source at commit `0a9492fa12044b852ae9e001cc74604b79c8bb0c`.
- Collect 2,000 training episodes per model, each with 50 primitive actions. Use only the existing `wide` policy: persistent correlated headings and uniformly sampled action lengths in `[0.2, 3.5]`. No independent-random or doorway-directed episodes.
- Use separate fixed and multi datasets, with independent manifests and raw/cache directories. Pair training episode IDs and seeds across datasets. The policy's sampled actions should agree for paired seeds; starts and goals may depend on layout. Save the realized action norms, per-layout wall-contact/door-crossing counts, and compact per-episode action hashes in an audit so the second run can be compared after the first raw dataset is deleted.
- Use the same two validation layouts `[[28, 24], [36, 36]]`, 200 validation episodes, and the same four test layouts `[[26, 22], [26, 38], [38, 22], [38, 38]]`, 400 test episodes, for both models. Corresponding validation/test episodes have identical seeds, layouts, policy, and rendered trajectories. These layouts and all test settings are fixed before training. Select each best checkpoint on the same validation distribution; never select with test results.
- Keep five primitive actions per world-model step, embedding stride 1, one history step, one-step prediction, model architecture, loss weights, optimizer, batch size, maximum epochs, initialization seed, and stopping/checkpoint rule identical. Training-only normalization statistics are computed independently for each model's training distribution.
- Prepare two explicit manifests and record their hashes, code commit, encoder hash, DINO-WM commit, seeds, policy parameters, and configuration in the outputs. Existing wide/door manifests and historical checkpoints must remain valid.

## Colab storage and execution

- Deliver one uploadable `.ipynb` using the existing GitHub branch and Colab path conventions. The notebook mounts Drive, copies the encoder to `/content`, clones pinned source, runs a small smoke test, then runs fixed and multi experiments sequentially.
- Raw episode files, image frames, embedding caches, and trajectory sidecars live only under `/content`. Store only small manifests/configurations, checkpoints, logs, aggregate results, and selected diagnostic figures on Drive, under a new versioned directory separate from earlier experiments.
- The notebook checks available `/content` disk before collection and embedding. Audit all raw episodes, embed in batches, verify each completed batch, then remove that batch's raw files. After training and cache-dependent evaluation, verify its checkpoint and results on Drive, then remove that model's entire runtime dataset and cache before starting the other model. Never remove a file before its replacement/output has been verified.
- A Colab runtime reset loses raw episodes and caches. The saved manifest, code commit, encoder hash, and checkpoint allow exact regeneration locally before resuming training. State this cost clearly in the notebook.

## Evaluation

- Primary held-out planning: position-cost CEM, identical planner settings, the same 50 start-goal seeds on each of the four test layouts for both models, and primitive-action Euclidean norm capped at 3.5. The `0.2` lower bound applies to collected training actions; the planner may choose shorter actions, including zero. Use the same layout config, encoder, CEM random seed, and success criterion for each matched pair. Report counts and paired differences per layout and overall. For the overall confidence interval, resample the 50 seed IDs with replacement, carrying all four layouts for each sampled seed, and compute the mean paired success difference across 10,000 replicates with a fixed bootstrap seed. This treats reused seeds across layouts as clustered rather than 200 independent seed draws.
- Training-layout sanity check: both models plan on `[32, 30]` with 20 matched seeds and the same 3.5 cap.
- Direct prediction: one-step and five-step rollout state/token errors by layout, plus matched counterfactual wall/door visual pairs and collision probes using the same case definitions for both checkpoints. Report contact/crossing coverage and probe sample sizes; selected probes are diagnostics, not population accuracy estimates.
- Run the existing test episode split only after model and planner settings are fixed. Keep a separate record of validation diagnostics and final test results.

## Acceptance and limits

- Both runs must use the specified policy and action range, have equal training episode counts, pass split/identity checks, and produce checkpoints and evaluation files with matching settings.
- Report wall-contact and doorway-crossing coverage from the pilot and full datasets. The old mixed-policy doorway threshold is diagnostic for this correlated-only experiment; it does not stop training. Do not silently introduce a different policy or change the predeclared 2,000-episode budget. Treat sparse doorway data as a limitation of the matched comparison.
- The experiment isolates layout diversity under this collector and simulator. It does not establish vision-only planning because the world model receives true ball position at each replanning update.
