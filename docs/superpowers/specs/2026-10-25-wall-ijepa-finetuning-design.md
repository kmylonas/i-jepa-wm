# Wall I-JEPA fine-tuning experiment

## Purpose

Test whether adapting the released I-JEPA ViT-H/14 representation to rendered
DotWall images improves the visual signal used by the existing action-conditioned
world model. The strongest earlier multi-layout system received true ball
position and planned much more reliably with position cost than latent cost.
The new experiment must isolate the effect of encoder fine-tuning from changes
to the Wall dataset, world model, planner, or evaluation seeds.

## Research comparison

- Primary comparison: the existing frozen ImageNet-1K encoder and a Wall-adapted
  encoder, both embedded on the **same** multi-layout raw episodes. Train a new
  proprioceptive world model from random initialization on the adapted tokens,
  using the same architecture, train/validation/test episode IDs, training
  settings, and planning settings as the frozen-encoder world model.
- Keep the existing manifest's nine train layouts, two validation layouts, and
  four test layouts. Wall and doorway coordinates remain disjoint across these
  splits. The test layouts must never supply fine-tuning images, select a
  fine-tuning checkpoint, or select downstream hyperparameters.
- Evaluate both real-frame latent-goal ordering and world-model behavior.
  Report per-layout ordering/rank inversions on identical rendered frames and
  goals; predicted-versus-real candidate ranking; token and state rollouts;
  collision/visual-input probes; and position- and latent-cost planning with
  matched held-out start-goal seeds. Compare rankings within each encoder's
  own token space, not raw MSE magnitudes across encoders.
- Treat the historical fixed-layout checkpoint as context only. The controlled
  contrast is the existing multi-layout frozen-encoder model versus the new
  multi-layout fine-tuned-encoder model. The canonical frozen checkpoint is
  `src/mylonas_ijepa_wm/wall/wall_multi/world_model/best.pt` in the main
  checkout. Its saved manifest hash matches the locally saved
  `src/mylonas_ijepa_wm/wall/wall_multi/extreme_x/episode_manifest.json`.
  It records best epoch 15, manifest IDs, model architecture, normalization,
  and one-step training objective. The runbook records the training command.
  Rerun the frozen checkpoint's evaluation cases with the same current
  evaluation code and seeds used for the adapted model. If the missing raw
  dataset or any unrecorded baseline setting prevents a matched comparison,
  train a new frozen-encoder control from the same manifest/cache and report
  it as the primary control; retain the saved checkpoint as historical context.

## Data

Use the `wall-layout-generalization` implementation and its versioned
`episode_manifest.json`. The fine-tuning dataset indexes frames from raw
`episode_XXXX.pt` files by manifest episode ID and frame index; it need not
duplicate rendered images on disk. The index stores the source manifest hash,
split, layout, episode ID, and frame count. It checks each episode's seed,
layout, environment, and manifest identity before training. It can be rebuilt
without changing sample identity. Do not mix frames from pilot and full
manifests.

Default to the already collected full multi-layout episodes when available.
If the raw files are not retained, recollect from the *same* full manifest and
the pinned DotWall checkout, then run the existing per-layout coverage audit.
Do not change the policy or layout assignment during recollection. Use training
layout frames for optimization, validation layout frames for checkpoint
selection, and hold test layout frames out. No action or position labels enter
the I-JEPA fine-tuning loss.

Use the complete 224×224 rendered frame with the same ImageNet channel
normalization as the existing embedding pipeline. Disable horizontal flips
and geometry-changing crops: both move or obscure the wall and doorway. Use
the original I-JEPA spatial context/target masking on the resulting full-frame
patch grid. The initial run samples every fifth frame per episode to limit
near duplicates and compute; the frame stride is a recorded configuration,
not an undocumented data change. A later all-frame sensitivity run is optional
and must have a separate identity.

## Fine-tuning objective and checkpoint

Use I-JEPA's original within-image masked patch-token
prediction objective. Load the released full ImageNet-1K ViT-H/14 checkpoint:
online encoder, predictor, and EMA target encoder. Check architecture and all
state-dict keys strictly before any optimization. Initialize a new optimizer
and schedule for this Wall run rather than restoring the pretraining optimizer
or continuing its epoch count. Train the online encoder and predictor; update
the target encoder by EMA. Select a checkpoint using validation masked-token
loss, with fixed validation masks and deterministic frame order. Save complete
model, optimizer, schedule, epoch, RNG, manifest, sample-index, and source
checkpoint identities to support exact resume. Also save a clearly named
encoder-only export from the selected checkpoint. Export the **online
encoder** for the primary run, matching the existing frozen baseline's
encoder-only extraction. An EMA encoder export can be a separately labeled
ablation; mixing it into the primary comparison would change two factors.

Start with conservative adaptation hyperparameters and a pilot on the local
Wall sample. Full ViT-H fine-tuning needs a CUDA GPU; the local Mac smoke test
checks data, checkpoint loading, a forward/backward update, export, and
embedding compatibility without claiming a complete training result. Record
batch size, effective batch size, learning rate, weight decay, EMA range,
mask parameters, precision, epochs, and selection rule in a run config.
Because full model checkpoints are large, retain a resumable latest checkpoint
and a best validation checkpoint rather than every epoch.

## Downstream workflow

1. Verify the exported encoder loads through the current `embed_frames` and
   online planning loaders and produces `(256, 1280)` tokens per frame.
2. Re-embed the same raw multi-layout episodes into a **new** dense cache with
   `--embedding-frame-stride 1`. Keep the frozen cache separate. Save the
   exported encoder hash in cache metadata so resume rejects a different
   encoder, even when episode and stride metadata match.
3. Train a new proprioceptive world model from random weights on the new
   cache. Use the existing manifest split and train-only state/action
   normalization. Record the encoder hash and fine-tuning run identity in its
   checkpoint. Do not initialize it from the frozen-encoder world model.
4. Evaluate with the same test layouts, seeds, planner action limits, CEM
   settings, and diagnostic cases as the frozen baseline. Save the exact model
   and encoder hashes with all results.

## Interfaces and safeguards

- Add a Wall frame dataset/index and a standalone I-JEPA fine-tuning command
  beside the existing I-JEPA trainer; preserve the upstream ImageNet training
  entry point. Add an encoder-export command or function with strict load
  validation. Extend embedding cache provenance only as needed to prevent
  stale token reuse.
- The command accepts paths to the full checkpoint, raw episodes, manifest,
  output directory, and an explicit run config. Paths remain external to Git.
- Fail before training on missing episodes, mismatched manifest/episode
  metadata, split leakage, incomplete checkpoint components, incompatible
  architecture, zero selected frames, or existing output with another run
  identity. Resume restores training state and rejects changed identities.
- Existing `embed_frames`, `wm_proprio`, and planning formats remain usable.
  Large raw data, caches, and full checkpoints stay outside Git.

## Decisions and remaining builder choices

The user selected the original masked-image I-JEPA objective, the full
ImageNet-1K ViT-H/14 checkpoint, and reuse of the existing multi-layout
episodes. The local checkpoint at `src/checkpoints/IN1K-vit.h.14-300e.pth.tar`
contains online encoder, predictor, and EMA target-encoder state dictionaries.
The full raw episodes are not in this worktree and must be located in the
existing experiment storage before a full run. Concrete tuning values can be
selected during the implementation plan after measuring one local pilot step
and the available CUDA memory.
