# Fine-tune I-JEPA on rendered Wall frames

This stage adapts the released full ImageNet-1K I-JEPA ViT-H/14 checkpoint with
the original within-image masked-token objective. It extends `src/train.py`;
the online encoder and predictor receive gradients, and the target encoder
follows by EMA. It does not train the action-conditioned world model.

## Inputs and split

Use the **same full** `episode_manifest.json` and raw `episode_XXXX.pt` files
from the multi-layout Wall experiment. The frame index validates all 2,600
episodes against manifest seed, layout, policy, hash, and 224×224 RGB shape.
Training reads the nine training layouts; checkpoint selection reads only the
two validation layouts. Four test layouts are indexed for integrity but are
not passed to the trainer. The default samples frames 0, 5, ..., 50 from each
51-frame episode and applies the normalization used by `embed_frames`.

The released full checkpoint must contain `encoder`, `predictor`, and
`target_encoder`. The local reference file is
`src/checkpoints/IN1K-vit.h.14-300e.pth.tar`; it contains all three. The
encoder-only file cannot initialize this training run.

## Colab run

Use a CUDA runtime with enough memory for ViT-H/14, its target copy, the
predictor, gradients, and AdamW state. PyTorch, NumPy, and PyYAML are needed.
Get this branch into the Colab checkout, mount Drive, and verify that the raw
episode directory, manifest, and full released checkpoint are present.
Copy [the config](../configs/wall_ijepa_finetune.yaml) and edit its four
Drive paths: `meta.init_checkpoint`, `data.raw_dir`,
`data.episode_manifest`, and `logging.folder`. Then run from the repository
root:

```bash
python main.py --fname /content/wall_ijepa_finetune.yaml --devices cuda:0
```

The sample config uses batch size 1 and 16 gradient-accumulation batches,
20 epochs, an online/predictor learning rate of `1e-5`, and the upstream
spatial mask parameters. It starts a **fresh** optimizer; the ImageNet
pretraining optimizer is never restored. Validation uses fixed masks and
frame order. `data.num_workers` must remain zero so mask order and epoch-boundary
resume are reproducible. If the GPU runs out of memory, reduce image batch
size only if it is above one; on this config, use a higher-memory runtime or
add activation checkpointing in a separately reviewed change.

The output directory contains:

| File | Purpose |
| --- | --- |
| `jepa-latest.pth.tar` | Full resumable state after each epoch |
| `jepa-best.pth.tar` | Full model selected by validation loss |
| `frame_index.json` | Manifest-linked frame sample identities |
| `history.json` | Train/validation loss and optimizer-update count |
| `params-ijepa.yaml` | Exact run configuration |

To resume after an epoch, change only `meta.load_checkpoint` from `false` to
`true` in the same config. Keep all other paths and settings identical. The
trainer rejects mismatched manifest, released checkpoint, frame stride, or
run config identity.

After training, export the selected **online** encoder:

```bash
python -m src.mylonas_ijepa_wm.export_wall_ijepa_encoder \
  --checkpoint /content/drive/MyDrive/datasets/wall_ijepa_finetune_1k/jepa-best.pth.tar \
  --output /content/drive/MyDrive/datasets/wall_ijepa_finetune_1k/jepa-best.encoder-only.pth.tar
```

The output has the `encoder` and `epoch` keys read by the current
`embed_frames.load_checkpoint`; it also retains Wall run identity and
validation loss. Keep the full best checkpoint too, since encoder-only
weights cannot resume masked-prediction training.

## Local validation completed

The ignored 150-episode Wall pilot indexes to 990 training, 220 validation,
and 440 held-out test frame references at stride five. Tiny-model local tests
exercise a parameter update, EMA, best-checkpoint selection, exact
epoch-boundary resume, mismatch rejection, and encoder export through the
existing embedding loader. A full ViT-H training run and downstream
world-model evaluation require the full raw dataset and the Colab GPU runtime.
