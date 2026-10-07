from pathlib import Path
import torch

full_checkpoint_path = Path(
    "src/checkpoints/IN1K-vit.h.14-300e.pth.tar"
)

encoder_checkpoint_path = Path(
    "src/checkpoints/IN1K-vit.h.14-300e.encoder-only.pth.tar"
)

print("Opening full checkpoint...")

full_checkpoint = torch.load(
    full_checkpoint_path,
    map_location="cpu",
    weights_only=True,
    mmap=True,
)

print("Checkpoint keys:", full_checkpoint.keys())
print("Checkpoint epoch:", full_checkpoint["epoch"])

encoder_checkpoint = {
    "encoder": full_checkpoint["encoder"],
    "epoch": full_checkpoint["epoch"],
}

print("Saving encoder-only checkpoint...")

torch.save(
    encoder_checkpoint,
    encoder_checkpoint_path,
)

print("Saved:", encoder_checkpoint_path)
print(
    "Size:",
    round(encoder_checkpoint_path.stat().st_size / 2**30, 2),
    "GiB",
)