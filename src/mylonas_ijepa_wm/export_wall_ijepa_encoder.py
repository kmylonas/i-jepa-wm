"""Export the selected online Wall I-JEPA encoder for existing embedding tools."""

import argparse
from pathlib import Path

import torch


def export_encoder(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True,
    )
    state = checkpoint.get("encoder")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   for key, value in state.items())):
        raise ValueError("Fine-tuned checkpoint has no valid online encoder")
    if ("best_val_loss" not in checkpoint or "val_loss" not in checkpoint
            or checkpoint["val_loss"] != checkpoint["best_val_loss"]):
        raise ValueError("Input is not the selected best validation checkpoint")
    if not isinstance(checkpoint.get("wall_run_identity"), dict):
        raise ValueError("Fine-tuned checkpoint lacks Wall run identity")
    encoder = {key.removeprefix("module."): value for key, value in state.items()}
    if len(encoder) != len(state):
        raise ValueError("Fine-tuned encoder has duplicate keys after prefix removal")
    output = {
        "encoder": encoder,
        "epoch": int(checkpoint["epoch"]),
        "wall_run_identity": checkpoint["wall_run_identity"],
        "val_loss": float(checkpoint["val_loss"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output_path) + ".tmp")
    torch.save(output, temporary)
    temporary.replace(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    export_encoder(args.checkpoint, args.output)
    print(f"Exported online encoder: {args.output}")


if __name__ == "__main__":
    main()
