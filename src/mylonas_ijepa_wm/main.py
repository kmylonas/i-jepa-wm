import numpy as np
from pathlib import Path
import torch
import src.models.vision_transformer as vit
import argparse




patch_size=14
crop_size=224
model_name="vit_huge"
frame_skip = 5


DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "checkpoints"
    / "IN1K-vit.h.14-300e.pth.tar"
)


IJEPA_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

IJEPA_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)


def preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    # [B, H, W, C] -> [B, C, H, W]
    images = frames.permute(0, 3, 1, 2).contiguous()

    # uint8 [0, 255] -> float32 [0, 1]
    images = images.to(dtype=torch.float32).div(255.0)

    # Apply ImageNet normalization.
    images = (images - IJEPA_MEAN) / IJEPA_STD

    return images




def load_checkpoint(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    pretrained_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["encoder"].items()
    }

    epoch = checkpoint['epoch']
    
    msg = encoder.load_state_dict(pretrained_dict)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    return encoder




def process_episode(
    episode,
    encoder,
    device,
    episode_name,
    output_directory,
):
    selected_frames = episode["frames"][::frame_skip]
    images = preprocess_frames(selected_frames)
    images = images.to(device)
    with torch.inference_mode():
        tokens = encoder(images) #(B, 256, 1280)

    tokens = tokens.to(
                device="cpu",
                dtype=torch.float16,
            )
    print("Final tokens:", tokens.shape, tokens.dtype)

    ### trajectory
    frame_indices = torch.arange(
        0,
        len(episode["frames"]),
        frame_skip,
        dtype=torch.int64,
    )


    actions = episode["actions"].to(
    device="cpu",
    dtype=torch.float32,
    )

    states = episode["states"].to(
        device="cpu",
        dtype=torch.float32,
    )

    # Five consecutive 2D actions connect two cached observations.
    #
    # [100, 2] -> [20, 5, 2] -> [20, 10]
    macro_actions = actions.reshape(20, frame_skip, 2).flatten(1)

    # States corresponding to frames 0, 5, 10, ..., 100.
    sampled_states = states[frame_indices]

    trajectory = {
        # Full original trajectory information
        "actions": actions,                    # [100, 2]
        "states": states,                      # [101, 4]

        # Information aligned with cached embeddings
        "macro_actions": macro_actions,        # [20, 10]
        "sampled_states": sampled_states,      # [21, 4]
        "frame_indices": frame_indices,        # [21]

        # Episode metadata
        "desired_goal": episode["desired_goal"].cpu(),
    }

    
    ### saving
    trajectory_directory = output_directory / "trajectories"
    trajectory_directory.mkdir(parents=True, exist_ok=True)

    latent_directory = output_directory / "latents"
    latent_directory.mkdir(parents=True, exist_ok=True)
    
    trajectory_path = (
            trajectory_directory / f"{episode_name}.pt"
        )
    torch.save(trajectory, trajectory_path)


    output_path = latent_directory / f"{episode_name}.npy"


    # tokens: torch.float16 CPU tensor [21, 256, 1280]
    np.save(
        output_path,
        tokens.numpy(),
        allow_pickle=False,
    )

    return



def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw episode_XXXX.pt files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for latents and trajectory sidecars.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the I-JEPA checkpoint.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First episode index to process (inclusive).",
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=2000,
        help="Last episode index to process (exclusive).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes whose outputs already exist.",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()

    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.end_index <= args.start_index:
        raise ValueError("end-index must be greater than start-index")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")


    # Instantiate only the encoder—not the original I-JEPA predictor.
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
    )

    encoder = load_checkpoint(
        encoder=encoder,
        checkpoint_path=args.checkpoint_path,
    )
    encoder = encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    latent_directory = args.output_dir / "latents"
    trajectory_directory = args.output_dir / "trajectories"
    latent_directory.mkdir(parents=True, exist_ok=True)
    trajectory_directory.mkdir(parents=True, exist_ok=True)


    for i in range(args.start_index, args.end_index):

        episode_name = f"episode_{i:04d}"

        episode_path = (
            args.input_dir
            / f"{episode_name}.pt"
        )

        latent_path = latent_directory / f"{episode_name}.npy"
        trajectory_path = trajectory_directory / f"{episode_name}.pt"

        if args.resume and latent_path.exists() and trajectory_path.exists():
            print(f"Skipping completed {episode_name}")
            continue

        if not episode_path.is_file():
            raise FileNotFoundError(f"Episode not found: {episode_path}")

        print(
            f"Processing {episode_name} "
            f"({i - args.start_index + 1}/{args.end_index - args.start_index})"
        )

        episode = torch.load(
            episode_path,
            map_location="cpu",
            weights_only=True,
        )

        process_episode(
            episode=episode,
            encoder=encoder,
            device=device,
            episode_name=episode_name,
            output_directory=args.output_dir,
        )
    

if __name__ == "__main__":
    main()

