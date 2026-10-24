import numpy as np
from pathlib import Path
import torch
import src.models.vision_transformer as vit
import argparse
from src.mylonas_ijepa_wm.environment_configs import load_environment_config




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
    environment=None,
    actions_per_model_step=None,
    encoder_batch_size=None,
    embedding_frame_stride=None,
):
    episode_env = episode.get("environment", "pointmaze")
    if environment is not None and episode_env != environment:
        raise ValueError(f"Raw episode environment {episode_env} does not match {environment}")
    recorded_stride = episode.get("actions_per_model_step")
    if (actions_per_model_step is not None and recorded_stride is not None
            and int(recorded_stride) != int(actions_per_model_step)):
        raise ValueError("Embedding action grouping differs from the raw episode")
    stride = int(actions_per_model_step or episode.get("actions_per_model_step", frame_skip))
    embedding_stride = stride if embedding_frame_stride is None else int(embedding_frame_stride)
    if embedding_stride < 1 or stride % embedding_stride:
        raise ValueError("embedding_frame_stride must be a positive divisor of actions_per_model_step")
    actions = episode["actions"].to(device="cpu", dtype=torch.float32)
    states = episode["states"].to(device="cpu", dtype=torch.float32)
    if stride < 1 or actions.ndim != 2 or len(actions) % stride:
        raise ValueError("actions must be [steps, action_dim] and divisible by model stride")
    if len(episode["frames"]) != len(actions) + 1 or len(states) != len(episode["frames"]):
        raise ValueError("raw frames, actions, and states are not aligned")
    selected_frames = episode["frames"][::embedding_stride]
    images = preprocess_frames(selected_frames)
    batch_size = len(images) if encoder_batch_size is None else int(encoder_batch_size)
    if batch_size < 1:
        raise ValueError("encoder_batch_size must be positive")
    with torch.inference_mode():
        tokens = torch.cat([
            encoder(images[start:start + batch_size].to(device)).to(
                device="cpu", dtype=torch.float16
            )
            for start in range(0, len(images), batch_size)
        ], dim=0)
    print("Final tokens:", tokens.shape, tokens.dtype)

    ### trajectory
    frame_indices = torch.arange(
        0,
        len(episode["frames"]),
        embedding_stride,
        dtype=torch.int64,
    )


    macro_actions = actions.reshape(-1, stride, actions.shape[-1]).flatten(1)

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
        "environment": episode_env,
        "environment_config": episode.get("environment_config", {}),
        "wall_layout": episode.get("wall_layout"),
        "collection_policy": episode.get("collection_policy"),
        "collection_policy_parameters": episode.get("collection_policy_parameters"),
        "state_fields": tuple(episode.get("state_fields", ("x", "y", "vx", "vy"))),
        "actions_per_model_step": stride,
        "embedding_frame_stride": embedding_stride,
        "physics_frame_skip": int(episode.get("physics_frame_skip", 1)),
        "control_dt_seconds": episode.get(
            "control_dt_seconds",
            0.01 * int(episode.get("physics_frame_skip", 1)),
        ),
    }

    
    ### saving
    trajectory_directory = output_directory / "trajectories"
    trajectory_directory.mkdir(parents=True, exist_ok=True)

    latent_directory = output_directory / "latents"
    latent_directory.mkdir(parents=True, exist_ok=True)
    
    trajectory_path = (
            trajectory_directory / f"{episode_name}.pt"
        )
    output_path = latent_directory / f"{episode_name}.npy"
    latent_temporary = output_path.with_suffix(".npy.tmp")
    trajectory_temporary = trajectory_path.with_suffix(".pt.tmp")
    # Commit both files only after their writes finish, so --resume cannot
    # treat an interrupted embedding as a completed episode.
    with latent_temporary.open("wb") as file:
        np.save(file, tokens.numpy(), allow_pickle=False)
    torch.save(trajectory, trajectory_temporary)
    latent_temporary.replace(output_path)
    trajectory_temporary.replace(trajectory_path)

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
    parser.add_argument("--env", choices=["pointmaze", "wall"], default="pointmaze")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument(
        "--encoder-batch-size", type=int, default=4,
        help="Frames per encoder forward pass; reduce this if the GPU runs out of memory.",
    )
    parser.add_argument(
        "--embedding-frame-stride", type=int,
        help="Cache every Nth frame. Use 1 for DINO-style windows starting at each action; "
             "the default caches one frame per model step.",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()
    config = load_environment_config(args.env, args.env_config)

    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.end_index <= args.start_index:
        raise ValueError("end-index must be greater than start-index")
    if args.embedding_frame_stride is not None and (
        args.embedding_frame_stride < 1
        or config.actions_per_model_step % args.embedding_frame_stride
    ):
        raise ValueError("embedding-frame-stride must be a positive divisor of actions-per-model-step")
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
            saved = torch.load(trajectory_path, map_location="cpu", weights_only=True)
            expected_stride = args.embedding_frame_stride or config.actions_per_model_step
            if int(saved.get("embedding_frame_stride", saved.get("actions_per_model_step", frame_skip))) != expected_stride:
                raise ValueError(f"{episode_name} was embedded with a different frame stride")
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
        if episode.get("environment_config") and episode["environment_config"] != config.portable_metadata():
            raise ValueError(f"{episode_name} environment settings differ from --env-config")

        process_episode(
            episode=episode,
            encoder=encoder,
            device=device,
            episode_name=episode_name,
            output_directory=args.output_dir,
            environment=config.env,
            actions_per_model_step=config.actions_per_model_step,
            encoder_batch_size=args.encoder_batch_size,
            embedding_frame_stride=args.embedding_frame_stride,
        )
    

if __name__ == "__main__":
    main()

