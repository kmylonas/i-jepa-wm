"""Probe held-out DotWall blocking, doorway passage, and old-wall free space."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm import wm, wm_proprio
from src.mylonas_ijepa_wm.environment_adapters import create_adapter
from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.wall_layout_manifest import load_manifest, validate_checkpoint_manifest


def build_probes(layout: tuple[int, int]) -> list[dict]:
    wall_x, door_y = layout
    old_wall_x = 40 if wall_x < 32 else 24
    probes = []
    for kind, reference_x, y in (
        ("solid_wall", wall_x, door_y + 10),
        ("doorway", wall_x, door_y),
        ("old_wall_open", old_wall_x, door_y + 10),
    ):
        for direction, side in (("left_to_right", -1), ("right_to_left", 1)):
            start = [float(reference_x + side * 5), float(y)]
            action = [float(-side), 0.0]
            probes.append(dict(
                kind=kind, direction=direction, reference_x=int(reference_x),
                start=start, actions=[action.copy() for _ in range(5)],
            ))
    return probes


def _crossed(start_x, end_x, reference_x):
    return (start_x < reference_x < end_x) or (end_x < reference_x < start_x)


def evaluate_single_probe(
    adapter, encoder, model, metadata, probe, layout, seed, device, precision,
):
    adapter.reset(seed, layout=layout)
    adapter.env.set_init_state(np.asarray(probe["start"], dtype=np.float32))
    adapter.env.reset()
    initial_state = np.asarray(adapter.state(), dtype=np.float32)
    initial_frame = adapter.render()
    latent = visual.encode_frame(encoder, initial_frame, device, precision)
    state_mean = metadata["state_mean"].to(device=device, dtype=torch.float32)
    state_std = metadata["state_std"].to(device=device, dtype=torch.float32)
    state = torch.as_tensor(initial_state, device=device, dtype=torch.float32)
    action = torch.as_tensor(
        np.asarray(probe["actions"], dtype=np.float32).reshape(1, 1, -1),
        device=device,
    )
    if metadata.get("normalize_actions", False):
        action = (
            action - metadata["action_mean"].to(device)
        ) / metadata["action_std"].to(device)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            predicted_tokens, predicted_states = model(
                latent[None, None],
                ((state - state_mean) / state_std)[None, None],
                action,
            )
    predicted_position = (
        predicted_states[0, -1].float() * state_std + state_mean
    ).cpu().numpy()
    actual_state = initial_state
    final_frame = initial_frame
    for primitive in probe["actions"]:
        outcome = adapter.step(primitive)
        actual_state = outcome.state
        final_frame = outcome.frame
    actual_tokens = visual.encode_frame(encoder, final_frame, device, precision)
    token_mse = torch.mean(
        (predicted_tokens[0, -1].float() - actual_tokens.float()).square()
    ).item()
    reference_x = probe["reference_x"]
    direction = 1.0 if probe["direction"] == "left_to_right" else -1.0
    predicted_progress = direction * float(predicted_position[0] - initial_state[0])
    actual_progress = direction * float(actual_state[0] - initial_state[0])
    return dict(
        seed=int(seed), wall_layout=list(layout), kind=probe["kind"],
        direction=probe["direction"], reference_x=int(reference_x),
        start=initial_state.tolist(), actions=probe["actions"],
        predicted_final_position=predicted_position.tolist(),
        actual_final_position=np.asarray(actual_state).tolist(),
        predicted_crossed=bool(_crossed(initial_state[0], predicted_position[0], reference_x)),
        actual_crossed=bool(_crossed(initial_state[0], actual_state[0], reference_x)),
        expected_crossed=probe["kind"] != "solid_wall",
        predicted_progress=predicted_progress,
        actual_progress=actual_progress,
        progress_error=predicted_progress - actual_progress,
        position_error=float(np.linalg.norm(predicted_position - actual_state)),
        token_mse=float(token_mse),
    )


def evaluate_probes(
    checkpoint: Path, encoder_checkpoint: Path, manifest: dict,
    seeds: list[int], output: Path, env_config: Path,
    device_name: str = "auto", precision_name: str = "auto",
) -> dict:
    config = load_environment_config("wall", env_config)
    device = wm.resolve_device(device_name)
    precision = wm.resolve_precision(precision_name, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(checkpoint, device)
    validate_checkpoint_manifest(metadata, manifest)
    if metadata.get("environment") != "wall" or model.num_hist != 1:
        raise ValueError("Collision probes require a one-history Wall checkpoint")
    encoder = visual.load_frozen_ijepa_encoder(encoder_checkpoint, device)
    adapter = create_adapter(config, max_micro_steps=5)
    records = []
    layouts = manifest["test_layouts"]
    try:
        for layout in layouts:
            for seed in seeds:
                for probe in build_probes(tuple(layout)):
                    records.append(evaluate_single_probe(
                        adapter, encoder, model, metadata, probe, tuple(layout),
                        seed, device, precision,
                    ))
    finally:
        adapter.close()
    summary = {}
    for layout in layouts:
        key = f"{layout[0]},{layout[1]}"
        subset = [record for record in records if record["wall_layout"] == list(layout)]
        summary[key] = dict(
            cases=len(subset),
            crossing_accuracy=sum(
                record["predicted_crossed"] == record["actual_crossed"]
                for record in subset
            ) / len(subset),
            mean_position_error=float(np.mean([r["position_error"] for r in subset])),
            mean_token_mse=float(np.mean([r["token_mse"] for r in subset])),
        )
    payload = dict(
        checkpoint=str(checkpoint), checkpoint_epoch=epoch,
        encoder_checkpoint=str(encoder_checkpoint),
        environment=config.portable_metadata(), seeds=seeds,
        summary_by_layout=summary, probes=records,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    args = parser.parse_args(argv)
    if args.num_seeds < 1:
        raise ValueError("num-seeds must be positive")
    manifest = load_manifest(args.episode_manifest)
    evaluate_probes(
        args.checkpoint, args.encoder_checkpoint, manifest,
        list(range(args.base_seed, args.base_seed + args.num_seeds)),
        args.output, args.env_config, args.device, args.precision,
    )
    print(f"Saved collision probes to {args.output}")


if __name__ == "__main__":
    main()
