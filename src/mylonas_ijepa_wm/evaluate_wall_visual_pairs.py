"""Compare DotWall predictions at identical states/actions in two visual layouts."""

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


def build_matched_pairs() -> list[dict]:
    """Predeclared contrasting cases; their geometry does not depend on model output."""
    return [
        dict(name="door_y22_left_to_right", layouts=[[26, 22], [26, 38]],
             start=[21.0, 22.0], actions=[[1.0, 0.0]] * 5,
             reference_x=26, expected_crossed=[True, False]),
        dict(name="door_y38_right_to_left", layouts=[[26, 22], [26, 38]],
             start=[31.0, 38.0], actions=[[-1.0, 0.0]] * 5,
             reference_x=26, expected_crossed=[False, True]),
        dict(name="wall_x26_right_to_left", layouts=[[26, 22], [38, 22]],
             start=[31.0, 32.0], actions=[[-1.0, 0.0]] * 5,
             reference_x=26, expected_crossed=[False, True]),
        dict(name="wall_x38_left_to_right", layouts=[[26, 22], [38, 22]],
             start=[33.0, 32.0], actions=[[1.0, 0.0]] * 5,
             reference_x=38, expected_crossed=[True, False]),
    ]


def _crossed(start_x: float, end_x: float, reference_x: float) -> bool:
    return bool((start_x < reference_x < end_x) or (end_x < reference_x < start_x))


def _evaluate_case(adapter, encoder, model, metadata, pair, layout, seed, device, precision):
    adapter.reset(seed, layout=layout)
    adapter.env.set_init_state(np.asarray(pair["start"], dtype=np.float32))
    adapter.env.reset()
    start = adapter.state()
    if not np.allclose(start, pair["start"], atol=1e-5):
        raise ValueError(f"Simulator did not use requested start in {pair['name']}")
    initial_frame = adapter.render()
    latent = visual.encode_frame(encoder, initial_frame, device, precision)
    state_mean = metadata["state_mean"].to(device=device, dtype=torch.float32)
    state_std = metadata["state_std"].to(device=device, dtype=torch.float32)
    action = torch.as_tensor(
        np.asarray(pair["actions"], dtype=np.float32).reshape(1, 1, -1),
        device=device,
    )
    if metadata.get("normalize_actions", False):
        action = (action - metadata["action_mean"].to(device)) / metadata["action_std"].to(device)
    state = torch.as_tensor(start, device=device, dtype=torch.float32)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            predicted_tokens, predicted_states = model(
                latent[None, None], ((state - state_mean) / state_std)[None, None], action,
            )
    predicted = (predicted_states[0, -1].float() * state_std + state_mean).cpu().numpy()

    positions = [start.tolist()]
    final_frame = initial_frame
    for primitive in pair["actions"]:
        outcome = adapter.step(primitive)
        positions.append(np.asarray(outcome.state).tolist())
        final_frame = outcome.frame
    actual = np.asarray(positions[-1], dtype=np.float32)
    actual_tokens = visual.encode_frame(encoder, final_frame, device, precision)
    token_mse = torch.mean((predicted_tokens[0, -1].float() - actual_tokens.float()).square())
    reference_x = pair["reference_x"]
    return dict(
        layout=list(layout), start=start.tolist(), actions=pair["actions"],
        actual_positions=positions, actual_final_position=actual.tolist(),
        predicted_final_position=predicted.tolist(),
        actual_crossed=_crossed(start[0], actual[0], reference_x),
        predicted_crossed=_crossed(start[0], predicted[0], reference_x),
        position_error=float(np.linalg.norm(predicted - actual)),
        token_mse=float(token_mse.item()),
    )


def evaluate_pair(adapter, encoder, model, metadata, pair, seed, device, precision) -> dict:
    cases = [
        _evaluate_case(adapter, encoder, model, metadata, pair, layout, seed, device, precision)
        for layout in pair["layouts"]
    ]
    if any(case["start"] != pair["start"] or case["actions"] != pair["actions"] for case in cases):
        raise ValueError(f"Matched inputs changed in {pair['name']}")
    actual = [case["actual_crossed"] for case in cases]
    if actual != pair["expected_crossed"] or actual[0] == actual[1]:
        raise ValueError(f"Simulator outcomes do not match predeclared contrast in {pair['name']}: {actual}")
    predicted = [case["predicted_crossed"] for case in cases]
    return dict(
        name=pair["name"], seed=seed, reference_x=pair["reference_x"],
        start=pair["start"], actions=pair["actions"],
        expected_crossed=pair["expected_crossed"], cases=cases,
        predicted_contrast=predicted[0] != predicted[1],
        both_correct=predicted == actual,
    )


def evaluate_pairs(
    checkpoint: Path, encoder_checkpoint: Path, manifest: dict,
    output: Path, env_config: Path, device_name: str = "auto",
    precision_name: str = "auto", seed: int = 10000,
) -> dict:
    config = load_environment_config("wall", env_config)
    device = wm.resolve_device(device_name)
    precision = wm.resolve_precision(precision_name, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(checkpoint, device)
    validate_checkpoint_manifest(metadata, manifest)
    if metadata.get("environment") != "wall" or model.num_hist != 1:
        raise ValueError("Visual pairs require a one-history Wall checkpoint")
    pairs = build_matched_pairs()
    test_layouts = {tuple(layout) for layout in manifest["test_layouts"]}
    if any(tuple(layout) not in test_layouts for pair in pairs for layout in pair["layouts"]):
        raise ValueError("A visual pair layout is outside the held-out test split")
    if config.actions_per_model_step != 5:
        raise ValueError("Visual pairs require five actions per model step")
    encoder = visual.load_frozen_ijepa_encoder(encoder_checkpoint, device)
    adapter = create_adapter(config, max_micro_steps=5)
    try:
        results = [
            evaluate_pair(adapter, encoder, model, metadata, pair, seed, device, precision)
            for pair in pairs
        ]
    finally:
        adapter.close()
    cases = [case for result in results for case in result["cases"]]
    payload = dict(
        checkpoint=str(checkpoint), checkpoint_epoch=epoch,
        encoder_checkpoint=str(encoder_checkpoint), seed=seed,
        summary=dict(
            pairs=len(results), cases=len(cases),
            predicted_contrasts=sum(result["predicted_contrast"] for result in results),
            both_correct_pairs=sum(result["both_correct"] for result in results),
            crossing_matches=sum(case["predicted_crossed"] == case["actual_crossed"] for case in cases),
            mean_position_error=float(np.mean([case["position_error"] for case in cases])),
        ),
        pairs=results,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    args = parser.parse_args(argv)
    result = evaluate_pairs(
        args.checkpoint, args.encoder_checkpoint, load_manifest(args.episode_manifest),
        args.output, args.env_config, args.device, args.precision, args.seed,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved visual pairs to {args.output}")


if __name__ == "__main__":
    main()
