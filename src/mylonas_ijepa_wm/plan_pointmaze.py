"""Plan in PointMaze with frozen I-JEPA latent world models."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import src.models.vision_transformer as vit

from src.mylonas_ijepa_wm.embed_frames import preprocess_frames
from src.mylonas_ijepa_wm.evaluate_rollouts import load_model_and_metadata
from src.mylonas_ijepa_wm.planning import (
    cem_optimize,
    compute_terminal_cost,
    score_action_sequences,
)
from src.mylonas_ijepa_wm.position_probe import load_probe_checkpoint
from src.mylonas_ijepa_wm.wm import (
    compile_model,
    resolve_device,
    resolve_precision,
)


IMAGE_SIZE = 224
PATCH_SIZE = 14
MODEL_NAME = "vit_huge"
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ENCODER_CHECKPOINT = (
    MODULE_DIR.parent
    / "checkpoints"
    / "IN1K-vit.h.14-300e.encoder-only.pth.tar"
)
DEFAULT_WORLD_MODEL_CHECKPOINT = (
    MODULE_DIR
    / "world_models"
    / "full_deafult_v1_k3"
    / "best.pt"
)
DEFAULT_POSITION_PROBE_CHECKPOINT = (
    MODULE_DIR
    / "position_probe_runs"
    / "spatial_softmax_mlp"
    / "best.pt"
)
DEFAULT_OUTPUT_PATH = (
    MODULE_DIR / "planning_runs" / "pointmaze_probe.json"
)


@dataclass(frozen=True)
class MacroStepResult:
    observation: dict
    info: dict
    terminated: bool
    truncated: bool
    success: bool
    micro_steps: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plan in PointMaze with an I-JEPA latent world model.",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        default=DEFAULT_ENCODER_CHECKPOINT,
    )
    parser.add_argument(
        "--world-model-checkpoint",
        type=Path,
        default=DEFAULT_WORLD_MODEL_CHECKPOINT,
    )
    parser.add_argument(
        "--position-probe-checkpoint",
        type=Path,
        default=DEFAULT_POSITION_PROBE_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--cost",
        choices=["probe", "latent", "oracle"],
        default="probe",
    )
    parser.add_argument(
        "--environment-id",
        default="PointMaze_UMaze-v3",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-macro-steps", type=int, default=20)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--num-elites", type=int, default=32)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def validate_arguments(args):
    if not args.encoder_checkpoint.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {args.encoder_checkpoint}"
        )
    if not args.world_model_checkpoint.is_file():
        raise FileNotFoundError(
            "World-model checkpoint not found: "
            f"{args.world_model_checkpoint}"
        )
    if (
        args.cost in {"probe", "oracle"}
        and not args.position_probe_checkpoint.is_file()
    ):
        raise FileNotFoundError(
            "Position probe checkpoint not found: "
            f"{args.position_probe_checkpoint}"
        )
    for name in [
        "num_episodes",
        "max_macro_steps",
        "planning_horizon",
        "population",
        "num_elites",
        "cem_iterations",
        "candidate_batch_size",
    ]:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be at least 1")
    if args.num_elites > args.population:
        raise ValueError("num-elites cannot exceed population")
    if args.min_std <= 0:
        raise ValueError("min-std must be positive")


def summarize_results(records):
    if not records:
        raise ValueError("at least one episode record is required")
    final_distances = np.asarray(
        [record["final_goal_distance"] for record in records],
        dtype=np.float64,
    )
    macro_steps = np.asarray(
        [record["macro_steps"] for record in records],
        dtype=np.float64,
    )
    return {
        "num_episodes": len(records),
        "num_successes": sum(bool(record["success"]) for record in records),
        "success_rate": float(
            np.mean([bool(record["success"]) for record in records])
        ),
        "mean_final_goal_distance": float(final_distances.mean()),
        "median_final_goal_distance": float(np.median(final_distances)),
        "mean_macro_steps": float(macro_steps.mean()),
    }


def load_frozen_ijepa_encoder(
    checkpoint_path,
    device,
    encoder_factory=None,
):
    if encoder_factory is None:
        encoder_factory = vit.__dict__[MODEL_NAME]
    encoder = encoder_factory(
        img_size=[IMAGE_SIZE],
        patch_size=PATCH_SIZE,
    )
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    encoder_state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["encoder"].items()
    }
    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def encode_frame(
    encoder,
    frame,
    device,
    precision="fp32",
):
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    frame = validate_frame(frame)
    frames = torch.from_numpy(frame).unsqueeze(0)
    images = preprocess_frames(frames).to(device=device)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            latent = encoder(images)

    return latent[0].float()


def validate_frame(frame):
    frame = np.asarray(frame)
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    if frame.shape != expected_shape:
        raise RuntimeError(
            f"Expected frame shape {expected_shape}, got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def render_target_observation(env):
    point_maze = env.unwrapped
    point_environment = point_maze.point_env
    original_qpos = point_environment.data.qpos.copy()
    original_qvel = point_environment.data.qvel.copy()
    target_qpos = original_qpos.copy()
    target_qpos[:2] = point_maze.goal
    target_qvel = np.zeros_like(original_qvel)

    try:
        point_environment.set_state(target_qpos, target_qvel)
        return validate_frame(env.render())
    finally:
        point_environment.set_state(original_qpos, original_qvel)


def execute_macro_action(env, macro_action):
    macro_action = torch.as_tensor(macro_action).detach().cpu()
    if macro_action.shape != (10,):
        raise ValueError("macro action must have shape [10]")

    micro_actions = macro_action.reshape(5, 2).numpy().astype(
        np.float32,
        copy=False,
    )
    result = None

    for micro_step, action in enumerate(micro_actions, start=1):
        observation, _, terminated, truncated, info = env.step(action)
        success = bool(info.get("success", False))
        result = MacroStepResult(
            observation=observation,
            info=info,
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=success,
            micro_steps=micro_step,
        )
        if success or terminated or truncated:
            break

    return result


def initialize_mpc_history(initial_latent, num_hist, action_dim):
    if initial_latent.ndim != 2:
        raise ValueError(
            "initial_latent must have shape [patches, dim]"
        )
    if num_hist < 2:
        raise ValueError("num_hist must be at least 2")
    if action_dim < 1:
        raise ValueError("action_dim must be at least 1")

    latent_history = initial_latent.unsqueeze(0).repeat(
        num_hist,
        1,
        1,
    )
    action_history = torch.zeros(
        num_hist - 1,
        action_dim,
        device=initial_latent.device,
        dtype=torch.float32,
    )
    return latent_history, action_history


def update_mpc_history(
    latent_history,
    action_history,
    executed_action,
    next_latent,
):
    if latent_history.ndim != 3 or next_latent.shape != latent_history.shape[1:]:
        raise ValueError("next_latent does not match latent history shape")
    if action_history.ndim != 2:
        raise ValueError("action_history must have shape [history, action_dim]")
    if action_history.shape[0] != latent_history.shape[0] - 1:
        raise ValueError("latent and action history lengths are not aligned")
    if executed_action.shape != action_history.shape[1:]:
        raise ValueError("executed_action does not match action dimension")

    next_latent = next_latent.to(
        device=latent_history.device,
        dtype=latent_history.dtype,
    )
    executed_action = executed_action.to(
        device=action_history.device,
        dtype=action_history.dtype,
    )
    new_latents = torch.cat(
        [latent_history[1:], next_latent.unsqueeze(0)],
        dim=0,
    )
    new_actions = torch.cat(
        [action_history[1:], executed_action.unsqueeze(0)],
        dim=0,
    )
    return new_latents, new_actions


def plan_action(
    model,
    latent_history,
    action_history,
    target_latent,
    device,
    cost_mode,
    action_low,
    action_high,
    planning_horizon,
    population,
    num_elites,
    num_iterations,
    candidate_batch_size,
    position_probe=None,
    position_mean=None,
    position_std=None,
    oracle_goal=None,
    action_mean=None,
    action_std=None,
    precision="fp32",
    min_std=0.05,
    generator=None,
):
    latent_history = latent_history.unsqueeze(0)
    action_low = action_low.to(device=device, dtype=torch.float32)
    action_high = action_high.to(device=device, dtype=torch.float32)

    def score_candidates(candidate_actions):
        return score_action_sequences(
            model=model,
            latent_history=latent_history,
            action_history=action_history,
            candidate_actions=candidate_actions,
            device=device,
            chunk_size=candidate_batch_size,
            action_mean=action_mean,
            action_std=action_std,
            precision=precision,
            cost_fn=lambda predicted_latents: compute_terminal_cost(
                predicted_latents=predicted_latents,
                target_latent=target_latent,
                mode=cost_mode,
                position_probe=position_probe,
                position_mean=position_mean,
                position_std=position_std,
                oracle_goal=oracle_goal,
            ),
        )

    return cem_optimize(
        score_fn=score_candidates,
        horizon=planning_horizon,
        action_dim=action_history.shape[-1],
        action_low=action_low,
        action_high=action_high,
        population=population,
        num_elites=num_elites,
        num_iterations=num_iterations,
        min_std=min_std,
        generator=generator,
    )


def hide_goal_marker(env):
    point_maze = env.unwrapped
    point_maze.point_env.model.site_rgba[
        point_maze.target_site_id,
        3,
    ] = 0.0


def _goal_distance(observation):
    return float(
        np.linalg.norm(
            np.asarray(observation["achieved_goal"])
            - np.asarray(observation["desired_goal"])
        )
    )


def run_planning_episode(
    env,
    encoder,
    model,
    device,
    seed,
    cost_mode,
    max_macro_steps,
    planning_horizon,
    population,
    num_elites,
    num_iterations,
    candidate_batch_size,
    position_probe=None,
    position_mean=None,
    position_std=None,
    action_mean=None,
    action_std=None,
    precision="fp32",
    min_std=0.05,
    generator=None,
    planner_fn=plan_action,
):
    observation, reset_info = env.reset(seed=seed)
    hide_goal_marker(env)
    initial_frame = validate_frame(env.render())
    target_frame = render_target_observation(env)
    initial_latent = encode_frame(
        encoder,
        initial_frame,
        device,
        precision,
    )
    target_latent = encode_frame(
        encoder,
        target_frame,
        device,
        precision,
    )

    num_hist = int(model.num_hist) if hasattr(model, "num_hist") else 3
    action_dim = 10
    latent_history, action_history = initialize_mpc_history(
        initial_latent=initial_latent,
        num_hist=num_hist,
        action_dim=action_dim,
    )
    action_low = torch.from_numpy(
        np.tile(np.asarray(env.action_space.low), 5)
    ).float()
    action_high = torch.from_numpy(
        np.tile(np.asarray(env.action_space.high), 5)
    ).float()
    oracle_goal = None
    if cost_mode == "oracle":
        oracle_goal = torch.as_tensor(
            observation["desired_goal"],
            dtype=torch.float32,
        )

    initial_distance = _goal_distance(observation)
    achieved_positions = [
        np.asarray(observation["achieved_goal"], dtype=np.float32).tolist()
    ]
    macro_actions = []
    total_micro_steps = 0
    success = bool(reset_info.get("success", False))
    terminated = False
    truncated = False
    final_planner_cost = None

    for _ in range(max_macro_steps):
        if success or terminated or truncated:
            break
        plan = planner_fn(
            model=model,
            latent_history=latent_history,
            action_history=action_history,
            target_latent=target_latent,
            device=device,
            cost_mode=cost_mode,
            action_low=action_low,
            action_high=action_high,
            planning_horizon=planning_horizon,
            population=population,
            num_elites=num_elites,
            num_iterations=num_iterations,
            candidate_batch_size=candidate_batch_size,
            position_probe=position_probe,
            position_mean=position_mean,
            position_std=position_std,
            oracle_goal=oracle_goal,
            action_mean=action_mean,
            action_std=action_std,
            precision=precision,
            min_std=min_std,
            generator=generator,
        )
        macro_action = plan.action_sequence[0].detach().cpu()
        step_result = execute_macro_action(env, macro_action)
        observation = step_result.observation
        macro_actions.append(macro_action.tolist())
        total_micro_steps += step_result.micro_steps
        achieved_positions.append(
            np.asarray(
                observation["achieved_goal"],
                dtype=np.float32,
            ).tolist()
        )
        final_planner_cost = plan.cost

        next_latent = encode_frame(
            encoder,
            validate_frame(env.render()),
            device,
            precision,
        )
        latent_history, action_history = update_mpc_history(
            latent_history=latent_history,
            action_history=action_history,
            executed_action=macro_action,
            next_latent=next_latent,
        )
        success = step_result.success
        terminated = step_result.terminated
        truncated = step_result.truncated

    return {
        "seed": seed,
        "cost_mode": cost_mode,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "macro_steps": len(macro_actions),
        "micro_steps": total_micro_steps,
        "initial_goal_distance": initial_distance,
        "final_goal_distance": _goal_distance(observation),
        "final_planner_cost": final_planner_cost,
        "macro_actions": macro_actions,
        "achieved_positions": achieved_positions,
    }


def create_environment(environment_id, max_micro_steps):
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    return gym.make(
        environment_id,
        render_mode="rgb_array",
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        continuing_task=True,
        reset_target=True,
        max_episode_steps=max_micro_steps,
    )


def _create_generator(device, seed):
    if device.type in {"cpu", "cuda"}:
        return torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    return None


def _write_json_atomically(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    temporary_path.replace(output_path)


def main(argv=None):
    args = parse_args(argv)
    validate_arguments(args)
    device = resolve_device(args.device)
    precision = resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    encoder = load_frozen_ijepa_encoder(
        checkpoint_path=args.encoder_checkpoint,
        device=device,
    )
    model, model_metadata, model_epoch = load_model_and_metadata(
        args.world_model_checkpoint,
        device,
    )
    model_configuration = model_metadata["model_configuration"]
    if model_configuration["action_dim"] != 10:
        raise ValueError("world model action_dim must be 10")
    model = compile_model(model, enabled=args.compile)

    position_probe = None
    position_mean = None
    position_std = None
    probe_epoch = None
    if args.cost in {"probe", "oracle"}:
        position_probe, probe_metadata, probe_epoch = (
            load_probe_checkpoint(
                args.position_probe_checkpoint,
                device,
            )
        )
        position_mean = probe_metadata["position_mean"]
        position_std = probe_metadata["position_std"]

    action_mean = None
    action_std = None
    if model_metadata.get("normalize_actions", False):
        action_mean = model_metadata.get("action_mean")
        action_std = model_metadata.get("action_std")
        if action_mean is None or action_std is None:
            raise ValueError(
                "normalized-action checkpoint is missing action statistics"
            )

    print(f"Device: {device}, precision: {precision}")
    print(f"World-model epoch: {model_epoch + 1}")
    if probe_epoch is not None:
        print(f"Position-probe epoch: {probe_epoch + 1}")
    print(f"Cost: {args.cost}")
    print(
        f"CEM: H={args.planning_horizon}, "
        f"population={args.population}, elites={args.num_elites}, "
        f"iterations={args.cem_iterations}"
    )

    environment = create_environment(
        environment_id=args.environment_id,
        max_micro_steps=args.max_macro_steps * 5,
    )
    records = []
    try:
        for episode_index in range(args.num_episodes):
            seed = args.base_seed + episode_index
            generator = _create_generator(device, seed)
            record = run_planning_episode(
                env=environment,
                encoder=encoder,
                model=model,
                device=device,
                seed=seed,
                cost_mode=args.cost,
                max_macro_steps=args.max_macro_steps,
                planning_horizon=args.planning_horizon,
                population=args.population,
                num_elites=args.num_elites,
                num_iterations=args.cem_iterations,
                candidate_batch_size=args.candidate_batch_size,
                position_probe=position_probe,
                position_mean=position_mean,
                position_std=position_std,
                action_mean=action_mean,
                action_std=action_std,
                precision=precision,
                min_std=args.min_std,
                generator=generator,
            )
            records.append(record)
            print(
                f"episode {episode_index + 1}/{args.num_episodes}: "
                f"seed={seed} success={record['success']} "
                f"macro_steps={record['macro_steps']} "
                f"final_distance={record['final_goal_distance']:.4f}"
            )
    finally:
        environment.close()

    summary = summarize_results(records)
    payload = {
        "configuration": {
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "world_model_checkpoint": str(args.world_model_checkpoint),
            "position_probe_checkpoint": (
                str(args.position_probe_checkpoint)
                if args.cost in {"probe", "oracle"}
                else None
            ),
            "environment_id": args.environment_id,
            "cost": args.cost,
            "num_episodes": args.num_episodes,
            "base_seed": args.base_seed,
            "max_macro_steps": args.max_macro_steps,
            "planning_horizon": args.planning_horizon,
            "population": args.population,
            "num_elites": args.num_elites,
            "cem_iterations": args.cem_iterations,
            "candidate_batch_size": args.candidate_batch_size,
            "min_std": args.min_std,
            "device": str(device),
            "precision": precision,
            "compile": args.compile,
        },
        "summary": summary,
        "episodes": records,
    }
    _write_json_atomically(payload, args.output)

    print(
        "Summary: "
        f"success_rate={summary['success_rate']:.1%}, "
        "mean_final_distance="
        f"{summary['mean_final_goal_distance']:.4f}, "
        "median_final_distance="
        f"{summary['median_final_goal_distance']:.4f}"
    )
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
