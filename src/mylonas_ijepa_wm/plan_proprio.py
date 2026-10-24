"""CEM planning with a proprioceptive world model in a selected environment."""

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm import plan_pointmaze_proprio as planner
from src.mylonas_ijepa_wm import wm, wm_proprio
from src.mylonas_ijepa_wm.environment_adapters import create_adapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig, load_environment_config


def positive_finite_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["pointmaze", "wall"], default="pointmaze")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path, default=visual.DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--cost", choices=["latent", "position"], default="latent")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-macro-steps", type=int, default=20)
    parser.add_argument("--planning-horizon", type=int, default=3)
    parser.add_argument("--actions-per-plan", type=int, default=1)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--num-elites", type=int, default=32)
    parser.add_argument("--cem-iterations", type=int, default=5)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument(
        "--unbounded-actions", action="store_true",
        help="Let Wall CEM search beyond the action range used during training.",
    )
    parser.add_argument(
        "--max-action-length", type=positive_finite_float,
        help="Maximum Euclidean length of each 2D environment action proposed by CEM.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def run_episode(adapter, config, encoder, model, metadata, args, device, precision, seed, frames):
    initial = adapter.reset(seed)
    state_mean = metadata["state_mean"].to(device=device, dtype=torch.float32)
    state_std = metadata["state_std"].to(device=device, dtype=torch.float32)
    num_hist = model.num_hist
    latent = visual.encode_frame(encoder, initial.frame, device, precision)
    goal_latent = (
        visual.encode_frame(encoder, adapter.render_goal(), device, precision)
        if args.cost == "latent" else None
    )
    latent_history = latent.unsqueeze(0).repeat(num_hist, 1, 1)
    state = torch.as_tensor(initial.state, device=device, dtype=torch.float32)
    state_history = ((state - state_mean) / state_std).unsqueeze(0).repeat(num_hist, 1)
    action_history = torch.zeros(num_hist - 1, config.macro_action_dim, device=device)
    low = torch.as_tensor(np.tile(adapter.action_low, config.actions_per_model_step))
    high = torch.as_tensor(np.tile(adapter.action_high, config.actions_per_model_step))
    if getattr(args, "unbounded_actions", False):
        low = torch.full_like(low, -torch.inf)
        high = torch.full_like(high, torch.inf)
    initial_distance = float(np.linalg.norm(initial.state[:2] - adapter.goal))
    positions = [initial.state[:2].tolist()]
    states = [initial.state.tolist()]
    actions = []
    if frames is not None:
        frames.append(adapter.render_video())
    success = initial.success
    terminated = truncated = False
    final_cost = None
    micro_steps = 0
    rng = visual._create_generator(device, seed)
    while len(actions) < args.max_macro_steps and not (success or terminated or truncated):
        plan = planner.plan_action(
            model=model, latent_history=latent_history,
            state_history=state_history, action_history=action_history,
            target_latent=goal_latent, action_low=low, action_high=high,
            device=device, planning_horizon=args.planning_horizon,
            population=args.population, num_elites=args.num_elites,
            num_iterations=args.cem_iterations,
            candidate_batch_size=args.candidate_batch_size,
            action_mean=metadata.get("action_mean"),
            action_std=metadata.get("action_std"), precision=precision,
            min_std=args.min_std, generator=rng, cost=args.cost,
            goal_position=torch.as_tensor(adapter.goal, device=device),
            state_mean=state_mean, state_std=state_std,
            max_action_length=getattr(args, "max_action_length", None),
            micro_action_dim=config.micro_action_dim,
        )
        for macro in plan.action_sequence[:min(args.actions_per_plan, args.max_macro_steps - len(actions))]:
            macro = macro.detach().cpu().numpy().astype(np.float32)
            for micro in macro.reshape(config.actions_per_model_step, config.micro_action_dim):
                outcome = adapter.step(micro)
                micro_steps += 1
                if frames is not None:
                    frames.append(adapter.render_video())
                success = outcome.success
                terminated = outcome.terminated
                truncated = outcome.truncated
                if success or terminated or truncated:
                    break
            next_state = torch.as_tensor(outcome.state, device=device, dtype=torch.float32)
            next_latent = visual.encode_frame(encoder, outcome.frame, device, precision)
            latent_history = torch.cat([latent_history[1:], next_latent.unsqueeze(0)], dim=0)
            state_history = torch.cat([
                state_history[1:], ((next_state - state_mean) / state_std).unsqueeze(0)
            ], dim=0)
            if num_hist > 1:
                action_history = torch.cat([
                    action_history[1:], torch.as_tensor(macro, device=device).unsqueeze(0)
                ], dim=0)
            actions.append(macro.tolist())
            positions.append(outcome.state[:2].tolist())
            states.append(outcome.state.tolist())
            final_cost = plan.cost
            if success or terminated or truncated:
                break
    return {
        "seed": seed, "environment": config.env, "cost_mode": args.cost,
        "wall_layout": list(adapter.layout) if config.env == "wall" else None,
        "success": bool(success), "terminated": bool(terminated),
        "truncated": bool(truncated), "macro_steps": len(actions),
        "micro_steps": micro_steps,
        "initial_goal_distance": initial_distance,
        "final_goal_distance": float(np.linalg.norm(states[-1][:2] - adapter.goal)),
        "final_planner_cost": final_cost,
        "failed_by_step_limit": not success and len(actions) == args.max_macro_steps,
        "goal_position": adapter.goal.tolist(),
        "macro_actions": actions, "achieved_positions": positions,
        "proprio_states": states,
    }


def main(argv=None):
    args = parse_args(argv)
    for path in (args.encoder_checkpoint, args.world_model_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.num_episodes, args.max_macro_steps, args.planning_horizon,
           args.actions_per_plan, args.population, args.num_elites,
           args.cem_iterations, args.candidate_batch_size, args.video_fps) < 1:
        raise ValueError("planning counts must be positive")
    if args.actions_per_plan > args.planning_horizon or args.num_elites > args.population:
        raise ValueError("invalid CEM planning configuration")
    if args.unbounded_actions and args.env != "wall":
        raise ValueError("--unbounded-actions is only supported for Wall")
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    model, metadata, epoch = wm_proprio.load_model_and_metadata(
        args.world_model_checkpoint, device
    )
    saved_env = metadata.get("environment", "pointmaze")
    if args.env != saved_env:
        raise ValueError(f"Checkpoint belongs to {saved_env}, not {args.env}")
    saved_config = metadata.get("environment_config")
    if args.env_config is not None:
        config = load_environment_config(args.env, args.env_config)
    elif saved_config:
        config = EnvironmentConfig(**saved_config)
    else:
        config = replace(
            load_environment_config(args.env),
            pointmaze_physics_frame_skip=int(metadata.get("physics_frame_skip", 1)),
            actions_per_model_step=metadata["model_configuration"]["action_dim"] // 2,
        )
    shape = metadata["model_configuration"]
    if shape["num_patches"] != 256 or shape["ijepa_dim"] != 1280:
        raise ValueError("Checkpoint does not match the frozen I-JEPA encoder")
    if shape["action_dim"] != config.macro_action_dim or shape["proprio_dim"] != len(config.state_fields):
        raise ValueError("Checkpoint action/state dimensions do not match environment")
    if args.env == "pointmaze" and config.pointmaze_physics_frame_skip != int(metadata.get("physics_frame_skip", 1)):
        raise ValueError("PointMaze physics timing differs from the checkpoint")
    model = wm.compile_model(model, enabled=args.compile)
    encoder = visual.load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
    adapter = create_adapter(config, args.max_macro_steps * config.actions_per_model_step)
    records = []
    try:
        for index in range(args.num_episodes):
            frames = [] if args.video_dir else None
            seed = args.base_seed + index
            record = run_episode(
                adapter, config, encoder, model, metadata, args, device, precision,
                seed, frames,
            )
            record["video_path"] = None
            if frames:
                record["micro_steps"] = len(frames) - 1
                record["video_path"] = str(visual.save_episode_video(
                    frames, args.video_dir, index, seed, record["success"], args.video_fps
                ))
            records.append(record)
            print(f"episode {index + 1}/{args.num_episodes}: success={record['success']}")
    finally:
        adapter.close()
    payload = {
        "configuration": {
            "environment": config.portable_metadata(),
            "checkpoint": str(args.world_model_checkpoint),
            "checkpoint_epoch": epoch,
            "cost": args.cost,
            "planning_horizon": args.planning_horizon,
            "actions_per_plan": args.actions_per_plan,
            "unbounded_actions": args.unbounded_actions,
            "max_action_length": args.max_action_length,
        },
        "summary": visual.summarize_results(records),
        "episodes": records,
    }
    visual._write_json_atomically(payload, args.output)
    print(f"Success rate: {payload['summary']['success_rate']:.1%}")


if __name__ == "__main__":
    main()
