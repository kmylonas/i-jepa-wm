"""Plan in top-down PointMaze with a proprioceptive I-JEPA world model."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as base
from src.mylonas_ijepa_wm import wm
from src.mylonas_ijepa_wm import wm_proprio
from src.mylonas_ijepa_wm.planning import (
    assemble_rollout_actions,
    cem_optimize,
    compute_terminal_cost,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plan using top-down images, true proprioception, "
            "and a latent or position goal cost."
        )
    )
    parser.add_argument(
        "--encoder-checkpoint", type=Path,
        default=base.DEFAULT_ENCODER_CHECKPOINT,
    )
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=None,
    )
    parser.add_argument("--cost", choices=["latent", "position"], default="latent")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--environment-id", default="PointMaze_UMaze-v3")
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def rollout_action_sequences(
    model,
    latent_history,
    state_history,
    action_history,
    candidate_actions,
    device,
    chunk_size,
    action_mean=None,
    action_std=None,
    precision="fp32",
):
    if latent_history.ndim != 4 or latent_history.shape[0] != 1:
        raise ValueError("latent_history must have shape [1, hist, patches, dim]")
    if state_history.shape[:2] != latent_history.shape[:2] or state_history.shape[-1] != 4:
        raise ValueError("state_history must align with latent_history")
    if action_history.shape[0] != latent_history.shape[1] - 1:
        raise ValueError("action_history must contain hist - 1 actions")
    if candidate_actions.ndim != 3 or chunk_size < 1:
        raise ValueError("candidate_actions or chunk_size are invalid")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("unknown precision")

    latent_history = latent_history.to(device=device, dtype=torch.float32)
    state_history = state_history.to(device=device, dtype=torch.float32)
    action_history = action_history.to(device=device, dtype=torch.float32)
    visual_chunks = []
    state_chunks = []
    with torch.inference_mode():
        for start in range(0, len(candidate_actions), chunk_size):
            candidates = candidate_actions[start:start + chunk_size].to(
                device=device, dtype=torch.float32
            )
            actions = assemble_rollout_actions(
                action_history,
                candidates,
                action_mean=action_mean,
                action_std=action_std,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                visual_pred, state_pred = wm_proprio.autoregressive_rollout(
                    model,
                    latent_history.expand(len(candidates), -1, -1, -1),
                    state_history.expand(len(candidates), -1, -1),
                    actions,
                )
            visual_chunks.append(visual_pred)
            state_chunks.append(state_pred)
    return torch.cat(visual_chunks, dim=0), torch.cat(state_chunks, dim=0)


def compute_terminal_position_cost(predicted_states, goal_position, state_mean, state_std):
    """Euclidean distance from the predicted terminal position to the maze goal."""
    if predicted_states.ndim != 3 or predicted_states.shape[-1] != 4:
        raise ValueError("predicted_states must have shape [population, horizon, 4]")
    terminal_xy = predicted_states[:, -1, :2].float()
    mean_xy = torch.as_tensor(
        state_mean, device=terminal_xy.device, dtype=torch.float32
    )[:2]
    std_xy = torch.as_tensor(
        state_std, device=terminal_xy.device, dtype=torch.float32
    )[:2]
    goal_xy = torch.as_tensor(
        goal_position, device=terminal_xy.device, dtype=torch.float32
    )
    if mean_xy.shape != (2,) or std_xy.shape != (2,) or goal_xy.shape != (2,):
        raise ValueError("state normalization and goal position must contain x and y")
    return torch.linalg.vector_norm(terminal_xy * std_xy + mean_xy - goal_xy, dim=-1)


def plan_action(
    model,
    latent_history,
    state_history,
    action_history,
    target_latent,
    action_low,
    action_high,
    device,
    planning_horizon,
    population,
    num_elites,
    num_iterations,
    candidate_batch_size,
    action_mean=None,
    action_std=None,
    precision="fp32",
    min_std=0.05,
    generator=None,
    cost="latent",
    goal_position=None,
    state_mean=None,
    state_std=None,
):
    if cost not in {"latent", "position"}:
        raise ValueError(f"unknown cost mode: {cost}")
    if cost == "latent" and target_latent is None:
        raise ValueError("latent cost requires target_latent")
    if cost == "position" and any(
        value is None for value in (goal_position, state_mean, state_std)
    ):
        raise ValueError("position cost requires goal_position, state_mean, and state_std")

    def score_candidates(candidates):
        costs = []
        for start in range(0, len(candidates), candidate_batch_size):
            chunk = candidates[start:start + candidate_batch_size]
            predicted_visual, predicted_states = rollout_action_sequences(
                model,
                latent_history.unsqueeze(0),
                state_history.unsqueeze(0),
                action_history,
                chunk,
                device=device,
                chunk_size=candidate_batch_size,
                action_mean=action_mean,
                action_std=action_std,
                precision=precision,
            )
            if cost == "latent":
                costs.append(compute_terminal_cost(
                    predicted_latents=predicted_visual,
                    target_latent=target_latent,
                    mode="latent",
                ))
            else:
                costs.append(compute_terminal_position_cost(
                    predicted_states=predicted_states,
                    goal_position=goal_position,
                    state_mean=state_mean,
                    state_std=state_std,
                ))
        return torch.cat(costs, dim=0)

    return cem_optimize(
        score_fn=score_candidates,
        horizon=planning_horizon,
        action_dim=action_history.shape[-1],
        action_low=action_low.to(device=device, dtype=torch.float32),
        action_high=action_high.to(device=device, dtype=torch.float32),
        population=population,
        num_elites=num_elites,
        num_iterations=num_iterations,
        min_std=min_std,
        generator=generator,
    )


def run_planning_episode(
    env,
    encoder,
    model,
    device,
    seed,
    max_macro_steps,
    planning_horizon,
    population,
    num_elites,
    num_iterations,
    candidate_batch_size,
    state_mean,
    state_std,
    actions_per_plan=1,
    action_mean=None,
    action_std=None,
    precision="fp32",
    min_std=0.05,
    generator=None,
    frame_callback=None,
    cost="latent",
):
    if not 1 <= actions_per_plan <= planning_horizon:
        raise ValueError("actions_per_plan must be between 1 and planning_horizon")
    observation, reset_info = env.reset(seed=seed)
    base.hide_goal_marker(env)
    initial_latent = base.encode_frame(encoder, base.validate_frame(env.render()), device, precision)
    target_latent = None
    if cost == "latent":
        target_latent = base.encode_frame(
            encoder, base.render_target_observation(env), device, precision
        )
    goal_position = torch.as_tensor(
        observation["desired_goal"][:2], device=device, dtype=torch.float32
    )
    if frame_callback is not None:
        frame_callback(base.render_visualization_frame(env))
    initial_state = torch.as_tensor(observation["observation"], dtype=torch.float32)
    if initial_state.shape != (4,):
        raise ValueError("PointMaze observation must contain (x,y,vx,vy)")
    state_mean = torch.as_tensor(state_mean, device=device, dtype=torch.float32)
    state_std = torch.as_tensor(state_std, device=device, dtype=torch.float32)
    if state_mean.shape != (4,) or state_std.shape != (4,) or not torch.all(state_std > 0):
        raise ValueError("state normalization must have four positive scales")
    num_hist = model.num_hist
    latent_history = initial_latent.unsqueeze(0).repeat(num_hist, 1, 1)
    state_history = ((initial_state.to(device) - state_mean) / state_std).unsqueeze(0).repeat(num_hist, 1)
    action_history = torch.zeros(num_hist - 1, 10, device=device)
    action_low = torch.from_numpy(np.tile(np.asarray(env.action_space.low), 5)).float()
    action_high = torch.from_numpy(np.tile(np.asarray(env.action_space.high), 5)).float()
    positions = [np.asarray(observation["achieved_goal"], dtype=np.float32).tolist()]
    proprio_states = [initial_state.tolist()]
    macro_actions = []
    total_micro_steps = 0
    success = bool(reset_info.get("success", False))
    terminated = truncated = False
    final_cost = None
    initial_distance = base._goal_distance(observation)

    while len(macro_actions) < max_macro_steps and not (success or terminated or truncated):
        plan = plan_action(
            model=model,
            latent_history=latent_history,
            state_history=state_history,
            action_history=action_history,
            target_latent=target_latent,
            action_low=action_low,
            action_high=action_high,
            device=device,
            planning_horizon=planning_horizon,
            population=population,
            num_elites=num_elites,
            num_iterations=num_iterations,
            candidate_batch_size=candidate_batch_size,
            action_mean=action_mean,
            action_std=action_std,
            precision=precision,
            min_std=min_std,
            generator=generator,
            cost=cost,
            goal_position=goal_position,
            state_mean=state_mean,
            state_std=state_std,
        )
        num_actions = min(actions_per_plan, max_macro_steps - len(macro_actions))
        for macro_action in plan.action_sequence[:num_actions]:
            macro_action = macro_action.detach().cpu()
            result = base.execute_macro_action(
                env, macro_action, frame_callback=frame_callback
            )
            observation = result.observation
            next_state = torch.as_tensor(
                observation["observation"], device=device, dtype=torch.float32
            )
            next_latent = base.encode_frame(
                encoder, base.validate_frame(env.render()), device, precision
            )
            latent_history = torch.cat(
                [latent_history[1:], next_latent.unsqueeze(0)], dim=0
            )
            state_history = torch.cat(
                [state_history[1:], ((next_state - state_mean) / state_std).unsqueeze(0)],
                dim=0,
            )
            action_history = torch.cat(
                [action_history[1:], macro_action.to(device).unsqueeze(0)], dim=0
            )
            macro_actions.append(macro_action.tolist())
            positions.append(np.asarray(observation["achieved_goal"], dtype=np.float32).tolist())
            proprio_states.append(next_state.cpu().tolist())
            total_micro_steps += result.micro_steps
            final_cost = plan.cost
            success = result.success
            terminated = result.terminated
            truncated = result.truncated
            if success or terminated or truncated:
                break

    return {
        "seed": seed,
        "camera": "topdown",
        "cost_mode": cost,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "macro_steps": len(macro_actions),
        "micro_steps": total_micro_steps,
        "initial_goal_distance": initial_distance,
        "final_goal_distance": base._goal_distance(observation),
        "final_planner_cost": final_cost,
        "failed_by_step_limit": not success and len(macro_actions) == max_macro_steps,
        "macro_actions": macro_actions,
        "achieved_positions": positions,
        "proprio_states": proprio_states,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.output is None:
        args.output = wm_proprio.DEFAULT_OUTPUT / f"planning_{args.cost}.json"
    for name in ("encoder_checkpoint", "world_model_checkpoint"):
        if not getattr(args, name).is_file():
            raise FileNotFoundError(getattr(args, name))
    for name in (
        "num_episodes", "max_macro_steps", "planning_horizon", "actions_per_plan", "population",
        "num_elites", "cem_iterations", "candidate_batch_size", "video_fps",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_elites > args.population or args.min_std <= 0:
        raise ValueError("invalid CEM elite count or minimum standard deviation")
    if args.actions_per_plan > args.planning_horizon:
        raise ValueError("actions_per_plan cannot exceed planning_horizon")
    device = wm.resolve_device(args.device)
    precision = wm.resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, metadata, epoch = wm_proprio.load_model_and_metadata(
        args.world_model_checkpoint, device
    )
    configuration = metadata["model_configuration"]
    if (
        configuration["num_patches"] != 256
        or configuration["ijepa_dim"] != 1280
        or configuration["action_dim"] != 10
        or configuration["proprio_dim"] != 4
    ):
        raise ValueError("checkpoint geometry is incompatible with PointMaze planning")
    model = wm.compile_model(model, enabled=args.compile)
    encoder = base.load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
    environment = base.create_environment(
        args.environment_id, args.max_macro_steps * 5, camera="topdown"
    )
    records = []
    print(f"Device: {device}, precision: {precision}, camera: topdown")
    print(f"World-model epoch: {epoch + 1}, cost: {args.cost}")
    try:
        for episode_index in range(args.num_episodes):
            seed = args.base_seed + episode_index
            frames = [] if args.video_dir is not None else None
            record = run_planning_episode(
                env=environment,
                encoder=encoder,
                model=model,
                device=device,
                seed=seed,
                max_macro_steps=args.max_macro_steps,
                planning_horizon=args.planning_horizon,
                actions_per_plan=args.actions_per_plan,
                population=args.population,
                num_elites=args.num_elites,
                num_iterations=args.cem_iterations,
                candidate_batch_size=args.candidate_batch_size,
                state_mean=metadata["state_mean"],
                state_std=metadata["state_std"],
                action_mean=metadata.get("action_mean"),
                action_std=metadata.get("action_std"),
                precision=precision,
                min_std=args.min_std,
                generator=base._create_generator(device, seed),
                frame_callback=frames.append if frames is not None else None,
                cost=args.cost,
            )
            record["video_path"] = None
            if frames is not None:
                record["video_path"] = str(base.save_episode_video(
                    frames,
                    args.video_dir,
                    episode_index,
                    seed,
                    record["success"],
                    args.video_fps,
                ))
            records.append(record)
            print(
                f"episode {episode_index + 1}/{args.num_episodes}: "
                f"seed={seed} success={record['success']} "
                f"final_distance={record['final_goal_distance']:.4f}"
            )
    finally:
        environment.close()
    summary = base.summarize_results(records)
    payload = {
        "configuration": {
            "encoder_checkpoint": str(args.encoder_checkpoint),
            "world_model_checkpoint": str(args.world_model_checkpoint),
            "world_model_epoch": epoch,
            "camera": "topdown",
            "cost": args.cost,
            "environment_id": args.environment_id,
            "num_episodes": args.num_episodes,
            "base_seed": args.base_seed,
            "max_macro_steps": args.max_macro_steps,
            "planning_horizon": args.planning_horizon,
            "actions_per_plan": args.actions_per_plan,
            "population": args.population,
            "num_elites": args.num_elites,
            "cem_iterations": args.cem_iterations,
            "candidate_batch_size": args.candidate_batch_size,
            "min_std": args.min_std,
            "device": str(device),
            "precision": precision,
            "compile": args.compile,
            "video_dir": None if args.video_dir is None else str(args.video_dir),
            "video_fps": args.video_fps,
        },
        "summary": summary,
        "episodes": records,
    }
    base._write_json_atomically(payload, args.output)
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
