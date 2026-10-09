"""Plan in PointMaze with frozen I-JEPA latent world models."""

from dataclasses import dataclass

import numpy as np
import torch

from src.mylonas_ijepa_wm.planning import (
    cem_optimize,
    compute_terminal_cost,
    rollout_action_sequences,
)


IMAGE_SIZE = 224


@dataclass(frozen=True)
class MacroStepResult:
    observation: dict
    info: dict
    terminated: bool
    truncated: bool
    success: bool
    micro_steps: int


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
        predicted_latents = rollout_action_sequences(
            model=model,
            latent_history=latent_history,
            action_history=action_history,
            candidate_actions=candidate_actions,
            device=device,
            chunk_size=candidate_batch_size,
            action_mean=action_mean,
            action_std=action_std,
            precision=precision,
        )
        return compute_terminal_cost(
            predicted_latents=predicted_latents,
            target_latent=target_latent,
            mode=cost_mode,
            position_probe=position_probe,
            position_mean=position_mean,
            position_std=position_std,
            oracle_goal=oracle_goal,
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
