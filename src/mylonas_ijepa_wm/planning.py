"""Planning primitives for the action-conditioned I-JEPA world model."""

import torch

from src.mylonas_ijepa_wm.evaluate_rollouts import (
    autoregressive_rollout,
    decode_positions,
)


def assemble_rollout_actions(
    action_history,
    candidate_actions,
    action_mean=None,
    action_std=None,
):
    if action_history.ndim != 2:
        raise ValueError(
            "action_history must have shape [history, action_dim]"
        )
    if candidate_actions.ndim != 3:
        raise ValueError(
            "candidate_actions must have shape "
            "[population, horizon, action_dim]"
        )
    if action_history.shape[-1] != candidate_actions.shape[-1]:
        raise ValueError(
            "history and candidate action dimensions differ"
        )
    if (action_mean is None) != (action_std is None):
        raise ValueError(
            "action_mean and action_std must both be set or both be None"
        )

    history = action_history.unsqueeze(0).expand(
        candidate_actions.shape[0],
        -1,
        -1,
    )
    actions = torch.cat([history, candidate_actions], dim=1)

    if action_mean is not None:
        action_mean = action_mean.to(
            device=actions.device,
            dtype=actions.dtype,
        )
        action_std = action_std.to(
            device=actions.device,
            dtype=actions.dtype,
        )
        actions = (actions - action_mean) / action_std

    return actions


def rollout_action_sequences(
    model,
    latent_history,
    action_history,
    candidate_actions,
    device,
    chunk_size,
    action_mean=None,
    action_std=None,
    precision="fp32",
):
    if latent_history.ndim != 4 or latent_history.shape[0] != 1:
        raise ValueError(
            "latent_history must have batch size 1 and shape "
            "[1, num_hist, patches, dim]"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")

    num_hist = latent_history.shape[1]
    if action_history.shape[0] != num_hist - 1:
        raise ValueError(
            "action_history must contain num_hist - 1 actions"
        )

    latent_history = latent_history.to(
        device=device,
        dtype=torch.float32,
    )
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(candidate_actions), chunk_size):
            candidate_chunk = candidate_actions[
                start:start + chunk_size
            ].to(device=device, dtype=torch.float32)
            action_chunk = assemble_rollout_actions(
                action_history=action_history.to(
                    device=device,
                    dtype=torch.float32,
                ),
                candidate_actions=candidate_chunk,
                action_mean=action_mean,
                action_std=action_std,
            )
            history_chunk = latent_history.expand(
                candidate_chunk.shape[0],
                -1,
                -1,
                -1,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                predictions.append(
                    autoregressive_rollout(
                        model,
                        history_chunk,
                        action_chunk,
                    )
                )

    return torch.cat(predictions, dim=0)


def compute_terminal_cost(
    predicted_latents,
    target_latent,
    mode,
    position_probe=None,
    position_mean=None,
    position_std=None,
    oracle_goal=None,
):
    if mode not in {"probe", "latent", "oracle"}:
        raise ValueError(f"unknown cost mode: {mode}")
    if predicted_latents.ndim != 4:
        raise ValueError(
            "predicted_latents must have shape "
            "[population, horizon, patches, dim]"
        )

    terminal_latents = predicted_latents[:, -1].float()

    if mode == "latent":
        target_latent = target_latent.to(
            device=terminal_latents.device,
            dtype=terminal_latents.dtype,
        )
        return (
            terminal_latents - target_latent.unsqueeze(0)
        ).square().mean(dim=(1, 2))

    if (
        position_probe is None
        or position_mean is None
        or position_std is None
    ):
        raise ValueError(
            "position probe and position statistics are required"
        )

    predicted_positions = decode_positions(
        position_probe=position_probe,
        latents=terminal_latents,
        position_mean=position_mean,
        position_std=position_std,
    )

    if mode == "probe":
        target_position = decode_positions(
            position_probe=position_probe,
            latents=target_latent.to(terminal_latents.device).float(),
            position_mean=position_mean,
            position_std=position_std,
        )
    else:
        if oracle_goal is None:
            raise ValueError("oracle_goal is required for oracle cost")
        target_position = oracle_goal.to(
            device=predicted_positions.device,
            dtype=predicted_positions.dtype,
        )

    return torch.linalg.vector_norm(
        predicted_positions - target_position,
        dim=-1,
    )
