"""Successful real-trajectory replay and fixed-schedule metric fitting."""

from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter, token_goal_cost


def eligible_pairs(states, steps, *, loop_radius=0.5, min_gap=2, min_separation=2.0,
                   goal=None, min_goal_progress=None):
    """Erase revisits, then retain separated temporal comparisons (not shortest paths)."""
    states = torch.as_tensor(states, device="cpu", dtype=torch.float32)
    steps = torch.as_tensor(steps, device="cpu")
    if (states.ndim != 2 or states.shape[1] != 2 or steps.shape != (len(states),)
            or not torch.isfinite(states).all() or not torch.isfinite(steps).all()
            or len(states) < 1 or not (steps[1:] > steps[:-1]).all()):
        raise ValueError("states must align with strictly increasing primitive steps")
    retained = []
    for index, state in enumerate(states):
        revisit = next((j for j, prior in enumerate(retained)
                        if torch.linalg.vector_norm(state - states[prior]) <= loop_radius), None)
        if revisit is not None:
            retained = retained[:revisit]
        retained.append(index)
    pairs = [(i, j) for offset, i in enumerate(retained) for j in retained[offset + 1:]
             if steps[j] - steps[i] >= min_gap
             and torch.linalg.vector_norm(states[i] - states[j]) >= min_separation]
    if min_goal_progress is not None:
        if (not np.isfinite(min_goal_progress) or min_goal_progress <= 0):
            raise ValueError("minimum goal progress must be finite and positive")
        try:
            goal = torch.as_tensor(goal, device="cpu", dtype=torch.float32)
        except (TypeError, ValueError):
            raise ValueError("oracle pair filtering requires a finite 2D goal") from None
        if goal.shape != (2,) or not torch.isfinite(goal).all():
            raise ValueError("oracle pair filtering requires a finite 2D goal")
        distances = torch.linalg.vector_norm(states - goal, dim=1)
        pairs = [(i, j) for i, j in pairs
                 if float(distances[i] - distances[j]) >= min_goal_progress]
    return retained, pairs


def fit_adapter(episodes, goal, *, seed, updates=500, batch_size=16, lr=1e-4,
                bottleneck=64, anchor_weight=1.0, device="cpu", progress=None,
                min_goal_progress=None, initial_adapter=None, balance_bands=False):
    """Fit on training successes only; evaluation data is never a selection set."""
    if (updates < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0
            or not np.isfinite(anchor_weight) or anchor_weight < 0):
        raise ValueError("invalid training schedule")
    goal = goal.detach().cpu().float().clone()
    if goal.ndim != 2 or not torch.isfinite(goal).all():
        raise ValueError("goal must be finite patch tokens")
    usable, summaries, all_costs = [], [], []
    for episode in episodes:
        if episode.get("split") != "train":
            raise ValueError("only training episodes may enter adaptation replay")
        if not episode.get("success"):
            continue
        tokens = episode["latents"].detach().cpu()
        if tokens.ndim != 3 or tokens.shape[1:] != goal.shape or not torch.isfinite(tokens).all():
            raise ValueError("replay tokens must be finite and match goal")
        if len(episode["states"]) != len(tokens):
            raise ValueError("replay frames, states and primitive steps must align")
        retained, temporal_pairs = eligible_pairs(episode["states"], episode["primitive_steps"])
        if min_goal_progress is None:
            pairs = temporal_pairs
            summary = dict(case_id=episode["case_id"], frames=len(tokens),
                           retained=len(retained), eligible_pairs=len(pairs))
        else:
            case = episode.get("case")
            if not isinstance(case, dict) or "goal" not in case:
                raise ValueError("oracle pair filtering requires each episode case goal")
            _, pairs = eligible_pairs(
                episode["states"], episode["primitive_steps"], goal=case["goal"],
                min_goal_progress=min_goal_progress,
            )
            summary = dict(case_id=episode["case_id"], frames=len(tokens), retained=len(retained),
                           temporal_pairs=len(temporal_pairs), eligible_pairs=len(pairs))
        summaries.append(summary)
        costs = torch.cat([(chunk.float() - goal).square().mean(dim=(1, 2))
                           for chunk in tokens.split(16)])
        all_costs.append(costs)
        if pairs:
            usable.append((episode, pairs, costs))
    if not all_costs:
        raise ValueError("no successful training episodes")
    if not usable:
        raise ValueError("no eligible temporal pairs in successful replay")
    positive = torch.cat(all_costs)
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("successful training replay has zero cost scale")
    scale = float(positive.median())
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    if initial_adapter is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            adapter = ResidualTokenAdapter(goal.shape[-1], bottleneck).to(device)
    else:
        if (not isinstance(initial_adapter, ResidualTokenAdapter)
                or initial_adapter.dim != goal.shape[-1] or initial_adapter.bottleneck != bottleneck):
            raise ValueError("initial adapter dimensions differ from fitting configuration")
        adapter = deepcopy(initial_adapter).to(device).requires_grad_(True).train()
    groups, band_sample_counts, episode_sample_counts = {}, {}, {}
    if balance_bands:
        for item in usable:
            band = item[0].get("case", {}).get("band")
            if not isinstance(band, str) or not band:
                raise ValueError("balanced replay requires a band on each usable episode")
            groups.setdefault(band, []).append(item)
        band_names = list(groups)
        band_sample_counts = dict.fromkeys(band_names, 0)
        episode_sample_counts = {item[0]["case_id"]: 0 for item in usable}
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    target = goal.to(device)
    history = []
    for update in range(updates):
        early, late, anchors, anchor_targets = [], [], [], []
        for _ in range(batch_size):
            if balance_bands:
                band = band_names[int(rng.integers(len(band_names)))]
                pool = groups[band]
                episode, pairs, raw_costs = pool[int(rng.integers(len(pool)))]
                band_sample_counts[band] += 1
                episode_sample_counts[episode["case_id"]] += 1
            else:
                episode, pairs, raw_costs = usable[int(rng.integers(len(usable)))]
            i, j = pairs[int(rng.integers(len(pairs)))]
            # Reserve this draw even in the ablation so temporal pairs have
            # the same RNG sequence as the anchored objective.
            anchor = int(rng.integers(len(episode["latents"])))
            early.append(episode["latents"][i])
            late.append(episode["latents"][j])
            if anchor_weight:
                anchors.append(episode["latents"][anchor])
                anchor_targets.append(float(raw_costs[anchor]))
        inputs = torch.stack(early + late + anchors).to(device=device, dtype=torch.float32)
        costs = token_goal_cost(adapter, inputs, target)
        first, second = costs[:batch_size], costs[batch_size:2*batch_size]
        rank_loss = F.softplus(0.2 - (first - second) / scale).mean()
        if anchor_weight:
            raw = torch.tensor(anchor_targets, device=device)
            anchor_loss = ((costs[2*batch_size:] - raw) / scale).square().mean()
            loss = rank_loss + anchor_weight * anchor_loss
        else:
            anchor_loss = rank_loss.new_zeros(())
            loss = rank_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite adapter loss; refusing to save a trained adapter")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(update=update + 1, loss=float(loss.detach()),
                            rank_loss=float(rank_loss.detach()), anchor_loss=float(anchor_loss.detach())))
        if progress is not None:
            progress(history[-1])
    adapter.eval()
    result = dict(seed=seed, updates=updates, batch_size=batch_size, lr=lr,
                  bottleneck=bottleneck, anchor_weight=anchor_weight, scale=scale,
                  history=history, optimizer=optimizer.state_dict(), trajectories=summaries,
                  training_case_ids=[item[0]["case_id"] for item in usable],
                  loss="softplus(0.2-(early-late)/scale)+anchor_weight*normalized_anchor_mse")
    if not anchor_weight:
        result.update(anchoring_enabled=False, loss="softplus(0.2-(early-late)/scale)")
    if min_goal_progress is not None:
        result["pair_filter"] = dict(kind="oracle_goal_progress",
                                     min_goal_progress=float(min_goal_progress))
    if initial_adapter is not None or balance_bands:
        result.update(warm_start=initial_adapter is not None, optimizer_reset=True)
    if balance_bands:
        result.update(sampling="uniform_band_then_episode_then_pair",
                      band_episode_counts={band: len(pool) for band, pool in groups.items()},
                      band_sample_counts=band_sample_counts, episode_sample_counts=episode_sample_counts)
    return adapter, result


def save_adapter(path, adapter, training, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(schema_version=1, dim=adapter.dim, bottleneck=adapter.bottleneck,
                    adapter={key: value.detach().cpu() for key, value in adapter.state_dict().items()},
                    training=training, provenance=provenance), temporary)
    temporary.replace(path)


def load_adapter(path, device, *, expected_runtime_id=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if expected_runtime_id is not None and saved["provenance"].get("runtime_id") != expected_runtime_id:
        raise ValueError("adapter provenance differs from current pilot")
    adapter = ResidualTokenAdapter(saved["dim"], saved["bottleneck"])
    adapter.load_state_dict(saved["adapter"])
    return adapter.to(device).eval().requires_grad_(False), saved
