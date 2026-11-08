"""Random first-action assignment and realized boundary-entry measurements."""

import numpy as np

from src.mylonas_ijepa_wm.geodesic_radial import WALL_RIGHT


def configure(manifest, probability, success_quota):
    """Add independent, reproducible actions to the pre-sampled start cases."""
    rng = np.random.default_rng([manifest["seed"], 1])
    max_length = manifest["planning"]["max_action_length"]
    manifest["exploration"] = dict(probability=probability, success_quota=success_quota,
                                   max_attempts_per_goal=manifest["starts_per_goal"])
    for cases in manifest["stages"]:
        for case in cases:
            draw, angle, radius = rng.random(3)
            case["random_first_action"] = (
                (max_length * np.sqrt(radius) * np.array([np.cos(2 * np.pi * angle),
                                                          np.sin(2 * np.pi * angle)])).tolist()
                if draw < probability else None
            )


def measure(record, inner, macro_steps, *, right_room_only=True):
    xy = np.asarray(record["primitive_positions"])
    offsets = xy - record["case"]["goal"]
    distance = np.linalg.norm(offsets, axis=1)
    extent = np.abs(offsets).max(axis=1)
    inside = extent[1:] <= inner + 1e-6
    if right_room_only:
        inside &= xy[1:, 0] > WALL_RIGHT
    entries = np.flatnonzero(inside) if inner else []
    first_entry = int(entries[0] + 1) if len(entries) else None
    prefix = record.get("prefix_steps", 0)
    return dict(
        arm="random first" if record["case"].get("random_first_action") is not None else "MPC first",
        prefix_steps=prefix,
        first_entry_step=first_entry,
        entered_by_first_action=first_entry == 1,
        entered_by_five_actions=first_entry is not None and first_entry <= macro_steps,
        first_distance_change=float(distance[min(1, len(xy) - 1)] - distance[0]),
        first_five_distance_change=float(distance[min(macro_steps, len(xy) - 1)] - distance[0]),
        final_distance_change=float(distance[-1] - distance[0]),
        best_distance=float(distance.min()),
        closest_extent=float(extent.min()),
        positions=record["primitive_positions"],
    )
