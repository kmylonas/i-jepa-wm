"""Audit DotWall action support, collisions, and doorway crossings before embedding."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm.wall_layout_manifest import (
    load_manifest, manifest_digest, validate_manifest,
)


CONTACT_THRESHOLD = 0.5


def classify_transition(start, action, end, layout) -> str:
    """Classify one primitive DotWall transition from actual versus free motion."""
    start = np.asarray(start, dtype=np.float64)[:2]
    action = np.asarray(action, dtype=np.float64)[:2]
    end = np.asarray(end, dtype=np.float64)[:2]
    wall_x, door_y = layout
    proposal = start + 2.0 * action
    discrepancy = np.linalg.norm(end - proposal)
    if discrepancy > CONTACT_THRESHOLD:
        segment = start[None] + np.linspace(0.0, 1.0, 81)[:, None] * (
            proposal - start
        )[None]
        solid = (
            (segment[:, 0] >= wall_x - 3)
            & (segment[:, 0] <= wall_x + 3)
            & ((segment[:, 1] < door_y - 4) | (segment[:, 1] > door_y + 4))
        )
        if solid.any():
            return "solid_contact"
        if (np.any(proposal <= 4) or np.any(proposal >= 60)
                or np.any(end <= 4.5) or np.any(end >= 59.5)):
            return "border_contact"
        return "other"
    if (start[0] < wall_x <= end[0]) or (end[0] < wall_x <= start[0]):
        if abs(proposal[0] - start[0]) > 1e-9:
            fraction = (wall_x - start[0]) / (proposal[0] - start[0])
            crossing_y = start[1] + fraction * (proposal[1] - start[1])
            if door_y - 4 <= crossing_y <= door_y + 4:
                return (
                    "door_crossing_left_to_right" if end[0] > start[0]
                    else "door_crossing_right_to_left"
                )
    return "other"


def _empty_counts():
    return dict(
        episodes=0, actions=0, length_gt_1_8=0, length_gt_3_0=0,
        length_le_3_5=0, solid_contacts=0,
        door_crossings_left_to_right=0, door_crossings_right_to_left=0,
        border_contacts=0, examples={}, policies={}, by_policy={},
    )


def audit_dataset(raw_dir: Path, manifest: dict) -> dict:
    manifest = validate_manifest(manifest)
    raw_dir = Path(raw_dir)
    digest = manifest_digest(manifest)
    expected_files = {
        f"episode_{entry['episode_id']:04d}.pt" for entry in manifest["episodes"]
    }
    actual_files = {path.name for path in raw_dir.glob("episode_*.pt")}
    missing = expected_files - actual_files
    extra = actual_files - expected_files
    if missing or extra:
        raise ValueError(f"Raw episode files disagree with manifest: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    layouts = {
        f"{layout[0]},{layout[1]}": _empty_counts()
        for split in ("train", "val", "test")
        for layout in manifest[f"{split}_layouts"]
    }
    action_hashes = {}
    action_range_violations = []
    for entry in manifest["episodes"]:
        episode_id = entry["episode_id"]
        episode = torch.load(
            raw_dir / f"episode_{episode_id:04d}.pt",
            map_location="cpu", weights_only=True,
        )
        if (episode.get("episode_index") != episode_id
                or episode.get("seed") != entry["seed"]
                or tuple(episode.get("wall_layout") or ()) != tuple(entry["wall_layout"])
                or episode.get("collection_policy") != entry["collection_policy"]
                or episode.get("manifest_sha256") != digest):
            raise ValueError(f"Raw episode {episode_id} disagrees with manifest")
        actions = np.asarray(episode["actions"], dtype=np.float64)
        states = np.asarray(episode["states"], dtype=np.float64)
        if actions.shape != (manifest["steps_per_episode"], 2) or states.shape != (len(actions) + 1, 2):
            raise ValueError(f"Raw episode {episode_id} has invalid action/state shapes")
        action_hashes[str(entry["seed"])] = sha256(np.asarray(
            episode["actions"], dtype=np.float32,
        ).tobytes()).hexdigest()
        key = ",".join(map(str, entry["wall_layout"]))
        counts = layouts[key]
        policy = entry["collection_policy"]
        policy_counts = counts["by_policy"].setdefault(policy, _empty_counts())
        counts["policies"][policy] = counts["policies"].get(policy, 0) + 1
        lengths = np.linalg.norm(actions, axis=1)
        profile = manifest.get("collection_profile")
        if (profile in {"wide_only", "matched_alternating"}
                and ((policy == "wide" and np.any(lengths < 0.2 - 1e-5))
                     or np.any(lengths > 3.5 + 1e-5))):
            action_range_violations.append(episode_id)
        for bucket in (counts, policy_counts):
            bucket["episodes"] += 1
            bucket["actions"] += len(actions)
            bucket["length_gt_1_8"] += int(np.count_nonzero(lengths > 1.8 + 1e-5))
            bucket["length_gt_3_0"] += int(np.count_nonzero(lengths > 3.0 + 1e-5))
            bucket["length_le_3_5"] += int(np.count_nonzero(lengths <= 3.5 + 1e-5))
        for step, action in enumerate(actions):
            label = classify_transition(
                states[step], action, states[step + 1], entry["wall_layout"],
            )
            field = {
                "solid_contact": "solid_contacts",
                "door_crossing_left_to_right": "door_crossings_left_to_right",
                "door_crossing_right_to_left": "door_crossings_right_to_left",
                "border_contact": "border_contacts",
            }.get(label)
            if field is None:
                continue
            for bucket in (counts, policy_counts):
                bucket[field] += 1
                examples = bucket["examples"].setdefault(label, [])
                if len(examples) < 3:
                    examples.append(dict(
                        episode_id=episode_id, step=step,
                        start=states[step].tolist(), action=action.tolist(),
                        end=states[step + 1].tolist(),
                    ))
    thresholds = dict(solid_contacts=25, doorway_crossings=25, each_direction=5)
    coverage = {}
    for layout in manifest["train_layouts"]:
        key = ",".join(map(str, layout))
        counts = layouts[key]
        left_to_right = counts["door_crossings_left_to_right"]
        right_to_left = counts["door_crossings_right_to_left"]
        coverage[key] = (
            counts["solid_contacts"] >= thresholds["solid_contacts"]
            and left_to_right + right_to_left >= thresholds["doorway_crossings"]
            and min(left_to_right, right_to_left) >= thresholds["each_direction"]
            and counts["by_policy"].get("wide", {}).get("length_gt_3_0", 0) > 0
        )
    if action_range_violations:
        raise ValueError(f"Action lengths outside policy bounds in episodes {action_range_violations[:5]}")
    return dict(
        manifest_sha256=digest, episode_counts=manifest["episode_counts"],
        layouts=layouts, action_hashes=action_hashes, coverage_thresholds=thresholds,
        coverage_by_training_layout=coverage, coverage_pass=all(coverage.values()),
    )


def compare_action_hashes(first: dict, second: dict) -> dict:
    """Compare sequential datasets without retaining the first raw episodes."""
    left, right = first["action_hashes"], second["action_hashes"]
    if set(left) != set(right):
        raise ValueError("Paired action seed sets differ")
    mismatches = [seed for seed in left if left[seed] != right[seed]]
    if mismatches:
        raise ValueError(f"Paired action sequences differ for seeds {mismatches[:5]}")
    return {"matching": len(left), "seeds": sorted(map(int, left))}


def compare_matched_action_hashes(
    first: dict, second: dict, first_manifest: dict, second_manifest: dict,
    shared_seeds=None,
) -> dict:
    """Check paired wide actions and held-out actions; train door actions depend on layout."""
    first_manifest = validate_manifest(first_manifest)
    second_manifest = validate_manifest(second_manifest)
    first_entries = {entry["seed"]: entry for entry in first_manifest["episodes"]}
    second_entries = {entry["seed"]: entry for entry in second_manifest["episodes"]}
    if shared_seeds is None:
        if first_entries.keys() != second_entries.keys():
            raise ValueError("Paired action seed sets differ")
        seeds = first_entries.keys()
    else:
        seeds = set(shared_seeds)
        if not seeds <= first_entries.keys() or not seeds <= second_entries.keys():
            raise ValueError("Requested paired action seeds are missing")
    matching = 0
    layout_dependent_train_door = 0
    for seed in seeds:
        left, right = first_entries[seed], second_entries[seed]
        if left["split"] != right["split"] or left["collection_policy"] != right["collection_policy"]:
            raise ValueError(f"Paired action policy differs for seed {seed}")
        if left["split"] != "train" and left["wall_layout"] != right["wall_layout"]:
            raise ValueError(f"Held-out layout differs for seed {seed}")
        if left["split"] == "train" and left["collection_policy"] == "door":
            layout_dependent_train_door += 1
            continue
        left_hash = first["action_hashes"].get(str(seed))
        right_hash = second["action_hashes"].get(str(seed))
        if not left_hash or not right_hash or left_hash != right_hash:
            raise ValueError(f"Paired action sequences differ for seed {seed}")
        matching += 1
    return {"matching": matching, "layout_dependent_train_door": layout_dependent_train_door}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-coverage", action="store_true")
    args = parser.parse_args(argv)
    result = audit_dataset(args.raw_dir, load_manifest(args.episode_manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"Coverage pass: {result['coverage_pass']}")
    if args.require_coverage and not result["coverage_pass"]:
        raise SystemExit("Training layout coverage is below threshold")


if __name__ == "__main__":
    main()
