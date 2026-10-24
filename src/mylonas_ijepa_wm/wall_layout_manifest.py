"""Deterministic layout and episode assignments for the DotWall experiment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig


TRAIN_LAYOUTS = [[x, y] for x in (24, 32, 40) for y in (18, 30, 42)]
VAL_LAYOUTS = [[28, 24], [36, 36]]
TEST_LAYOUTS = [[26, 22], [26, 38], [38, 22], [38, 38]]
SPLITS = ("train", "val", "test")


def manifest_digest(manifest: dict) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _validate_layout(layout):
    if (not isinstance(layout, list) or len(layout) != 2
            or any(type(value) is not int for value in layout)
            or not 20 <= layout[0] <= 44 or not 10 <= layout[1] <= 54):
        raise ValueError(f"Invalid wall layout: {layout}")
    return tuple(layout)


def validate_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported wall layout manifest schema")
    layouts_by_split = {}
    seen_layouts = set()
    seen_wall_positions = set()
    seen_door_positions = set()
    for split in SPLITS:
        layouts = manifest.get(f"{split}_layouts")
        if not isinstance(layouts, list) or not layouts:
            raise ValueError(f"{split}_layouts must be nonempty")
        values = [_validate_layout(layout) for layout in layouts]
        if len(set(values)) != len(values) or seen_layouts.intersection(values):
            raise ValueError("Wall layouts must be unique and disjoint across splits")
        wall_positions = {wall_x for wall_x, _ in values}
        door_positions = {door_y for _, door_y in values}
        if seen_wall_positions.intersection(wall_positions) or seen_door_positions.intersection(door_positions):
            raise ValueError("Wall and doorway coordinates must be unseen across splits")
        layouts_by_split[split] = set(values)
        seen_layouts.update(values)
        seen_wall_positions.update(wall_positions)
        seen_door_positions.update(door_positions)
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest episodes must be nonempty")
    counts = {split: 0 for split in SPLITS}
    assigned_layouts = {split: set() for split in SPLITS}
    seeds = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("episode_id") != index:
            raise ValueError(f"Manifest episode IDs must be contiguous from zero (index {index})")
        split = entry.get("split")
        if split not in SPLITS:
            raise ValueError(f"Episode {index} has invalid split")
        if _validate_layout(entry.get("wall_layout")) not in layouts_by_split[split]:
            raise ValueError(f"Episode {index} layout does not belong to {split}")
        seed = entry.get("seed")
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError(f"Episode {index} has invalid or repeated seed")
        seeds.add(seed)
        if entry.get("collection_policy") not in {"wide", "door"}:
            raise ValueError(f"Episode {index} has invalid collection policy")
        counts[split] += 1
        assigned_layouts[split].add(tuple(entry["wall_layout"]))
    if any(value == 0 for value in counts.values()):
        raise ValueError("Every split must contain episodes")
    if manifest.get("episode_counts") != counts:
        raise ValueError("Manifest episode counts disagree with assignments")
    if any(assigned_layouts[split] != layouts_by_split[split] for split in SPLITS):
        raise ValueError("Every declared layout must have assigned episodes")
    if manifest.get("steps_per_episode") != 50:
        raise ValueError("Wall layout experiment requires 50 primitive actions per episode")
    if manifest.get("max_training_action_length") != 3.5:
        raise ValueError("Wall layout experiment training action maximum must be 3.5")
    return manifest


def validate_checkpoint_manifest(metadata: dict, manifest: dict) -> None:
    """Reject evaluation with a model trained on different episode assignments."""
    manifest = validate_manifest(manifest)
    if metadata.get("environment") != "wall":
        raise ValueError("Checkpoint is not a Wall model")
    if metadata.get("experiment_manifest_sha256") != manifest_digest(manifest):
        raise ValueError("Checkpoint manifest identity differs from evaluation manifest")
    expected_layouts = {split: manifest[f"{split}_layouts"] for split in SPLITS}
    if metadata.get("layout_splits") != expected_layouts:
        raise ValueError("Checkpoint layout splits differ from evaluation manifest")
    for split in SPLITS:
        expected_ids = [entry["episode_id"] for entry in manifest["episodes"]
                        if entry["split"] == split]
        if metadata.get(f"{split}_ids") != expected_ids:
            raise ValueError(f"Checkpoint {split} IDs differ from evaluation manifest")


def prepare_manifest(
    base_seed: int = 0, train_episodes: int = 2000,
    val_episodes: int = 200, test_episodes: int = 400,
) -> dict:
    counts = dict(train=train_episodes, val=val_episodes, test=test_episodes)
    layouts_by_split = dict(train=TRAIN_LAYOUTS, val=VAL_LAYOUTS, test=TEST_LAYOUTS)
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    for split in SPLITS:
        if type(counts[split]) is not int or counts[split] < len(layouts_by_split[split]):
            raise ValueError(f"{split} needs at least one episode per layout")
    episodes = []
    for split in SPLITS:
        layouts = layouts_by_split[split]
        for local_index in range(counts[split]):
            episode_id = len(episodes)
            layout_index = local_index % len(layouts)
            visit = local_index // len(layouts)
            episodes.append(dict(
                episode_id=episode_id,
                seed=base_seed + episode_id,
                split=split,
                wall_layout=list(layouts[layout_index]),
                collection_policy="wide" if visit % 2 == 0 else "door",
            ))
    manifest = dict(
        schema_version=1,
        train_layouts=[list(layout) for layout in TRAIN_LAYOUTS],
        val_layouts=[list(layout) for layout in VAL_LAYOUTS],
        test_layouts=[list(layout) for layout in TEST_LAYOUTS],
        episode_counts=counts,
        steps_per_episode=50,
        max_training_action_length=3.5,
        episodes=episodes,
    )
    return validate_manifest(manifest)


def load_manifest(path: Path) -> dict:
    return validate_manifest(json.loads(Path(path).read_text()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dino-wm-root", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--val-episodes", type=int, default=200)
    parser.add_argument("--test-episodes", type=int, default=400)
    args = parser.parse_args(argv)
    manifest = prepare_manifest(
        args.base_seed, args.train_episodes, args.val_episodes, args.test_episodes,
    )
    root = args.dino_wm_root.expanduser().resolve()
    if not (root / "env/wall/envs/wall.py").is_file():
        raise FileNotFoundError(f"DINO-WM Wall source not found in {root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "episode_manifest.json"
    config_path = args.output_dir / "environment.json"
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError("Experiment manifest or config already exists")
    config = EnvironmentConfig(
        env="wall",
        wall_layouts=[*TRAIN_LAYOUTS, *VAL_LAYOUTS, *TEST_LAYOUTS],
        dino_wm_root=str(root),
    )
    config_path.write_text(json.dumps({
        **config.portable_metadata(), "dino_wm_root": config.dino_wm_root,
    }, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path} ({manifest_digest(manifest)})")
    print(f"Environment: {config_path}")


if __name__ == "__main__":
    main()
