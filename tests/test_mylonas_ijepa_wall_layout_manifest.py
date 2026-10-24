import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm import generate_episodes
from src.mylonas_ijepa_wm.environment_adapters import StepResult, WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig


def _replace_test_layout(manifest, replacement):
    original = manifest["test_layouts"][0]
    manifest["test_layouts"][0] = replacement
    for episode in manifest["episodes"]:
        if episode["wall_layout"] == original:
            episode["wall_layout"] = replacement


def test_prepare_manifest_balances_disjoint_layouts_and_policies():
    from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest

    manifest = prepare_manifest(base_seed=100, train_episodes=90, val_episodes=20, test_episodes=40)
    assert manifest["schema_version"] == 1
    assert manifest["train_layouts"] == [
        [24, 18], [24, 30], [24, 42], [32, 18], [32, 30],
        [32, 42], [40, 18], [40, 30], [40, 42],
    ]
    assert manifest["val_layouts"] == [[28, 24], [36, 36]]
    assert manifest["test_layouts"] == [[26, 22], [26, 38], [38, 22], [38, 38]]
    entries = manifest["episodes"]
    assert len(entries) == 150
    assert [e["episode_id"] for e in entries] == list(range(150))
    assert [e["seed"] for e in entries] == list(range(100, 250))
    by_split = defaultdict(list)
    for entry in entries:
        by_split[entry["split"]].append(entry)
    assert {key: len(value) for key, value in by_split.items()} == {
        "train": 90, "val": 20, "test": 40,
    }
    for split, expected_per_layout in (("train", 10), ("val", 10), ("test", 10)):
        counts = Counter(tuple(e["wall_layout"]) for e in by_split[split])
        assert set(counts.values()) == {expected_per_layout}
        for layout in counts:
            policies = Counter(
                e["collection_policy"] for e in by_split[split]
                if tuple(e["wall_layout"]) == layout
            )
            assert policies == {"wide": 5, "door": 5}


@pytest.mark.parametrize("mutation", [
    lambda m: m["episodes"][1].update(episode_id=0),
    lambda m: m["episodes"][1].update(episode_id=3),
    lambda m: m["test_layouts"].append([24, 18]),
    lambda m: _replace_test_layout(m, [24, 22]),
    lambda m: _replace_test_layout(m, [26, 18]),
    lambda m: _replace_test_layout(m, [28, 22]),
    lambda m: m["episodes"][0].update(wall_layout=[20, 10]),
])
def test_load_manifest_rejects_corrupt_assignments(tmp_path, mutation):
    from src.mylonas_ijepa_wm.wall_layout_manifest import load_manifest, prepare_manifest

    manifest = prepare_manifest(train_episodes=90, val_episodes=20, test_episodes=40)
    mutation(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_checkpoint_manifest_identity_requires_matching_layouts_and_ids():
    from src.mylonas_ijepa_wm.wall_layout_manifest import (
        manifest_digest, prepare_manifest, validate_checkpoint_manifest,
    )

    manifest = prepare_manifest(train_episodes=90, val_episodes=20, test_episodes=40)
    metadata = {
        "environment": "wall",
        "experiment_manifest_sha256": manifest_digest(manifest),
        "layout_splits": {split: manifest[f"{split}_layouts"] for split in ("train", "val", "test")},
        **{f"{split}_ids": [entry["episode_id"] for entry in manifest["episodes"]
                            if entry["split"] == split] for split in ("train", "val", "test")},
    }
    validate_checkpoint_manifest(metadata, manifest)
    with pytest.raises(ValueError, match="manifest"):
        validate_checkpoint_manifest({**metadata, "experiment_manifest_sha256": "other"}, manifest)
    with pytest.raises(ValueError, match="layout"):
        validate_checkpoint_manifest({**metadata, "layout_splits": None}, manifest)
    with pytest.raises(ValueError, match="IDs"):
        validate_checkpoint_manifest({**metadata, "test_ids": []}, manifest)


def test_wall_adapter_explicit_layout_overrides_seed_selection():
    class FakeEnv:
        def seed(self, value):
            pass

        def generate_random_state(self, seed):
            return np.array([10., 20.]), np.array([50., 20.])

        def set_init_state(self, value):
            pass

        def reset(self):
            return None, torch.tensor([10., 20.])

    adapter = object.__new__(WallAdapter)
    adapter.config = EnvironmentConfig(env="wall", wall_layouts=[[24, 18], [32, 30]])
    adapter.env = FakeEnv()
    adapter._make_environment = lambda wall_x, door_y, seed: setattr(adapter, "layout_made", (wall_x, door_y))
    adapter.render = lambda: np.zeros((224, 224, 3), dtype=np.uint8)
    result = adapter.reset(0, layout=(38, 22))
    assert adapter.layout == (38, 22)
    assert adapter.layout_made == (38, 22)
    assert np.array_equal(result.state, [10., 20.])


def test_manifest_collection_records_assigned_seed_layout_and_digest(tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm.wall_layout_manifest import manifest_digest, prepare_manifest

    manifest = prepare_manifest(train_episodes=90, val_episodes=20, test_episodes=40)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = EnvironmentConfig(env="wall", wall_layouts=[
        *manifest["train_layouts"], *manifest["val_layouts"], *manifest["test_layouts"],
    ])
    config_path = tmp_path / "env.json"
    config_path.write_text(json.dumps(config.portable_metadata()))

    class FakeAdapter:
        action_low = np.array([-1., -1.], dtype=np.float32)
        action_high = np.array([1., 1.], dtype=np.float32)
        control_dt_seconds = None
        goal = np.array([50., 20.], dtype=np.float32)

        def reset(self, seed, layout=None):
            self.layout = tuple(layout)
            self.position = np.array([10., 20.], dtype=np.float32)
            return StepResult(self.position.copy(), np.zeros((224, 224, 3), dtype=np.uint8), False)

        def step(self, action):
            self.position += np.asarray(action, dtype=np.float32)
            return StepResult(self.position.copy(), np.zeros((224, 224, 3), dtype=np.uint8), False)

        def close(self):
            pass

    monkeypatch.setattr(generate_episodes, "create_adapter", lambda config, steps: FakeAdapter())
    output = tmp_path / "raw"
    generate_episodes.main([
        "--env", "wall", "--env-config", str(config_path),
        "--episode-manifest", str(manifest_path), "--output-dir", str(output),
        "--start-index", "9", "--end-index", "10", "--steps-per-episode", "50",
    ])
    episode = torch.load(output / "episode_0009.pt", weights_only=True)
    assert episode["seed"] == manifest["episodes"][9]["seed"]
    assert episode["wall_layout"] == tuple(manifest["episodes"][9]["wall_layout"])
    assert episode["collection_policy"] == "door"
    assert episode["manifest_sha256"] == manifest_digest(manifest)
