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


def test_matched_manifests_use_separate_training_layouts_and_wide_policy():
    from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest, validate_manifest

    fixed = prepare_manifest(
        train_episodes=18, val_episodes=4, test_episodes=8,
        train_layout_mode="fixed", collection_profile="wide_only",
    )
    multi = prepare_manifest(
        train_episodes=18, val_episodes=4, test_episodes=8,
        train_layout_mode="multi", collection_profile="wide_only",
    )
    assert fixed["train_layouts"] == [[32, 30]]
    assert len(multi["train_layouts"]) == 9
    assert fixed["max_training_action_length"] == multi["max_training_action_length"] == 3.5
    assert all(e["collection_policy"] == "wide" for m in (fixed, multi) for e in m["episodes"])
    for split in ("val", "test"):
        assert [e for e in fixed["episodes"] if e["split"] == split] == [
            e for e in multi["episodes"] if e["split"] == split
        ]
    assert [e["seed"] for e in fixed["episodes"]] == [e["seed"] for e in multi["episodes"]]
    fixed["episodes"][0]["collection_policy"] = "door"
    with pytest.raises(ValueError, match="wide_only"):
        validate_manifest(fixed)


def test_matched_alternating_manifests_pair_policy_by_seed_and_cover_each_layout():
    from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest, validate_manifest

    fixed, multi = [prepare_manifest(
        train_episodes=90, val_episodes=20, test_episodes=40,
        train_layout_mode=mode, collection_profile="matched_alternating",
    ) for mode in ("fixed", "multi")]
    assert fixed["train_layouts"] == [[32, 30]]
    assert len(multi["train_layouts"]) == 9
    assert [e["collection_policy"] for e in fixed["episodes"]] == [
        e["collection_policy"] for e in multi["episodes"]]
    for manifest in (fixed, multi):
        for layout in manifest["train_layouts"]:
            entries = [e for e in manifest["episodes"] if e["split"] == "train"
                       and e["wall_layout"] == layout]
            assert set(e["collection_policy"] for e in entries) == {"wide", "door"}
            assert all(a["collection_policy"] != b["collection_policy"]
                       for a, b in zip(entries, entries[1:]))
    assert [e for e in fixed["episodes"] if e["split"] != "train"] == [
        e for e in multi["episodes"] if e["split"] != "train"]
    fixed["episodes"][0]["collection_policy"] = "door"
    with pytest.raises(ValueError, match="matched_alternating"):
        validate_manifest(fixed)


def test_training_layout_validation_uses_new_episodes_and_preserves_unseen_test():
    from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest, validate_manifest

    fixed, multi = [prepare_manifest(
        train_episodes=90, val_episodes=36, test_episodes=40,
        train_layout_mode=mode, collection_profile="matched_alternating",
        validation_layout_mode="training",
    ) for mode in ("fixed", "multi")]
    for manifest in (fixed, multi):
        assert manifest["val_layouts"] == manifest["train_layouts"]
        assert manifest["validation_layout_mode"] == "training"
        train = [e for e in manifest["episodes"] if e["split"] == "train"]
        val = [e for e in manifest["episodes"] if e["split"] == "val"]
        assert {e["seed"] for e in train}.isdisjoint(e["seed"] for e in val)
        assert {tuple(e["wall_layout"]) for e in val} == {
            tuple(e["wall_layout"]) for e in train}
        for layout in manifest["val_layouts"]:
            policies = [e["collection_policy"] for e in val if e["wall_layout"] == layout]
            assert set(policies) == {"wide", "door"}
            assert all(a != b for a, b in zip(policies, policies[1:]))
        assert set(map(tuple, manifest["test_layouts"])).isdisjoint(
            map(tuple, manifest["train_layouts"]))
        validate_manifest(manifest)
    assert fixed["val_layouts"] == [[32, 30]]
    assert len(multi["val_layouts"]) == 9
    assert [e for e in fixed["episodes"] if e["split"] == "test"] == [
        e for e in multi["episodes"] if e["split"] == "test"]
    assert [e["collection_policy"] for e in fixed["episodes"]] == [
        e["collection_policy"] for e in multi["episodes"]]


def test_training_layout_validation_rejects_unseen_validation_or_seen_test():
    import copy
    from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest, validate_manifest

    manifest = prepare_manifest(train_episodes=18, val_episodes=18, test_episodes=8,
                                train_layout_mode="fixed", validation_layout_mode="training")
    bad_val = copy.deepcopy(manifest)
    bad_val["val_layouts"] = [[28, 24]]
    with pytest.raises(ValueError, match="Validation layouts"):
        validate_manifest(bad_val)
    bad_test = copy.deepcopy(manifest)
    _replace_test_layout(bad_test, [32, 22])
    with pytest.raises(ValueError, match="unseen"):
        validate_manifest(bad_test)


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
