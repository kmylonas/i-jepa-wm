import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.generate_episodes import sample_dino_wall_actions
from src.mylonas_ijepa_wm.wall_layout_manifest import manifest_digest, validate_manifest


def test_wide_action_policy_exposes_long_lengths_without_changing_dino_policy():
    from src.mylonas_ijepa_wm.generate_episodes import sample_wide_wall_actions

    wide = sample_wide_wall_actions(np.random.default_rng(42), 2000)
    lengths = np.linalg.norm(wide, axis=1)
    assert lengths.min() >= 0.19999
    assert lengths.max() <= 3.50001
    assert (lengths > 3.0).sum() > 200
    headings = np.arctan2(wide[:, 1], wide[:, 0])
    assert np.cos(np.diff(headings)).mean() > 0.7
    old = sample_dino_wall_actions(np.random.default_rng(42), 2000)
    assert np.linalg.norm(old, axis=1).max() <= 1.80001


@pytest.mark.parametrize(("start", "action", "end", "expected"), [
    ([28, 20], [2, 0], [29, 20], "solid_contact"),
    ([29, 30], [2, 0], [33, 30], "door_crossing_left_to_right"),
    ([35, 30], [-2, 0], [31, 30], "door_crossing_right_to_left"),
    ([59, 20], [2, 0], [59.5, 20], "border_contact"),
    ([10, 20], [1, 0], [12, 20], "other"),
])
def test_classifier_distinguishes_wall_door_and_border(start, action, end, expected):
    from src.mylonas_ijepa_wm.audit_wall_dataset import classify_transition

    assert classify_transition(start, action, end, (32, 30)) == expected


def _small_manifest():
    layouts = {"train": [32, 30], "val": [28, 24], "test": [26, 22]}
    entries = [dict(
        episode_id=index, seed=100 + index, split=split,
        wall_layout=layout, collection_policy="wide",
    ) for index, (split, layout) in enumerate(layouts.items())]
    return validate_manifest(dict(
        schema_version=1,
        train_layouts=[layouts["train"]],
        val_layouts=[layouts["val"]],
        test_layouts=[layouts["test"]],
        episode_counts={split: 1 for split in layouts},
        steps_per_episode=50,
        max_training_action_length=3.5,
        episodes=entries,
    ))


def _write_episode(path: Path, entry: dict, digest: str):
    states = np.repeat(np.array([[10., 20.]], dtype=np.float32), 51, axis=0)
    actions = np.zeros((50, 2), dtype=np.float32)
    if entry["split"] == "train":
        states[0] = [28., 20.]
        states[1:] = [29., 20.]
        states[2:] = [29., 26.4]
        actions[0] = [2., 0.]
        actions[1] = [0., 3.2]
    elif entry["split"] == "val":
        states[0] = [25., 24.]
        states[1:] = [29., 24.]
        actions[0] = [2., 0.]
    else:
        states[0] = [29., 22.]
        states[1:] = [25., 22.]
        actions[0] = [-2., 0.]
    torch.save({
        "episode_index": entry["episode_id"], "seed": entry["seed"],
        "wall_layout": tuple(entry["wall_layout"]),
        "manifest_sha256": digest,
        "collection_policy": entry["collection_policy"],
        "states": torch.tensor(states), "actions": torch.tensor(actions),
    }, path)


def test_audit_reports_coverage_by_layout_and_rejects_metadata_mismatch(tmp_path):
    from src.mylonas_ijepa_wm.audit_wall_dataset import audit_dataset

    manifest = _small_manifest()
    digest = manifest_digest(manifest)
    for entry in manifest["episodes"]:
        _write_episode(tmp_path / f"episode_{entry['episode_id']:04d}.pt", entry, digest)
    result = audit_dataset(tmp_path, manifest)
    train = result["layouts"]["32,30"]
    assert train["episodes"] == 1
    assert train["solid_contacts"] == 1
    assert train["length_gt_3_0"] == 1
    assert train["by_policy"]["wide"]["solid_contacts"] == 1
    assert train["by_policy"]["wide"]["length_gt_3_0"] == 1
    assert result["layouts"]["28,24"]["door_crossings_left_to_right"] == 1
    assert result["layouts"]["26,22"]["door_crossings_right_to_left"] == 1
    assert not result["coverage_pass"]

    bad = copy.deepcopy(manifest)
    bad["episodes"][0]["wall_layout"] = [24, 18]
    with pytest.raises(ValueError):
        audit_dataset(tmp_path, bad)
