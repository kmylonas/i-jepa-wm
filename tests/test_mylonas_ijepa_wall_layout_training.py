import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm import embed_frames, evaluate_rollouts_proprio, wm_proprio
from src.mylonas_ijepa_wm.proprio_vit import ProprioViT
from src.mylonas_ijepa_wm.wall_layout_manifest import manifest_digest, validate_manifest


def _manifest():
    entries = [
        dict(episode_id=0, seed=10, split="train", wall_layout=[32, 30], collection_policy="wide"),
        dict(episode_id=1, seed=11, split="val", wall_layout=[28, 24], collection_policy="door"),
        dict(episode_id=2, seed=12, split="test", wall_layout=[26, 22], collection_policy="wide"),
    ]
    return validate_manifest(dict(
        schema_version=1, train_layouts=[[32, 30]], val_layouts=[[28, 24]],
        test_layouts=[[26, 22]], episode_counts={"train": 1, "val": 1, "test": 1},
        steps_per_episode=50, max_training_action_length=3.5, episodes=entries,
    ))


def _sidecars(directory: Path, manifest: dict):
    digest = manifest_digest(manifest)
    for entry in manifest["episodes"]:
        value = 0.0 if entry["split"] == "train" else 100.0
        torch.save(dict(
            episode_index=entry["episode_id"], seed=entry["seed"],
            wall_layout=tuple(entry["wall_layout"]), manifest_sha256=digest,
            states=torch.full((51, 2), value),
            sampled_states=torch.full((51, 2), value),
            actions=torch.zeros(50, 2),
        ), directory / f"episode_{entry['episode_id']:04d}.pt")


def test_manifest_split_uses_layouts_and_training_only_statistics(tmp_path):
    manifest = _manifest()
    _sidecars(tmp_path, manifest)
    assert wm_proprio.episode_ids_by_split(manifest, tmp_path) == ([0], [1], [2])
    mean, _ = wm_proprio.compute_proprio_statistics(tmp_path, [0], state_dim=2)
    assert mean.tolist() == [0., 0.]


@pytest.mark.parametrize(("field", "value"), [
    ("seed", 99), ("wall_layout", (40, 42)), ("manifest_sha256", "wrong"),
])
def test_manifest_split_rejects_inconsistent_sidecar(tmp_path, field, value):
    manifest = _manifest()
    _sidecars(tmp_path, manifest)
    path = tmp_path / "episode_0001.pt"
    sidecar = torch.load(path, weights_only=True)
    sidecar[field] = value
    torch.save(sidecar, path)
    with pytest.raises(ValueError, match="1"):
        wm_proprio.episode_ids_by_split(manifest, tmp_path)


def test_manifest_identity_is_checked_on_resume():
    saved = {"experiment_manifest_sha256": "old"}
    current = {"experiment_manifest_sha256": "new"}
    with pytest.raises(ValueError, match="manifest"):
        wm_proprio.validate_checkpoint_metadata(saved, current)


def test_embedding_sidecar_retains_seed_and_manifest_identity(tmp_path):
    class Encoder(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(len(images), 1, 1, device=images.device)

    episode = dict(
        frames=torch.zeros(6, 224, 224, 3, dtype=torch.uint8),
        actions=torch.zeros(5, 2), states=torch.zeros(6, 2),
        desired_goal=torch.zeros(2), environment="wall",
        wall_layout=(32, 30), seed=71, manifest_sha256="manifest-id",
    )
    embed_frames.process_episode(
        episode, Encoder(), torch.device("cpu"), "episode_0000", tmp_path,
        environment="wall", actions_per_model_step=5,
        embedding_frame_stride=1, encoder_batch_size=3,
    )
    sidecar = torch.load(tmp_path / "trajectories/episode_0000.pt", weights_only=True)
    assert sidecar["seed"] == 71
    assert sidecar["manifest_sha256"] == "manifest-id"


def test_embedding_resume_rejects_cached_episode_from_another_manifest():
    raw = {"seed": 71, "wall_layout": (32, 30), "manifest_sha256": "new"}
    cached = {"seed": 71, "wall_layout": (32, 30), "manifest_sha256": "old"}
    with pytest.raises(ValueError, match="manifest"):
        embed_frames.validate_cached_episode_identity(raw, cached, "episode_0000")


def test_rollout_helpers_group_layouts_and_contact_windows(tmp_path):
    manifest = _manifest()
    _sidecars(tmp_path, manifest)
    groups = evaluate_rollouts_proprio.group_episode_ids_by_wall_layout([0, 1, 2], tmp_path)
    assert groups == {"32,30": [0], "28,24": [1], "26,22": [2]}

    path = tmp_path / "episode_0000.pt"
    sidecar = torch.load(path, weights_only=True)
    states = np.repeat(np.array([[10., 20.]], dtype=np.float32), 51, axis=0)
    actions = np.zeros((50, 2), dtype=np.float32)
    states[2] = [28., 20.]
    states[3] = [29., 20.]
    actions[2] = [2., 0.]
    states[15] = [29., 30.]
    states[16] = [33., 30.]
    actions[15] = [2., 0.]
    sidecar["states"] = torch.tensor(states)
    sidecar["actions"] = torch.tensor(actions)
    torch.save(sidecar, path)
    dataset = SimpleNamespace(
        samples=[(0, 0), (0, 10), (0, 15), (0, 30)], num_hist=1,
        rollout_steps=2, model_frame_steps={0: 5}, action_group_sizes={0: 5},
    )
    subsets = evaluate_rollouts_proprio.transition_subset_indices(dataset, tmp_path)
    assert subsets == {"solid_contact": [0], "door_crossing": [2]}


def test_rollout_layout_group_rejects_sidecar_from_another_manifest(tmp_path):
    manifest = _manifest()
    _sidecars(tmp_path, manifest)
    with pytest.raises(ValueError, match="manifest"):
        evaluate_rollouts_proprio.group_episode_ids_by_wall_layout(
            [2], tmp_path, expected_manifest_sha256="other",
            allowed_layouts=manifest["test_layouts"],
        )
    with pytest.raises(ValueError, match="layout"):
        evaluate_rollouts_proprio.group_episode_ids_by_wall_layout(
            [2], tmp_path, expected_manifest_sha256=manifest_digest(manifest),
            allowed_layouts=manifest["val_layouts"],
        )


def test_wall_rollout_cli_uses_two_state_components_for_full_and_layout_datasets(tmp_path):
    latent_dir = tmp_path / "latents"
    trajectory_dir = tmp_path / "trajectories"
    latent_dir.mkdir()
    trajectory_dir.mkdir()
    np.save(latent_dir / "episode_0001.npy", np.zeros((6, 2, 3), dtype=np.float16))
    torch.save(dict(
        wall_layout=(28, 24), manifest_sha256="wall-manifest",
        sampled_states=torch.zeros(6, 2), states=torch.zeros(6, 2),
        actions=torch.zeros(5, 2), macro_actions=torch.zeros(1, 10),
        actions_per_model_step=5, embedding_frame_stride=1,
    ), trajectory_dir / "episode_0001.pt")
    configuration = dict(
        num_patches=2, num_hist=1, num_t_blocks=1, ijepa_dim=3,
        action_dim=10, action_embed_dim=4, proprio_dim=2,
        proprio_embed_dim=4, embed_dim=8, num_heads=2,
        mlp_dim=16, dropout=0.0,
    )
    model = ProprioViT(**configuration)
    checkpoint = tmp_path / "best.pt"
    torch.save(dict(
        model=model.state_dict(), epoch=0,
        metadata=dict(
            model_type="proprio_v1", environment="wall", camera="topdown",
            proprio_fields=("x", "y"), model_configuration=configuration,
            val_ids=[1], layout_splits={"val": [[28, 24]]},
            experiment_manifest_sha256="wall-manifest",
            normalize_actions=False, state_mean=torch.zeros(2),
            state_std=torch.ones(2), proprio_loss_weight=1.0,
        ),
    ), checkpoint)
    output = tmp_path / "metrics.json"

    evaluate_rollouts_proprio.main([
        "--checkpoint", str(checkpoint),
        "--latent-dir", str(latent_dir),
        "--trajectory-dir", str(trajectory_dir),
        "--split", "val", "--horizon", "1", "--by-wall-layout",
        "--batch-size", "1", "--device", "cpu", "--precision", "fp32",
        "--output", str(output),
    ])

    metrics = json.loads(output.read_text())
    assert metrics["split"] == "val"
    assert metrics["per_horizon"][0]["horizon"] == 1
    assert metrics["by_wall_layout"]["28,24"]["num_episodes"] == 1
