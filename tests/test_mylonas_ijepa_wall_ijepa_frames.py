import pytest
import torch

from src.mylonas_ijepa_wm.wall_layout_manifest import manifest_digest, prepare_manifest


@pytest.fixture
def wall_raw(tmp_path):
    manifest = prepare_manifest(train_episodes=9, val_episodes=2, test_episodes=4)
    raw = tmp_path / "raw"
    raw.mkdir()
    frames = torch.arange(51, dtype=torch.uint8).view(51, 1, 1, 1).expand(51, 224, 224, 3).contiguous()
    for entry in manifest["episodes"]:
        episode = {
            "frames": frames,
            "episode_index": entry["episode_id"],
            "seed": entry["seed"],
            "wall_layout": tuple(entry["wall_layout"]),
            "collection_policy": entry["collection_policy"],
            "manifest_sha256": manifest_digest(manifest),
            "environment": "wall",
        }
        torch.save(episode, raw / f"episode_{entry['episode_id']:04d}.pt")
    return raw, manifest


def test_index_uses_manifest_splits_and_fifth_frames(wall_raw):
    from src.datasets.wall_frames import WallFrames, build_frame_index
    from src.mylonas_ijepa_wm.embed_frames import preprocess_frames

    raw, manifest = wall_raw
    index = build_frame_index(raw, manifest, frame_stride=5)
    assert index["manifest_sha256"] == manifest_digest(manifest)
    assert [len(index["splits"][s]) for s in ("train", "val", "test")] == [9, 2, 4]
    train = WallFrames(raw, index, "train")
    val = WallFrames(raw, index, "val")
    assert len(train) == 99
    assert len(val) == 22
    image, target = train[1]
    assert target == 0
    expected = preprocess_frames(torch.full((1, 224, 224, 3), 5, dtype=torch.uint8))[0]
    torch.testing.assert_close(image, expected)
    assert train[10][0][0, 0, 0] == preprocess_frames(
        torch.full((1, 224, 224, 3), 50, dtype=torch.uint8)
    )[0, 0, 0, 0]


def test_index_rejects_missing_and_extra_episode_files(wall_raw):
    from src.datasets.wall_frames import build_frame_index

    raw, manifest = wall_raw
    (raw / "episode_0000.pt").unlink()
    with pytest.raises(ValueError, match="missing"):
        build_frame_index(raw, manifest)
    torch.save({}, raw / "episode_9999.pt")
    with pytest.raises(ValueError, match="missing.*extra"):
        build_frame_index(raw, manifest)


@pytest.mark.parametrize("field,value", [
    ("manifest_sha256", "other"),
    ("seed", -1),
    ("wall_layout", (1, 2)),
    ("environment", "pointmaze"),
    ("frames", torch.zeros((50, 224, 224, 3), dtype=torch.uint8)),
    ("frames", torch.zeros((51, 223, 224, 3), dtype=torch.uint8)),
])
def test_index_rejects_episode_mismatch(wall_raw, field, value):
    from src.datasets.wall_frames import build_frame_index

    raw, manifest = wall_raw
    path = raw / "episode_0000.pt"
    episode = torch.load(path, weights_only=True)
    episode[field] = value
    torch.save(episode, path)
    with pytest.raises(ValueError, match="episode 0"):
        build_frame_index(raw, manifest)


def test_index_rejects_invalid_frame_stride(wall_raw):
    from src.datasets.wall_frames import build_frame_index

    raw, manifest = wall_raw
    with pytest.raises(ValueError, match="stride"):
        build_frame_index(raw, manifest, frame_stride=0)
