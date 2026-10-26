"""Manifest-validated rendered Wall frames for the original I-JEPA trainer."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.mylonas_ijepa_wm.embed_frames import preprocess_frames
from src.mylonas_ijepa_wm.wall_layout_manifest import manifest_digest, validate_manifest


def _episode_path(raw_dir: Path, episode_id: int) -> Path:
    return raw_dir / f"episode_{episode_id:04d}.pt"


def build_frame_index(raw_dir: Path, manifest: dict, frame_stride: int = 5) -> dict:
    """Validate a complete raw collection and index every selected frame."""
    manifest = validate_manifest(manifest)
    if type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("frame stride must be a positive integer")
    raw_dir = Path(raw_dir)
    expected = {f"episode_{entry['episode_id']:04d}.pt" for entry in manifest["episodes"]}
    actual = {path.name for path in raw_dir.glob("episode_*.pt")}
    missing, extra = expected - actual, actual - expected
    if missing or extra:
        raise ValueError(
            f"Raw episode files disagree with manifest: missing={sorted(missing)[:3]}, "
            f"extra={sorted(extra)[:3]}"
        )

    digest = manifest_digest(manifest)
    splits = {split: [] for split in ("train", "val", "test")}
    for entry in manifest["episodes"]:
        episode_id = entry["episode_id"]
        episode = torch.load(
            _episode_path(raw_dir, episode_id), map_location="cpu",
            weights_only=True, mmap=True,
        )
        frames = episode.get("frames")
        valid_metadata = (
            episode.get("environment") == "wall"
            and episode.get("episode_index") == episode_id
            and episode.get("seed") == entry["seed"]
            and tuple(episode.get("wall_layout") or ()) == tuple(entry["wall_layout"])
            and episode.get("collection_policy") == entry["collection_policy"]
            and episode.get("manifest_sha256") == digest
        )
        valid_frames = (
            isinstance(frames, torch.Tensor)
            and frames.dtype == torch.uint8
            and tuple(frames.shape) == (manifest["steps_per_episode"] + 1, 224, 224, 3)
        )
        if not valid_metadata or not valid_frames:
            raise ValueError(f"Raw episode {episode_id} disagrees with manifest or image format")
        splits[entry["split"]].append({
            "episode_id": episode_id,
            "seed": entry["seed"],
            "wall_layout": entry["wall_layout"],
            "frame_count": len(frames),
        })
    return {
        "manifest_sha256": digest,
        "frame_stride": frame_stride,
        "splits": splits,
        "sample_counts": {
            split: sum(len(range(0, item["frame_count"], frame_stride)) for item in records)
            for split, records in splits.items()
        },
    }


class WallFrames(Dataset):
    """Map one selected frame to an image and dummy ImageNet-style label."""

    def __init__(self, raw_dir: Path, index: dict, split: str):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown Wall split: {split}")
        self.raw_dir = Path(raw_dir)
        self.samples = [
            (item["episode_id"], frame)
            for item in index["splits"][split]
            for frame in range(0, item["frame_count"], index["frame_stride"])
        ]
        self._cached_id = None
        self._cached_frames = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_id, frame_index = self.samples[index]
        if episode_id != self._cached_id:
            episode = torch.load(
                _episode_path(self.raw_dir, episode_id), map_location="cpu",
                weights_only=True, mmap=True,
            )
            self._cached_frames = episode["frames"]
            self._cached_id = episode_id
        frame = self._cached_frames[frame_index].unsqueeze(0)
        return preprocess_frames(frame)[0], 0
