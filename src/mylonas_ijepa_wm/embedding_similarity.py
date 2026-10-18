"""Measure how frozen patch embeddings change with PointMaze ball displacement."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_EDGES = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


def distance_bin(distance, edges):
    """Return the index of a (lower, upper] distance bin."""
    if not 0 < distance <= edges[-1]:
        return None
    return int(np.searchsorted(edges, distance, side="left") - 1)


def mean_patch_cosine(first, second):
    """Average cosine similarity of patch tokens at matching grid positions."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Expected matching [patch, channel] token arrays")
    numerator = np.sum(first * second, axis=-1)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    if np.any(denominator == 0):
        raise ValueError("Cannot calculate cosine similarity of a zero token")
    return float(np.mean(numerator / denominator))


def sample_episode_pairs(positions, edges, rng):
    """Select at most one observed frame pair in each displacement bin."""
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Expected [frame, xy] positions")
    candidates = {bin_index: [] for bin_index in range(len(edges) - 1)}
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            bin_index = distance_bin(distance, edges)
            if bin_index is not None:
                candidates[bin_index].append((first, second, distance))
    return {
        bin_index: pairs[int(rng.integers(len(pairs)))]
        for bin_index, pairs in candidates.items()
        if pairs
    }


def load_aligned_positions(cache_dirs, episode_id):
    """Read position labels and require the same trajectory for both cameras."""
    reference_positions = None
    reference_indices = None
    for camera, cache_dir in cache_dirs.items():
        path = Path(cache_dir) / "trajectories" / f"episode_{episode_id:04d}.pt"
        trajectory = torch.load(path, map_location="cpu", weights_only=True)
        positions = trajectory["sampled_states"][:, :2].numpy()
        indices = trajectory["frame_indices"].numpy()
        if reference_positions is None:
            reference_positions = positions
            reference_indices = indices
        elif not np.array_equal(indices, reference_indices):
            raise ValueError(f"Episode {episode_id} frame-index mismatch for {camera}")
        elif not np.allclose(positions, reference_positions, atol=1e-5, rtol=0):
            raise ValueError(f"Episode {episode_id} position mismatch for {camera}")
    return reference_positions


def available_episode_ids(cache_dir):
    cache_dir = Path(cache_dir)
    latents = {
        int(path.stem.removeprefix("episode_"))
        for path in (cache_dir / "latents").glob("episode_*.npy")
    }
    trajectories = {
        int(path.stem.removeprefix("episode_"))
        for path in (cache_dir / "trajectories").glob("episode_*.pt")
    }
    return latents & trajectories


def plot_results(rows, summary, output_path, edges):
    figure, ax = plt.subplots(figsize=(10, 5))
    cameras = list(dict.fromkeys(row["camera"] for row in rows))
    colors = {camera: plt.get_cmap("tab10")(i) for i, camera in enumerate(cameras)}
    rng = np.random.default_rng(0)
    for camera_index, camera in enumerate(cameras):
        offset = (camera_index - (len(cameras) - 1) / 2) * 0.18
        camera_rows = [row for row in rows if row["camera"] == camera]
        for row in camera_rows:
            x = row["bin_index"] + offset + rng.uniform(-0.035, 0.035)
            ax.scatter(x, row["mean_patch_cosine"], s=12, alpha=0.15,
                       color=colors[camera], linewidths=0)
        camera_summary = [row for row in summary if row["camera"] == camera and row["count"]]
        if camera_summary:
            x = np.array([row["bin_index"] + offset for row in camera_summary])
            medians = np.array([row["median_cosine"] for row in camera_summary])
            lower = medians - np.array([row["q25_cosine"] for row in camera_summary])
            upper = np.array([row["q75_cosine"] for row in camera_summary]) - medians
            ax.errorbar(x, medians, yerr=[lower, upper], marker="o", capsize=3,
                        color=colors[camera], linewidth=1.5, label=camera)
    labels = [f"({edges[i]:g}, {edges[i + 1]:g}]" for i in range(len(edges) - 1)]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("Ball displacement (maze units)")
    ax.set_ylabel("Mean same-grid-patch cosine similarity")
    ax.set_title("I-JEPA sensitivity to ball movement")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summarize_rows(rows, cameras, edges):
    summary = []
    for camera in cameras:
        for bin_index in range(len(edges) - 1):
            values = np.array([
                row["mean_patch_cosine"] for row in rows
                if row["camera"] == camera and row["bin_index"] == bin_index
            ])
            summary.append({
                "camera": camera,
                "bin_index": bin_index,
                "lower_exclusive": edges[bin_index],
                "upper_inclusive": edges[bin_index + 1],
                "count": len(values),
                "median_cosine": float(np.median(values)) if len(values) else "",
                "q25_cosine": float(np.quantile(values, 0.25)) if len(values) else "",
                "q75_cosine": float(np.quantile(values, 0.75)) if len(values) else "",
            })
    return summary


def save_results(rows, summary, metadata, output_dir, edges):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "pairs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["bin_index"], row["episode_id"], row["camera"])))
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(rows, summary, output_dir / "similarity.png", edges)


def run_experiment(
    cache_dirs,
    output_dir,
    max_episodes=100,
    pairs_per_bin=30,
    seed=42,
    edges=DEFAULT_EDGES,
):
    """Sample matched observed frame pairs and write data, summaries, and plot."""
    if not cache_dirs:
        raise ValueError("Provide at least one camera cache")
    if max_episodes < 1 or pairs_per_bin < 1:
        raise ValueError("max_episodes and pairs_per_bin must be positive")
    if len(edges) < 2 or edges[0] != 0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("edges must start at zero and increase")

    cache_dirs = {camera: Path(path) for camera, path in cache_dirs.items()}
    episode_sets = [available_episode_ids(path) for path in cache_dirs.values()]
    common_ids = sorted(set.intersection(*episode_sets))
    if not common_ids:
        raise ValueError("No episodes with both latents and trajectories in every camera cache")

    rng = np.random.default_rng(seed)
    selected_ids = sorted(int(i) for i in rng.choice(
        common_ids, size=min(max_episodes, len(common_ids)), replace=False
    ))
    candidates = {bin_index: [] for bin_index in range(len(edges) - 1)}
    frame_counts = {}
    for episode_id in selected_ids:
        positions = load_aligned_positions(cache_dirs, episode_id)
        frame_counts[episode_id] = len(positions)
        episode_pairs = sample_episode_pairs(positions, edges, rng)
        for bin_index, (first, second, distance) in episode_pairs.items():
            candidates[bin_index].append((episode_id, first, second, distance))

    chosen = []
    for bin_index, pairs in candidates.items():
        if pairs:
            indices = rng.choice(len(pairs), size=min(pairs_per_bin, len(pairs)), replace=False)
            chosen.extend((bin_index, *pairs[int(i)]) for i in indices)
    if not chosen:
        raise ValueError("No frame pairs have displacement inside the requested bins")

    by_episode = {}
    for bin_index, episode_id, first, second, distance in chosen:
        by_episode.setdefault(episode_id, []).append((bin_index, first, second, distance))

    rows = []
    for episode_id, pairs in sorted(by_episode.items()):
        for camera, cache_dir in cache_dirs.items():
            latent_path = cache_dir / "latents" / f"episode_{episode_id:04d}.npy"
            latents = np.load(latent_path, mmap_mode="r", allow_pickle=False)
            if latents.ndim != 3 or len(latents) != frame_counts[episode_id] or latents.shape[1] != 256:
                raise ValueError(f"Unexpected latent shape for {latent_path}: {latents.shape}")
            for bin_index, first, second, distance in pairs:
                first_tokens = np.asarray(latents[first], dtype=np.float32)
                second_tokens = np.asarray(latents[second], dtype=np.float32)
                rows.append({
                    "camera": camera,
                    "episode_id": episode_id,
                    "frame_a": first,
                    "frame_b": second,
                    "distance": distance,
                    "bin_index": bin_index,
                    "mean_patch_cosine": mean_patch_cosine(first_tokens, second_tokens),
                    "mean_patch_mse": float(np.mean((first_tokens - second_tokens) ** 2)),
                })

    summary = summarize_rows(rows, cache_dirs, edges)
    metadata = {
        "cache_dirs": {camera: str(path.resolve()) for camera, path in cache_dirs.items()},
        "seed": seed,
        "max_episodes": max_episodes,
        "selected_episode_ids": selected_ids,
        "pairs_per_bin_limit": pairs_per_bin,
        "bin_edges": list(edges),
        "sampled_pairs": len(chosen),
        "method": "one within-episode frame pair per distance bin, per selected episode",
    }
    save_results(rows, summary, metadata, output_dir, edges)
    return summary


def load_saved_run(result_dir):
    result_dir = Path(result_dir)
    metadata = json.loads((result_dir / "metadata.json").read_text())
    cameras = list(metadata["cache_dirs"])
    if len(cameras) != 1:
        raise ValueError(f"Expected a single-camera result in {result_dir}")
    camera = cameras[0]
    with (result_dir / "pairs.csv").open(newline="") as file:
        saved_rows = list(csv.DictReader(file))
    if not saved_rows:
        raise ValueError(f"No saved pairs in {result_dir}")
    rows = []
    for row in saved_rows:
        if row["camera"] != camera:
            raise ValueError(f"Unexpected camera in {result_dir}/pairs.csv")
        rows.append({
            "camera": camera,
            "episode_id": int(row["episode_id"]),
            "frame_a": int(row["frame_a"]),
            "frame_b": int(row["frame_b"]),
            "distance": float(row["distance"]),
            "bin_index": int(row["bin_index"]),
            "mean_patch_cosine": float(row["mean_patch_cosine"]),
            "mean_patch_mse": float(row["mean_patch_mse"]),
        })
    return metadata, camera, rows


def combine_results(result_dirs, output_dir):
    """Combine two saved single-camera runs without reading either latent cache."""
    if len(result_dirs) != 2:
        raise ValueError("Provide exactly two single-camera result directories")
    if any(Path(output_dir).resolve() == Path(path).resolve() for path in result_dirs):
        raise ValueError("Combined output must be separate from input results")
    first_metadata, first_camera, first_rows = load_saved_run(result_dirs[0])
    second_metadata, second_camera, second_rows = load_saved_run(result_dirs[1])
    if {first_camera, second_camera} != {"topdown", "angled"}:
        raise ValueError("Results must contain one topdown run and one angled run")
    settings = ("seed", "max_episodes", "selected_episode_ids", "pairs_per_bin_limit", "bin_edges")
    for setting in settings:
        if first_metadata[setting] != second_metadata[setting]:
            raise ValueError(f"Run setting differs: {setting}; rerun both with matching settings and episodes")

    def pair_key(row):
        return (row["episode_id"], row["frame_a"], row["frame_b"], row["bin_index"])

    first_pairs = {pair_key(row): row for row in first_rows}
    second_pairs = {pair_key(row): row for row in second_rows}
    if (len(first_pairs) != len(first_rows) or len(second_pairs) != len(second_rows)
            or first_pairs.keys() != second_pairs.keys()):
        raise ValueError("Saved pair selection differs between cameras")
    for key in first_pairs:
        if abs(first_pairs[key]["distance"] - second_pairs[key]["distance"]) > 1e-5:
            raise ValueError(f"Saved pair position distance differs for {key}")

    edges = tuple(first_metadata["bin_edges"])
    rows = first_rows + second_rows
    summary = summarize_rows(rows, (first_camera, second_camera), edges)
    metadata = {
        "source_results": [str(Path(path).resolve()) for path in result_dirs],
        "cameras": [first_camera, second_camera],
        "seed": first_metadata["seed"],
        "max_episodes": first_metadata["max_episodes"],
        "selected_episode_ids": first_metadata["selected_episode_ids"],
        "pairs_per_bin_limit": first_metadata["pairs_per_bin_limit"],
        "bin_edges": list(edges),
        "sampled_pairs": len(first_pairs),
        "method": "combined matched single-camera result files; no latent caches accessed",
    }
    save_results(rows, summary, metadata, output_dir, edges)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topdown-cache", type=Path, help="Directory containing latents/ and trajectories/")
    parser.add_argument("--angled-cache", type=Path, help="Directory containing latents/ and trajectories/")
    parser.add_argument("--combine-results", nargs=2, type=Path,
                        metavar=("TOPDOWN_RESULTS", "ANGLED_RESULTS"),
                        help="Combine two saved single-camera result directories without caches")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--pairs-per-bin", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.combine_results and (args.topdown_cache or args.angled_cache):
        parser.error("--combine-results cannot be used with camera cache arguments")
    if not args.combine_results and not (args.topdown_cache or args.angled_cache):
        parser.error("provide a camera cache or --combine-results")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.combine_results:
        summary = combine_results(args.combine_results, args.output_dir)
    else:
        cache_dirs = {
            camera: path for camera, path in (
                ("topdown", args.topdown_cache), ("angled", args.angled_cache)
            ) if path is not None
        }
        summary = run_experiment(
            cache_dirs, args.output_dir, args.max_episodes, args.pairs_per_bin, args.seed
        )
    for row in summary:
        print(f"{row['camera']:7s} ({row['lower_exclusive']:g}, {row['upper_inclusive']:g}]: "
              f"n={row['count']:3d} median_cosine={row['median_cosine']}")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
