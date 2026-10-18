import tempfile
import unittest
from pathlib import Path
import csv
import shutil

import numpy as np
import torch

from src.mylonas_ijepa_wm import embedding_similarity as similarity


class EmbeddingSimilarityTest(unittest.TestCase):
    def test_mean_patch_cosine_weights_grid_cells_equally(self):
        first = np.array([[100.0, 0.0], [0.0, 1.0]])
        second = np.array([[100.0, 0.0], [0.0, -1.0]])

        self.assertAlmostEqual(similarity.mean_patch_cosine(first, second), 0.0)

    def test_distance_bins_include_upper_endpoint_and_exclude_zero(self):
        edges = (0.0, 0.1, 0.2, 0.4)

        self.assertIsNone(similarity.distance_bin(0.0, edges))
        self.assertEqual(similarity.distance_bin(0.1, edges), 0)
        self.assertEqual(similarity.distance_bin(0.2, edges), 1)
        self.assertEqual(similarity.distance_bin(0.4, edges), 2)
        self.assertIsNone(similarity.distance_bin(0.41, edges))

    def test_sampling_uses_real_positions_with_one_pair_per_bin(self):
        positions = np.array(
            [[0.0, 0.0], [0.05, 0.0], [0.15, 0.0], [0.35, 0.0]],
            dtype=np.float32,
        )
        pairs = similarity.sample_episode_pairs(
            positions,
            edges=(0.0, 0.1, 0.2, 0.4),
            rng=np.random.default_rng(1),
        )

        self.assertEqual(set(pairs), {0, 1, 2})
        for bin_index, (first, second, distance) in pairs.items():
            self.assertLess(first, second)
            self.assertEqual(similarity.distance_bin(distance, (0.0, 0.1, 0.2, 0.4)), bin_index)
            self.assertAlmostEqual(
                distance,
                float(np.linalg.norm(positions[first] - positions[second])),
            )

    def test_rejects_cross_camera_trajectory_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("topdown", "angled")]
            for root in roots:
                (root / "latents").mkdir(parents=True)
                (root / "trajectories").mkdir()
                np.save(root / "latents/episode_0000.npy", np.ones((2, 256, 2), dtype=np.float16))
            for root, x in zip(roots, (0.0, 0.5)):
                torch.save(
                    {
                        "sampled_states": torch.tensor([[x, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                        "frame_indices": torch.tensor([0, 5]),
                    },
                    root / "trajectories/episode_0000.pt",
                )

            with self.assertRaisesRegex(ValueError, "position mismatch"):
                similarity.load_aligned_positions(
                    {"topdown": roots[0], "angled": roots[1]}, 0
                )

    def test_experiment_outputs_matched_camera_pairs_and_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            caches = {camera: root / camera for camera in ("topdown", "angled")}
            positions = torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0],
                 [0.15, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
            )
            for camera, cache in caches.items():
                (cache / "latents").mkdir(parents=True)
                (cache / "trajectories").mkdir()
                tokens = np.ones((4, 256, 2), dtype=np.float16)
                tokens[:, :, 1] = np.arange(4, dtype=np.float16)[:, None]
                if camera == "angled":
                    tokens[:, :, 1] *= -1
                np.save(cache / "latents/episode_0000.npy", tokens)
                torch.save(
                    {
                        "sampled_states": positions,
                        "frame_indices": torch.tensor([0, 5, 10, 15]),
                    },
                    cache / "trajectories/episode_0000.pt",
                )

            output = root / "results"
            similarity.run_experiment(
                cache_dirs=caches,
                output_dir=output,
                max_episodes=1,
                pairs_per_bin=1,
                seed=42,
                edges=(0.0, 0.1, 0.2, 0.4),
            )

            with (output / "pairs.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["camera"] for row in rows}, {"topdown", "angled"})
            for bin_index in range(3):
                matches = [row for row in rows if int(row["bin_index"]) == bin_index]
                self.assertEqual(len(matches), 2)
                self.assertEqual(
                    (matches[0]["episode_id"], matches[0]["frame_a"], matches[0]["frame_b"]),
                    (matches[1]["episode_id"], matches[1]["frame_a"], matches[1]["frame_b"]),
                )
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "similarity.png").is_file())

    def test_separate_runs_combine_after_source_caches_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dirs = []
            for camera in ("topdown", "angled"):
                cache = root / f"{camera}_cache"
                (cache / "latents").mkdir(parents=True)
                (cache / "trajectories").mkdir()
                tokens = np.ones((4, 256, 2), dtype=np.float16)
                tokens[:, :, 1] = np.arange(4, dtype=np.float16)[:, None]
                np.save(cache / "latents/episode_0000.npy", tokens)
                torch.save(
                    {
                        "sampled_states": torch.tensor(
                            [[0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0],
                             [0.15, 0.0, 0.0, 0.0], [0.35, 0.0, 0.0, 0.0]]
                        ),
                        "frame_indices": torch.tensor([0, 5, 10, 15]),
                    },
                    cache / "trajectories/episode_0000.pt",
                )
                output = root / f"{camera}_results"
                similarity.run_experiment(
                    cache_dirs={camera: cache},
                    output_dir=output,
                    max_episodes=1,
                    pairs_per_bin=1,
                    seed=42,
                    edges=(0.0, 0.1, 0.2, 0.4),
                )
                result_dirs.append(output)
                shutil.rmtree(cache)

            combined = root / "combined"
            similarity.main([
                "--combine-results", str(result_dirs[0]), str(result_dirs[1]),
                "--output-dir", str(combined),
            ])

            with (combined / "pairs.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["camera"] for row in rows}, {"topdown", "angled"})
            self.assertTrue((combined / "summary.csv").is_file())
            self.assertTrue((combined / "similarity.png").is_file())

    def test_combine_rejects_different_frame_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dirs = []
            for camera in ("topdown", "angled"):
                cache = root / f"{camera}_cache"
                (cache / "latents").mkdir(parents=True)
                (cache / "trajectories").mkdir()
                np.save(cache / "latents/episode_0000.npy", np.ones((2, 256, 2), dtype=np.float16))
                torch.save(
                    {
                        "sampled_states": torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.0, 0.0]]),
                        "frame_indices": torch.tensor([0, 5]),
                    },
                    cache / "trajectories/episode_0000.pt",
                )
                output = root / f"{camera}_results"
                similarity.run_experiment(
                    {camera: cache}, output, max_episodes=1, pairs_per_bin=1,
                    seed=42, edges=(0.0, 0.1),
                )
                result_dirs.append(output)

            path = result_dirs[1] / "pairs.csv"
            with path.open(newline="") as file:
                rows = list(csv.DictReader(file))
            rows[0]["frame_b"] = "0"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "pair selection differs"):
                similarity.combine_results(result_dirs, root / "combined")


if __name__ == "__main__":
    unittest.main()
