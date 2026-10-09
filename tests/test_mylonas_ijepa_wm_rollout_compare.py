import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import compare_rollout_models
from src.mylonas_ijepa_wm.position_probe import SpatialSoftmaxProbe
from src.mylonas_ijepa_wm.vit import ViT


REPO_ROOT = Path(__file__).resolve().parents[1]


class AddLastActionWorldModel(torch.nn.Module):
    def forward(self, latent_history, actions):
        predictions = torch.zeros_like(latent_history)
        predictions[:, -1] = (
            latent_history[:, -1]
            + actions[:, -1].unsqueeze(-1)
        )
        return predictions


class PersistenceWorldModel(torch.nn.Module):
    def forward(self, latent_history, actions):
        return latent_history


class ScalarPositionProbe(torch.nn.Module):
    def forward(self, latents):
        x_coordinate = latents[:, 0, 0]
        return torch.stack(
            [x_coordinate, torch.zeros_like(x_coordinate)],
            dim=1,
        )


class SequenceDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "latent_sequence": torch.tensor(
                [[[0.0]], [[1.0]], [[3.0]], [[6.0]], [[10.0]]],
            ),
            "actions": torch.tensor(
                [[1.0], [2.0], [3.0], [4.0]],
            ),
            "positions": torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [6.0, 0.0],
                    [10.0, 0.0],
                ]
            ),
            "episode_id": 17,
            "start": 4,
        }


class ImprovementSummaryTest(unittest.TestCase):
    def test_positive_improvement_means_candidate_has_lower_error(self):
        summary = compare_rollout_models.summarize_improvements(
            baseline_errors=torch.tensor(
                [[4.0, 5.0], [2.0, 4.0], [1.0, 3.0]],
            ),
            candidate_errors=torch.tensor(
                [[2.0, 3.0], [3.0, 2.0], [1.0, 4.0]],
            ),
        )

        self.assertEqual(summary["num_samples"], 3)
        self.assertAlmostEqual(
            summary["final_horizon"]["percent_improved"],
            100.0 * 2.0 / 3.0,
            places=4,
        )
        self.assertAlmostEqual(
            summary["final_horizon"]["median_improvement"],
            2.0,
        )
        self.assertAlmostEqual(
            summary["mean_over_horizons"]["mean_improvement"],
            2.0 / 3.0,
        )
        self.assertEqual(len(summary["per_horizon"]), 2)

    def test_median_interpolates_the_two_middle_windows(self):
        summary = compare_rollout_models.summarize_improvements(
            baseline_errors=torch.tensor([[2.0], [4.0]]),
            candidate_errors=torch.tensor([[1.0], [1.0]]),
        )

        self.assertEqual(
            summary["final_horizon"]["median_improvement"],
            2.0,
        )

    def test_selects_unique_best_median_unchanged_and_worst_samples(self):
        selected = compare_rollout_models.select_representative_indices(
            torch.tensor([5.0, -5.0, 0.1, 1.0, 2.0]),
        )

        self.assertEqual(selected["best"], 0)
        self.assertEqual(selected["median"], 3)
        self.assertEqual(selected["unchanged"], 2)
        self.assertEqual(selected["worst"], 1)
        self.assertEqual(len(set(selected.values())), 4)


class PositionRolloutComparisonTest(unittest.TestCase):
    def test_returns_per_sample_errors_and_identifiers(self):
        loader = torch.utils.data.DataLoader(
            SequenceDataset(),
            batch_size=1,
        )

        comparison = compare_rollout_models.compare_position_rollouts(
            baseline_model=PersistenceWorldModel(),
            candidate_model=AddLastActionWorldModel(),
            position_probe=ScalarPositionProbe(),
            data_loader=loader,
            device=torch.device("cpu"),
            num_hist=3,
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
        )

        torch.testing.assert_close(
            comparison["baseline_errors"],
            torch.tensor([[3.0, 7.0]]),
        )
        torch.testing.assert_close(
            comparison["candidate_errors"],
            torch.zeros(1, 2),
        )
        torch.testing.assert_close(
            comparison["oracle_errors"],
            torch.zeros(1, 2),
        )
        self.assertEqual(comparison["episode_ids"].tolist(), [17])
        self.assertEqual(comparison["starts"].tolist(), [4])


class ComparisonPlotTest(unittest.TestCase):
    def test_saves_summary_and_representative_trajectory_figures(self):
        summary = {
            "per_horizon": [
                {
                    "horizon": 1,
                    "baseline_mean_error": 0.5,
                    "candidate_mean_error": 0.3,
                    "mean_improvement": 0.2,
                    "percent_improved": 75.0,
                },
                {
                    "horizon": 2,
                    "baseline_mean_error": 0.8,
                    "candidate_mean_error": 0.4,
                    "mean_improvement": 0.4,
                    "percent_improved": 80.0,
                },
            ],
        }
        trajectory = {
            "category": "best",
            "improvement": 0.4,
            "episode_id": 12,
            "start": 0,
            "true": torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]],
            ),
            "oracle": torch.tensor(
                [[0.0, 0.0], [1.1, 0.0], [2.1, 1.0]],
            ),
            "baseline_autoregressive": torch.tensor(
                [[0.0, 0.0], [0.5, 0.2], [1.0, 0.5]],
            ),
            "candidate_autoregressive": torch.tensor(
                [[0.0, 0.0], [0.9, 0.1], [1.8, 0.9]],
            ),
            "baseline_teacher_forced": torch.tensor(
                [[0.0, 0.0], [0.7, 0.1], [1.6, 0.8]],
            ),
            "candidate_teacher_forced": torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.9, 1.0]],
            ),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary.png"
            trajectory_path = root / "trajectories.png"
            compare_rollout_models.save_comparison_plot(
                summary=summary,
                final_improvements=torch.tensor([-0.2, 0.1, 0.4]),
                output_path=summary_path,
            )
            compare_rollout_models.save_representative_trajectory_plot(
                trajectories=[trajectory],
                baseline_label="K=1",
                candidate_label="K=3",
                output_path=trajectory_path,
            )

            self.assertTrue(summary_path.is_file())
            self.assertTrue(trajectory_path.is_file())


class ParserTest(unittest.TestCase):
    def test_defaults_to_test_split_and_horizon_ten(self):
        args = compare_rollout_models.parse_args([])

        self.assertEqual(args.split, "test")
        self.assertEqual(args.horizon, 10)
        self.assertEqual(args.batch_size, 16)
        self.assertIsNone(args.output)


class CommandLineComparisonTest(unittest.TestCase):
    def test_compares_checkpoints_and_saves_all_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            model_configuration = {
                "num_patches": 4,
                "num_hist": 3,
                "num_t_blocks": 1,
                "ijepa_dim": 1,
                "action_dim": 1,
                "action_embed_dim": 2,
                "embed_dim": 4,
                "num_heads": 1,
                "mlp_dim": 8,
                "dropout": 0.0,
            }
            model = ViT(**model_configuration)
            metadata = {
                "model_configuration": model_configuration,
                "train_ids": [],
                "val_ids": [],
                "test_ids": [0],
                "normalize_actions": False,
            }
            baseline_path = root / "baseline.pt"
            candidate_path = root / "candidate.pt"
            for checkpoint_path in [baseline_path, candidate_path]:
                torch.save(
                    {
                        "epoch": 1,
                        "model": model.state_dict(),
                        "metadata": metadata,
                    },
                    checkpoint_path,
                )

            probe = SpatialSoftmaxProbe(
                ijepa_dim=1,
                grid_size=2,
                scorer="mlp",
                hidden_dim=2,
            )
            probe_path = root / "probe.pt"
            torch.save(
                {
                    "epoch": 1,
                    "model": probe.state_dict(),
                    "metadata": {
                        "model_configuration": {
                            "ijepa_dim": 1,
                            "grid_size": 2,
                            "temperature": 1.0,
                            "scorer": "mlp",
                            "hidden_dim": 2,
                        },
                        "position_mean": torch.zeros(2),
                        "position_std": torch.ones(2),
                        "test_ids": [0],
                    },
                },
                probe_path,
            )
            np.save(
                latent_dir / "episode_0000.npy",
                np.zeros((8, 4, 1), dtype=np.float16),
            )
            torch.save(
                {
                    "macro_actions": torch.zeros(7, 1),
                    "sampled_states": torch.zeros(8, 4),
                },
                trajectory_dir / "episode_0000.pt",
            )

            output_path = root / "comparison.json"
            plot_path = root / "comparison.png"
            trajectory_plot_path = root / "trajectories.png"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.mylonas_ijepa_wm.compare_rollout_models",
                    "--baseline-checkpoint", str(baseline_path),
                    "--candidate-checkpoint", str(candidate_path),
                    "--position-probe-checkpoint", str(probe_path),
                    "--latent-dir", str(latent_dir),
                    "--trajectory-dir", str(trajectory_dir),
                    "--horizon", "1",
                    "--batch-size", "2",
                    "--num-workers", "0",
                    "--device", "cpu",
                    "--log-every", "0",
                    "--output", str(output_path),
                    "--plot-output", str(plot_path),
                    "--trajectory-plot-output",
                    str(trajectory_plot_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + result.stderr,
            )
            self.assertTrue(plot_path.is_file())
            self.assertTrue(trajectory_plot_path.is_file())
            with output_path.open(encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)

            self.assertEqual(metrics["summary"]["num_samples"], 5)
            self.assertEqual(len(metrics["windows"]), 5)
            self.assertEqual(len(metrics["representative_cases"]), 4)


if __name__ == "__main__":
    unittest.main()
