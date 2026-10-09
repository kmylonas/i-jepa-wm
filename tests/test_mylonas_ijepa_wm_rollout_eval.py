import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm.position_probe import SpatialSoftmaxProbe

try:
    import src.mylonas_ijepa_wm.evaluate_rollouts as evaluate_rollouts
except ModuleNotFoundError:
    evaluate_rollouts = None


REPO_ROOT = Path(__file__).resolve().parents[1]


class RolloutEvaluationModuleTest(unittest.TestCase):
    def test_rollout_evaluation_module_is_available(self):
        self.assertIsNotNone(evaluate_rollouts)


class RolloutEvaluationDatasetTest(unittest.TestCase):
    def test_returns_aligned_history_targets_and_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            latents = np.arange(8, dtype=np.float16).reshape(8, 1, 1)
            actions = torch.arange(7, dtype=torch.float32).reshape(7, 1)
            sampled_states = torch.arange(
                8 * 4,
                dtype=torch.float32,
            ).reshape(8, 4)
            np.save(latent_dir / "episode_0003.npy", latents)
            torch.save(
                {
                    "macro_actions": actions,
                    "sampled_states": sampled_states,
                },
                trajectory_dir / "episode_0003.pt",
            )

            dataset = evaluate_rollouts.RolloutEvaluationDataset(
                latent_dir=latent_dir,
                trajectory_dir=trajectory_dir,
                episode_ids=[3],
                num_hist=3,
                horizon=2,
                action_mean=torch.tensor([1.0]),
                action_std=torch.tensor([2.0]),
            )

            self.assertEqual(len(dataset), 4)
            sample = dataset[1]

            torch.testing.assert_close(
                sample["latent_sequence"],
                torch.from_numpy(latents[1:6]),
            )
            torch.testing.assert_close(
                sample["actions"],
                torch.tensor([[0.0], [0.5], [1.0], [1.5]]),
            )
            torch.testing.assert_close(
                sample["positions"],
                sampled_states[1:6, :2],
            )
            self.assertEqual(sample["episode_id"], 3)
            self.assertEqual(sample["start"], 1)


class EpisodeIntersectionTest(unittest.TestCase):
    def test_preserves_world_model_order_and_rejects_empty_intersection(self):
        self.assertEqual(
            evaluate_rollouts.intersect_episode_ids(
                [7, 3, 9, 1],
                [1, 7, 8],
            ),
            [7, 1],
        )

        with self.assertRaisesRegex(ValueError, "no shared episodes"):
            evaluate_rollouts.intersect_episode_ids([1, 2], [3, 4])


class AddLastActionWorldModel(torch.nn.Module):
    def forward(self, latent_history, actions):
        predictions = torch.zeros_like(latent_history)
        next_latent = (
            latent_history[:, -1]
            + actions[:, -1].unsqueeze(-1)
        )
        predictions[:, -1] = next_latent
        return predictions


class AutoregressiveRolloutTest(unittest.TestCase):
    def test_feeds_predictions_back_and_advances_action_window(self):
        model = AddLastActionWorldModel()
        latent_history = torch.tensor(
            [[[[0.0]], [[1.0]], [[3.0]]]],
        )
        actions = torch.tensor(
            [[[1.0], [2.0], [3.0], [4.0]]],
        )

        predictions = evaluate_rollouts.autoregressive_rollout(
            model,
            latent_history,
            actions,
        )

        torch.testing.assert_close(
            predictions,
            torch.tensor([[[[6.0]], [[10.0]]]]),
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
        }


class ScalarPositionProbe(torch.nn.Module):
    def forward(self, latents):
        x_coordinate = latents[:, 0, 0]
        return torch.stack(
            [x_coordinate, torch.zeros_like(x_coordinate)],
            dim=1,
        )


class RolloutMetricsTest(unittest.TestCase):
    def test_reports_autoregressive_teacher_forced_and_persistence_mse(self):
        loader = torch.utils.data.DataLoader(
            SequenceDataset(),
            batch_size=1,
        )

        metrics = evaluate_rollouts.evaluate_model(
            model=AddLastActionWorldModel(),
            data_loader=loader,
            device=torch.device("cpu"),
            num_hist=3,
        )

        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(
            metrics["per_horizon"],
            [
                {
                    "horizon": 1,
                    "autoregressive_mse": 0.0,
                    "teacher_forced_mse": 0.0,
                    "persistence_mse": 9.0,
                },
                {
                    "horizon": 2,
                    "autoregressive_mse": 0.0,
                    "teacher_forced_mse": 0.0,
                    "persistence_mse": 49.0,
                },
            ],
        )

    def test_decodes_position_from_all_rollout_sources_per_horizon(self):
        loader = torch.utils.data.DataLoader(
            SequenceDataset(),
            batch_size=1,
        )

        metrics = evaluate_rollouts.evaluate_model(
            model=AddLastActionWorldModel(),
            data_loader=loader,
            device=torch.device("cpu"),
            num_hist=3,
            position_probe=ScalarPositionProbe(),
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
        )

        first = metrics["per_horizon"][0]["position"]
        second = metrics["per_horizon"][1]["position"]

        for method in ["oracle", "autoregressive", "teacher_forced"]:
            self.assertAlmostEqual(
                first[method]["mean_euclidean_error"],
                0.0,
            )
            self.assertAlmostEqual(
                second[method]["mean_euclidean_error"],
                0.0,
            )

        self.assertAlmostEqual(
            first["persistence"]["mean_euclidean_error"],
            3.0,
        )
        self.assertAlmostEqual(
            second["persistence"]["mean_euclidean_error"],
            7.0,
        )


class RolloutTrajectoryPlotTest(unittest.TestCase):
    def test_predicts_true_oracle_autoregressive_and_teacher_forced_paths(self):
        sample = SequenceDataset()[0]
        sample["episode_id"] = 12
        sample["start"] = 0

        trajectory = evaluate_rollouts.predict_position_trajectory(
            model=AddLastActionWorldModel(),
            position_probe=ScalarPositionProbe(),
            sample=sample,
            device=torch.device("cpu"),
            num_hist=3,
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
        )

        expected = torch.tensor(
            [[3.0, 0.0], [6.0, 0.0], [10.0, 0.0]]
        )
        self.assertEqual(trajectory["episode_id"], 12)
        self.assertEqual(trajectory["start"], 0)
        torch.testing.assert_close(trajectory["true"], expected)
        torch.testing.assert_close(trajectory["oracle"], expected)
        torch.testing.assert_close(
            trajectory["autoregressive"],
            expected,
        )
        torch.testing.assert_close(
            trajectory["teacher_forced"],
            expected,
        )

    def test_saves_trajectory_figure(self):
        trajectory = {
            "episode_id": 12,
            "start": 0,
            "true": torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]
            ),
            "oracle": torch.tensor(
                [[0.0, 0.0], [1.1, 0.0], [2.1, 1.0]]
            ),
            "autoregressive": torch.tensor(
                [[0.0, 0.0], [0.8, 0.1], [1.5, 0.8]]
            ),
            "teacher_forced": torch.tensor(
                [[0.0, 0.0], [0.9, 0.0], [1.9, 0.9]]
            ),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "trajectories.png"
            evaluate_rollouts.save_trajectory_plot(
                [trajectory],
                output_path,
            )
            self.assertTrue(output_path.is_file())


class ParserTest(unittest.TestCase):
    def test_defaults_to_validation_rollouts_at_horizon_five(self):
        args = evaluate_rollouts.parse_args([])

        self.assertEqual(args.split, "val")
        self.assertEqual(args.horizon, 5)
        self.assertEqual(args.batch_size, 16)
        self.assertIsNone(args.position_probe_checkpoint)
        self.assertIsNone(args.plot_output)
        self.assertEqual(args.num_trajectory_plots, 4)
        self.assertIsNone(args.trajectory_plot_output)

    def test_teacher_forcing_uses_real_history_at_each_horizon(self):
        model = AddLastActionWorldModel()
        latent_sequence = torch.tensor(
            [[[[0.0]], [[1.0]], [[3.0]], [[6.0]], [[10.0]]]],
        )
        actions = torch.tensor(
            [[[1.0], [2.0], [3.0], [4.0]]],
        )

        predictions = evaluate_rollouts.teacher_forced_rollout(
            model,
            latent_sequence,
            actions,
            num_hist=3,
        )

        torch.testing.assert_close(
            predictions,
            torch.tensor([[[[6.0]], [[10.0]]]]),
        )


class CommandLineEvaluationTest(unittest.TestCase):
    def test_evaluates_checkpoint_and_saves_horizon_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            checkpoint_path = root / "best.pt"
            output_path = root / "metrics.json"
            plot_path = root / "metrics.png"
            trajectory_plot_path = root / "trajectories.png"
            probe_checkpoint_path = root / "probe.pt"
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
            model = evaluate_rollouts.ViT(**model_configuration)
            torch.save(
                {
                    "epoch": 2,
                    "model": model.state_dict(),
                    "metadata": {
                        "model_configuration": model_configuration,
                        "val_ids": [0],
                        "normalize_actions": False,
                    },
                },
                checkpoint_path,
            )
            np.save(
                latent_dir / "episode_0000.npy",
                np.zeros((6, 4, 1), dtype=np.float16),
            )
            torch.save(
                {
                    "macro_actions": torch.zeros(5, 1),
                    "sampled_states": torch.zeros(6, 4),
                },
                trajectory_dir / "episode_0000.pt",
            )
            probe = SpatialSoftmaxProbe(
                ijepa_dim=1,
                grid_size=2,
                scorer="mlp",
                hidden_dim=2,
            )
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
                        "val_ids": [0],
                    },
                },
                probe_checkpoint_path,
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.mylonas_ijepa_wm.evaluate_rollouts",
                    "--checkpoint", str(checkpoint_path),
                    "--position-probe-checkpoint",
                    str(probe_checkpoint_path),
                    "--latent-dir", str(latent_dir),
                    "--trajectory-dir", str(trajectory_dir),
                    "--horizon", "2",
                    "--batch-size", "2",
                    "--num-workers", "0",
                    "--device", "cpu",
                    "--log-every", "0",
                    "--output", str(output_path),
                    "--plot-output", str(plot_path),
                    "--trajectory-plot-output",
                    str(trajectory_plot_path),
                    "--num-trajectory-plots", "1",
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
            with output_path.open(encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)

            self.assertEqual(metrics["checkpoint_epoch"], 2)
            self.assertEqual(metrics["split"], "val")
            self.assertEqual(metrics["num_samples"], 2)
            self.assertEqual(len(metrics["per_horizon"]), 2)
            self.assertIn("position", metrics["per_horizon"][0])
            self.assertEqual(metrics["num_probe_split_episodes"], 1)
            self.assertEqual(metrics["num_evaluated_episodes"], 1)
            self.assertEqual(metrics["trajectory_plot_episode_ids"], [0])
            self.assertTrue(plot_path.is_file())
            self.assertTrue(trajectory_plot_path.is_file())


if __name__ == "__main__":
    unittest.main()
