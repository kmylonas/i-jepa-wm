import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


try:
    import src.mylonas_ijepa_wm.position_probe as position_probe
except ModuleNotFoundError:
    position_probe = None

try:
    import src.mylonas_ijepa_wm.visualize_position_probe as visualize_probe
except ModuleNotFoundError:
    visualize_probe = None


class PositionProbeModuleTest(unittest.TestCase):
    def test_position_probe_module_is_available(self):
        self.assertIsNotNone(position_probe)

    def test_visualization_module_is_available(self):
        self.assertIsNotNone(visualize_probe)


class SpatialSoftmaxProbeTest(unittest.TestCase):
    def test_dominant_patch_maps_to_its_grid_coordinate(self):
        self.assertTrue(hasattr(position_probe, "SpatialSoftmaxProbe"))
        if not hasattr(position_probe, "SpatialSoftmaxProbe"):
            return

        probe = position_probe.SpatialSoftmaxProbe(
            ijepa_dim=1,
            grid_size=2,
            temperature=1.0,
        )
        with torch.no_grad():
            probe.patch_scorer.weight.fill_(1.0)
            probe.patch_scorer.bias.zero_()
            probe.coordinate_mapping.weight.copy_(torch.eye(2))
            probe.coordinate_mapping.bias.zero_()

        tokens = torch.tensor([[[0.0], [0.0], [0.0], [20.0]]])

        predicted_position = probe(tokens)
        probabilities = probe.patch_probabilities(tokens)

        torch.testing.assert_close(
            predicted_position,
            torch.tensor([[1.0, 1.0]]),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertEqual(probabilities.shape, (1, 4))
        self.assertEqual(probabilities.argmax(dim=1).item(), 3)
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones(1),
        )

    def test_mlp_scorer_produces_trainable_patch_probabilities(self):
        parameters = inspect.signature(
            position_probe.SpatialSoftmaxProbe
        ).parameters
        self.assertIn("scorer", parameters)
        self.assertIn("hidden_dim", parameters)
        if "scorer" not in parameters or "hidden_dim" not in parameters:
            return

        probe = position_probe.SpatialSoftmaxProbe(
            ijepa_dim=4,
            grid_size=2,
            temperature=1.0,
            scorer="mlp",
            hidden_dim=3,
        )
        tokens = torch.randn(2, 4, 4)

        probabilities = probe.patch_probabilities(tokens)
        loss = probe(tokens).square().mean()
        loss.backward()

        self.assertEqual(probabilities.shape, (2, 4))
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones(2),
        )
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in probe.patch_scorer.parameters()
            )
        )


class PositionProbeDatasetTest(unittest.TestCase):
    def test_returns_an_episode_of_latents_and_aligned_xy_positions(self):
        self.assertTrue(hasattr(position_probe, "PositionProbeDataset"))
        if not hasattr(position_probe, "PositionProbeDataset"):
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            latents = np.arange(
                3 * 4 * 2,
                dtype=np.float16,
            ).reshape(3, 4, 2)
            sampled_states = torch.tensor(
                [
                    [1.0, 2.0, 10.0, 20.0],
                    [3.0, 4.0, 30.0, 40.0],
                    [5.0, 6.0, 50.0, 60.0],
                ]
            )

            np.save(latent_dir / "episode_0007.npy", latents)
            torch.save(
                {"sampled_states": sampled_states},
                trajectory_dir / "episode_0007.pt",
            )

            dataset = position_probe.PositionProbeDataset(
                latent_dir=latent_dir,
                trajectory_dir=trajectory_dir,
                episode_ids=[7],
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]

            torch.testing.assert_close(
                sample["latents"],
                torch.from_numpy(latents),
            )
            torch.testing.assert_close(
                sample["positions"],
                sampled_states[:, :2],
            )
            self.assertEqual(sample["episode_id"], 7)

    def test_preloaded_latents_do_not_require_another_disk_read(self):
        self.assertTrue(hasattr(position_probe, "PositionProbeDataset"))
        if not hasattr(position_probe, "PositionProbeDataset"):
            return
        parameters = inspect.signature(
            position_probe.PositionProbeDataset
        ).parameters
        self.assertIn("preload_latents", parameters)
        if "preload_latents" not in parameters:
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            latent_path = latent_dir / "episode_0000.npy"
            latents = np.ones((2, 4, 3), dtype=np.float16)
            np.save(latent_path, latents)
            torch.save(
                {"sampled_states": torch.zeros(2, 4)},
                trajectory_dir / "episode_0000.pt",
            )

            dataset = position_probe.PositionProbeDataset(
                latent_dir=latent_dir,
                trajectory_dir=trajectory_dir,
                episode_ids=[0],
                preload_latents=True,
            )
            latent_path.unlink()

            torch.testing.assert_close(
                dataset[0]["latents"],
                torch.from_numpy(latents),
            )


class PositionStatisticsTest(unittest.TestCase):
    def test_statistics_use_only_sampled_xy_coordinates(self):
        self.assertTrue(
            hasattr(position_probe, "compute_position_statistics")
        )
        if not hasattr(position_probe, "compute_position_statistics"):
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            trajectory_dir = Path(temporary_directory)
            torch.save(
                {
                    "sampled_states": torch.tensor(
                        [
                            [1.0, 2.0, 100.0, 200.0],
                            [3.0, 6.0, 300.0, 600.0],
                        ]
                    )
                },
                trajectory_dir / "episode_0002.pt",
            )

            mean, std = position_probe.compute_position_statistics(
                trajectory_dir,
                episode_ids=[2],
            )

            torch.testing.assert_close(mean, torch.tensor([2.0, 4.0]))
            torch.testing.assert_close(std, torch.tensor([1.0, 2.0]))


class PositionMetricsTest(unittest.TestCase):
    def test_reports_world_coordinate_errors_and_r_squared(self):
        self.assertTrue(hasattr(position_probe, "compute_position_metrics"))
        if not hasattr(position_probe, "compute_position_metrics"):
            return

        predictions = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        targets = torch.tensor([[1.0, 1.0], [5.0, 5.0]])

        metrics = position_probe.compute_position_metrics(
            predictions,
            targets,
            baseline_position=torch.tensor([3.0, 3.0]),
        )

        self.assertEqual(metrics["num_samples"], 2)
        self.assertAlmostEqual(metrics["mse"], 1.25)
        self.assertAlmostEqual(metrics["mae_x"], 1.0)
        self.assertAlmostEqual(metrics["mae_y"], 0.5)
        self.assertAlmostEqual(metrics["mean_euclidean_error"], 1.5)
        self.assertAlmostEqual(metrics["r2_x"], 0.5)
        self.assertAlmostEqual(metrics["r2_y"], 0.875)
        self.assertAlmostEqual(
            metrics["baseline_mean_euclidean_error"],
            2.8284271,
            places=6,
        )


class ParserTest(unittest.TestCase):
    def test_defaults_match_the_cached_pointmaze_dataset(self):
        self.assertTrue(hasattr(position_probe, "parse_args"))
        if not hasattr(position_probe, "parse_args"):
            return

        args = position_probe.parse_args([])

        self.assertEqual(args.latent_dir, position_probe.LATENTS_PATH)
        self.assertEqual(
            args.trajectory_dir,
            position_probe.TRAJECTORIES_PATH,
        )
        self.assertEqual(args.num_episodes, 2000)
        self.assertEqual(args.train_episodes, 1600)
        self.assertEqual(args.val_episodes, 200)
        self.assertEqual(args.test_episodes, 200)
        self.assertEqual(args.grid_size, 16)
        self.assertEqual(args.ijepa_dim, 1280)
        self.assertTrue(hasattr(args, "scorer"))
        self.assertTrue(hasattr(args, "hidden_dim"))
        self.assertTrue(hasattr(args, "resume"))
        if not all(
            hasattr(args, name)
            for name in ("scorer", "hidden_dim", "resume")
        ):
            return
        self.assertEqual(args.scorer, "linear")
        self.assertEqual(args.hidden_dim, 128)
        self.assertFalse(args.preload_latents)
        self.assertFalse(args.resume)

    def test_mlp_and_resume_can_be_selected_from_the_command_line(self):
        defaults = position_probe.parse_args([])
        self.assertTrue(hasattr(defaults, "scorer"))
        self.assertTrue(hasattr(defaults, "hidden_dim"))
        self.assertTrue(hasattr(defaults, "resume"))
        if not all(
            hasattr(defaults, name)
            for name in ("scorer", "hidden_dim", "resume")
        ):
            return

        args = position_probe.parse_args([
            "--scorer", "mlp",
            "--hidden-dim", "64",
            "--resume",
        ])

        self.assertEqual(args.scorer, "mlp")
        self.assertEqual(args.hidden_dim, 64)
        self.assertTrue(args.resume)


class FixedNormalizedPositionModel(torch.nn.Module):
    def forward(self, tokens):
        return torch.zeros(
            len(tokens),
            2,
            device=tokens.device,
            dtype=tokens.dtype,
        )


class ProbeEpochTest(unittest.TestCase):
    def test_evaluation_flattens_episode_frames_and_reports_world_metrics(self):
        self.assertTrue(hasattr(position_probe, "run_epoch"))
        if not hasattr(position_probe, "run_epoch"):
            return

        batch = {
            "latents": torch.zeros(1, 2, 4, 3),
            "positions": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            "episode_id": torch.tensor([0]),
        }

        metrics = position_probe.run_epoch(
            model=FixedNormalizedPositionModel(),
            data_loader=[batch],
            device=torch.device("cpu"),
            position_mean=torch.tensor([2.0, 3.0]),
            position_std=torch.tensor([1.0, 1.0]),
        )

        self.assertEqual(metrics["num_samples"], 2)
        self.assertAlmostEqual(metrics["loss"], 1.0)
        self.assertAlmostEqual(metrics["mse"], 1.0)
        self.assertAlmostEqual(
            metrics["mean_euclidean_error"],
            2 ** 0.5,
        )


class ConstantPositionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position = torch.nn.Parameter(torch.ones(2))

    def forward(self, tokens):
        return self.position.expand(len(tokens), -1)


class ProbeTrainingTest(unittest.TestCase):
    def test_training_saves_best_and_last_checkpoints(self):
        self.assertTrue(hasattr(position_probe, "train_model"))
        if not hasattr(position_probe, "train_model"):
            return

        batch = {
            "latents": torch.zeros(1, 2, 4, 3),
            "positions": torch.zeros(1, 2, 2),
            "episode_id": torch.tensor([0]),
        }
        model = ConstantPositionModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temporary_directory:
            history, best_val_error = position_probe.train_model(
                model=model,
                optimizer=optimizer,
                train_loader=[batch],
                val_loader=[batch],
                device=torch.device("cpu"),
                position_mean=torch.zeros(2),
                position_std=torch.ones(2),
                epochs=2,
                output_dir=Path(temporary_directory),
                checkpoint_metadata={"probe": "test"},
            )

            self.assertEqual(len(history), 2)
            self.assertLessEqual(
                best_val_error,
                history[0]["val"]["mean_euclidean_error"],
            )
            self.assertTrue(
                (Path(temporary_directory) / "best.pt").is_file()
            )
            self.assertTrue(
                (Path(temporary_directory) / "last.pt").is_file()
            )

            checkpoint = torch.load(
                Path(temporary_directory) / "best.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["metadata"], {"probe": "test"})


class ProbeCommandTest(unittest.TestCase):
    def test_main_trains_and_writes_test_metrics(self):
        self.assertTrue(hasattr(position_probe, "main"))
        if not hasattr(position_probe, "main"):
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            output_dir = root / "output"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            for episode_id in range(3):
                rng = np.random.default_rng(episode_id)
                np.save(
                    latent_dir / f"episode_{episode_id:04d}.npy",
                    rng.normal(size=(2, 4, 3)).astype(np.float16),
                )
                torch.save(
                    {
                        "sampled_states": torch.tensor(
                            [
                                [episode_id, 0.0, 0.0, 0.0],
                                [episode_id + 1.0, 2.0, 0.0, 0.0],
                            ]
                        )
                    },
                    trajectory_dir / f"episode_{episode_id:04d}.pt",
                )

            position_probe.main([
                "--latent-dir", str(latent_dir),
                "--trajectory-dir", str(trajectory_dir),
                "--output-dir", str(output_dir),
                "--num-episodes", "3",
                "--train-episodes", "1",
                "--val-episodes", "1",
                "--test-episodes", "1",
                "--grid-size", "2",
                "--ijepa-dim", "3",
                "--batch-size", "1",
                "--epochs", "1",
                "--device", "cpu",
                "--log-every", "0",
            ])

            metrics_path = output_dir / "test_metrics.json"
            self.assertTrue(metrics_path.is_file())
            metrics = json.loads(metrics_path.read_text())
            self.assertEqual(metrics["num_samples"], 2)
            self.assertIn("mean_euclidean_error", metrics)
            self.assertIn("baseline_mean_euclidean_error", metrics)

    def test_resume_accepts_the_existing_linear_checkpoint_format(self):
        self.assertTrue(hasattr(position_probe, "main"))
        if not hasattr(position_probe, "main"):
            return
        defaults = position_probe.parse_args([])
        self.assertTrue(hasattr(defaults, "resume"))
        if not hasattr(defaults, "resume"):
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            output_dir = root / "output"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            for episode_id in range(3):
                np.save(
                    latent_dir / f"episode_{episode_id:04d}.npy",
                    np.zeros((2, 4, 3), dtype=np.float16),
                )
                torch.save(
                    {
                        "sampled_states": torch.tensor(
                            [
                                [episode_id, 0.0, 0.0, 0.0],
                                [episode_id + 1.0, 2.0, 0.0, 0.0],
                            ]
                        )
                    },
                    trajectory_dir / f"episode_{episode_id:04d}.pt",
                )

            common_arguments = [
                "--latent-dir", str(latent_dir),
                "--trajectory-dir", str(trajectory_dir),
                "--output-dir", str(output_dir),
                "--num-episodes", "3",
                "--train-episodes", "1",
                "--val-episodes", "1",
                "--test-episodes", "1",
                "--grid-size", "2",
                "--ijepa-dim", "3",
                "--batch-size", "1",
                "--device", "cpu",
                "--log-every", "0",
            ]
            position_probe.main(common_arguments + ["--epochs", "1"])

            checkpoint_path = output_dir / "last.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["metadata"]["model_configuration"].pop(
                "scorer",
                None,
            )
            checkpoint["metadata"]["model_configuration"].pop(
                "hidden_dim",
                None,
            )
            torch.save(checkpoint, checkpoint_path)

            position_probe.main(
                common_arguments + ["--epochs", "2", "--resume"]
            )

            resumed = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(len(resumed["history"]), 2)


class ProbeVisualizationTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.backends.mps.is_available(),
        "requires an MPS device",
    )
    def test_load_probe_keeps_normalization_metadata_on_cpu(self):
        model = position_probe.SpatialSoftmaxProbe(
            ijepa_dim=3,
            grid_size=2,
        )
        checkpoint = {
            "model": model.state_dict(),
            "metadata": {
                "model_configuration": {
                    "ijepa_dim": 3,
                    "grid_size": 2,
                    "temperature": 1.0,
                },
                "position_mean": torch.tensor([1.0, 2.0]),
                "position_std": torch.tensor([3.0, 4.0]),
            },
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "probe.pt"
            torch.save(checkpoint, checkpoint_path)

            loaded_model, metadata = visualize_probe.load_probe(
                checkpoint_path,
                torch.device("mps"),
            )

        self.assertEqual(next(loaded_model.parameters()).device.type, "mps")
        self.assertEqual(metadata["position_mean"].device.type, "cpu")
        self.assertEqual(metadata["position_std"].device.type, "cpu")

    def test_writes_coordinate_diagnostics_and_attention_overlay(self):
        self.assertIsNotNone(visualize_probe)
        if visualize_probe is None:
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            raw_dir = root / "raw"
            run_dir = root / "run"
            figure_dir = root / "figures"
            latent_dir.mkdir()
            trajectory_dir.mkdir()
            raw_dir.mkdir()

            for episode_id in range(3):
                rng = np.random.default_rng(episode_id)
                np.save(
                    latent_dir / f"episode_{episode_id:04d}.npy",
                    rng.normal(size=(2, 4, 3)).astype(np.float16),
                )
                torch.save(
                    {
                        "sampled_states": torch.tensor(
                            [
                                [episode_id, 0.0, 0.0, 0.0],
                                [episode_id + 1.0, 2.0, 0.0, 0.0],
                            ]
                        ),
                        "frame_indices": torch.tensor([0, 5]),
                    },
                    trajectory_dir / f"episode_{episode_id:04d}.pt",
                )
                torch.save(
                    {
                        "frames": torch.zeros(
                            6,
                            8,
                            8,
                            3,
                            dtype=torch.uint8,
                        )
                    },
                    raw_dir / f"episode_{episode_id:04d}.pt",
                )

            position_probe.main([
                "--latent-dir", str(latent_dir),
                "--trajectory-dir", str(trajectory_dir),
                "--output-dir", str(run_dir),
                "--num-episodes", "3",
                "--train-episodes", "1",
                "--val-episodes", "1",
                "--test-episodes", "1",
                "--grid-size", "2",
                "--ijepa-dim", "3",
                "--scorer", "mlp",
                "--hidden-dim", "2",
                "--batch-size", "1",
                "--epochs", "1",
                "--device", "cpu",
                "--log-every", "0",
            ])

            visualize_probe.main([
                "--checkpoint", str(run_dir / "best.pt"),
                "--latent-dir", str(latent_dir),
                "--trajectory-dir", str(trajectory_dir),
                "--output-dir", str(figure_dir),
                "--split", "test",
                "--episode-id", "0",
                "--raw-dir", str(raw_dir),
                "--frame-index", "1",
                "--device", "cpu",
            ])

            self.assertTrue(
                (figure_dir / "position_diagnostics.png").is_file()
            )
            self.assertTrue(
                (
                    figure_dir
                    / "attention_episode_0000_frame_01.png"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
