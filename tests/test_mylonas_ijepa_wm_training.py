import os
import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import src.mylonas_ijepa_wm.wm as wm


REPO_ROOT = Path(__file__).resolve().parents[1]


class WorldModelModuleTest(unittest.TestCase):
    def test_module_can_be_imported_without_loading_local_data(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import src.mylonas_ijepa_wm.wm",
                ],
                cwd=temporary_directory,
                env=environment,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            result.returncode,
            0,
            msg=result.stdout + result.stderr,
        )


class ParserTest(unittest.TestCase):
    def test_defaults_match_the_local_world_model_setup(self):
        self.assertTrue(hasattr(wm, "parse_args"))

        args = wm.parse_args([])

        self.assertEqual(args.latent_dir, wm.LATENTS_PATH)
        self.assertEqual(args.trajectory_dir, wm.TRAJECTORIES_PATH)
        self.assertEqual(
            args.output_dir,
            wm.REPO_ROOT / "world_model_runs" / "baseline",
        )
        self.assertEqual(args.num_episodes, 2000)
        self.assertEqual(args.train_episodes, 1600)
        self.assertEqual(args.val_episodes, 200)
        self.assertEqual(args.test_episodes, 200)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.validate_every, 1)
        self.assertEqual(args.precision, "auto")
        self.assertFalse(args.compile)
        self.assertFalse(args.preload_latents)
        self.assertEqual(args.rollout_steps, 1)
        self.assertEqual(args.rollout_loss_weight, 0.0)
        self.assertIsNone(args.finetune_from)
        self.assertTrue(hasattr(args, "normalize_actions"))
        if hasattr(args, "normalize_actions"):
            self.assertTrue(args.normalize_actions)
        self.assertFalse(args.resume)

    def test_multistep_finetuning_can_be_configured_from_the_command_line(self):
        checkpoint_path = Path("runs/k1/best.pt")

        args = wm.parse_args([
            "--rollout-steps", "3",
            "--rollout-loss-weight", "1.0",
            "--finetune-from", str(checkpoint_path),
        ])

        self.assertEqual(args.rollout_steps, 3)
        self.assertEqual(args.rollout_loss_weight, 1.0)
        self.assertEqual(args.finetune_from, checkpoint_path)

    def test_preload_latents_can_be_enabled_from_the_command_line(self):
        args = wm.parse_args(["--preload-latents"])

        self.assertTrue(args.preload_latents)


class DataLoaderWorkerTest(unittest.TestCase):
    def test_spawn_workers_are_disabled_for_preloaded_numpy_arrays(self):
        num_workers = wm.resolve_num_workers(
            num_workers=2,
            preload_latents=True,
            start_method="spawn",
        )

        self.assertEqual(num_workers, 0)

    def test_fork_workers_can_share_preloaded_numpy_arrays(self):
        num_workers = wm.resolve_num_workers(
            num_workers=2,
            preload_latents=True,
            start_method="fork",
        )

        self.assertEqual(num_workers, 2)


class PrecisionTest(unittest.TestCase):
    def test_auto_uses_bfloat16_when_cuda_supports_it(self):
        precision = wm.resolve_precision(
            "auto",
            torch.device("cuda"),
            bf16_supported=True,
        )

        self.assertEqual(precision, "bf16")

    def test_auto_uses_float32_without_supported_cuda(self):
        precision = wm.resolve_precision(
            "auto",
            torch.device("cpu"),
            bf16_supported=False,
        )

        self.assertEqual(precision, "fp32")

    def test_explicit_bfloat16_rejects_unsupported_device(self):
        with self.assertRaises(ValueError):
            wm.resolve_precision(
                "bf16",
                torch.device("cpu"),
                bf16_supported=False,
            )


class CompileModelTest(unittest.TestCase):
    def test_compilation_preserves_outputs_and_checkpoint_keys(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 2)
        inputs = torch.randn(4, 3)

        expected = model(inputs)
        original_keys = list(model.state_dict())

        compiled_model = wm.compile_model(
            model,
            enabled=True,
            backend="eager",
        )
        actual = compiled_model(inputs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            list(compiled_model.state_dict()),
            original_keys,
        )


class WorldModelDatasetTest(unittest.TestCase):
    def test_returns_one_contiguous_latent_window_and_aligned_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            latents = np.arange(
                5 * 2 * 3,
                dtype=np.float16,
            ).reshape(5, 2, 3)
            actions = torch.arange(
                4 * 2,
                dtype=torch.float32,
            ).reshape(4, 2)

            np.save(latent_dir / "episode_0007.npy", latents)
            torch.save(
                {"macro_actions": actions},
                trajectory_dir / "episode_0007.pt",
            )

            dataset = wm.WorldModelDataset(
                latent_dir,
                trajectory_dir,
                episode_ids=[7],
                num_hist=2,
            )

            self.assertEqual(len(dataset), 3)

            sample = dataset[1]

            torch.testing.assert_close(
                sample["latent_window"],
                torch.from_numpy(latents[1:4]),
            )
            torch.testing.assert_close(
                sample["actions"],
                actions[1:3],
            )

    def test_multistep_samples_include_future_latents_and_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            latents = np.arange(7, dtype=np.float16).reshape(7, 1, 1)
            actions = torch.arange(6, dtype=torch.float32).reshape(6, 1)
            np.save(latent_dir / "episode_0000.npy", latents)
            torch.save(
                {"macro_actions": actions},
                trajectory_dir / "episode_0000.pt",
            )

            dataset = wm.WorldModelDataset(
                latent_dir,
                trajectory_dir,
                episode_ids=[0],
                num_hist=2,
                rollout_steps=3,
            )

            self.assertEqual(len(dataset), 3)
            sample = dataset[1]
            torch.testing.assert_close(
                sample["latent_window"],
                torch.from_numpy(latents[1:6]),
            )
            torch.testing.assert_close(
                sample["actions"],
                actions[1:5],
            )

    def test_applies_training_action_statistics(self):
        parameters = inspect.signature(
            wm.WorldModelDataset
        ).parameters
        self.assertIn("action_mean", parameters)
        self.assertIn("action_std", parameters)

        if "action_mean" not in parameters or "action_std" not in parameters:
            return

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            np.save(
                latent_dir / "episode_0000.npy",
                np.zeros((5, 2, 3), dtype=np.float16),
            )
            torch.save(
                {
                    "macro_actions": torch.tensor(
                        [
                            [0.0, 2.0],
                            [2.0, 4.0],
                            [4.0, 6.0],
                            [6.0, 8.0],
                        ]
                    )
                },
                trajectory_dir / "episode_0000.pt",
            )

            dataset = wm.WorldModelDataset(
                latent_dir,
                trajectory_dir,
                episode_ids=[0],
                num_hist=2,
                action_mean=torch.tensor([1.0, 3.0]),
                action_std=torch.tensor([1.0, 1.0]),
            )

            sample = dataset[0]

            torch.testing.assert_close(
                sample["actions"],
                torch.tensor([[-1.0, -1.0], [1.0, 1.0]]),
            )

    def test_reuses_episode_data_across_overlapping_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            np.save(
                latent_dir / "episode_0000.npy",
                np.zeros((5, 2, 3), dtype=np.float16),
            )
            torch.save(
                {"macro_actions": torch.zeros(4, 2)},
                trajectory_dir / "episode_0000.pt",
            )

            original_load = np.load
            with mock.patch.object(
                wm.np,
                "load",
                wraps=original_load,
            ) as numpy_load:
                dataset = wm.WorldModelDataset(
                    latent_dir,
                    trajectory_dir,
                    episode_ids=[0],
                    num_hist=2,
                )
                first = dataset[0]
                second = dataset[1]

            self.assertEqual(numpy_load.call_count, 2)
            torch.testing.assert_close(
                first["actions"],
                torch.zeros(2, 2),
            )
            torch.testing.assert_close(
                second["actions"],
                torch.zeros(2, 2),
            )

    def test_preloaded_latents_do_not_require_the_source_file_per_sample(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            latent_path = latent_dir / "episode_0000.npy"
            latents = np.arange(
                5 * 2 * 3,
                dtype=np.float16,
            ).reshape(5, 2, 3)
            np.save(latent_path, latents)
            torch.save(
                {"macro_actions": torch.zeros(4, 2)},
                trajectory_dir / "episode_0000.pt",
            )

            dataset = wm.WorldModelDataset(
                latent_dir,
                trajectory_dir,
                episode_ids=[0],
                num_hist=2,
                preload_latents=True,
            )
            latent_path.unlink()

            sample = dataset[1]

            torch.testing.assert_close(
                sample["latent_window"],
                torch.from_numpy(latents[1:4]),
            )

    def test_preloaded_latents_are_stored_as_float16(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            np.save(
                latent_dir / "episode_0000.npy",
                np.zeros((5, 2, 3), dtype=np.float32),
            )
            torch.save(
                {"macro_actions": torch.zeros(4, 2)},
                trajectory_dir / "episode_0000.pt",
            )

            dataset = wm.WorldModelDataset(
                latent_dir,
                trajectory_dir,
                episode_ids=[0],
                num_hist=2,
                preload_latents=True,
            )

            self.assertEqual(
                dataset._latent_cache[0].dtype,
                np.dtype(np.float16),
            )
            self.assertEqual(
                dataset[0]["latent_window"].dtype,
                torch.float16,
            )


class ActionStatisticsTest(unittest.TestCase):
    def test_statistics_are_computed_from_selected_episodes(self):
        self.assertTrue(hasattr(wm, "compute_action_statistics"))

        with tempfile.TemporaryDirectory() as temporary_directory:
            trajectory_dir = Path(temporary_directory)

            torch.save(
                {
                    "macro_actions": torch.tensor(
                        [[0.0, 2.0], [2.0, 4.0]]
                    )
                },
                trajectory_dir / "episode_0000.pt",
            )
            torch.save(
                {
                    "macro_actions": torch.tensor(
                        [[100.0, 100.0]]
                    )
                },
                trajectory_dir / "episode_0001.pt",
            )

            mean, std = wm.compute_action_statistics(
                trajectory_dir,
                episode_ids=[0],
            )

            torch.testing.assert_close(
                mean,
                torch.tensor([1.0, 3.0]),
            )
            torch.testing.assert_close(
                std,
                torch.tensor([1.0, 1.0]),
            )


class EpisodeSplitTest(unittest.TestCase):
    def test_split_is_reproducible_and_has_no_episode_leakage(self):
        self.assertTrue(hasattr(wm, "create_episode_splits"))

        first = wm.create_episode_splits(
            num_episodes=10,
            train_episodes=6,
            val_episodes=2,
            test_episodes=2,
            seed=42,
        )
        second = wm.create_episode_splits(
            num_episodes=10,
            train_episodes=6,
            val_episodes=2,
            test_episodes=2,
            seed=42,
        )

        self.assertEqual(first, second)

        train_ids, val_ids, test_ids = first

        self.assertEqual(len(train_ids), 6)
        self.assertEqual(len(val_ids), 2)
        self.assertEqual(len(test_ids), 2)
        self.assertEqual(
            set(train_ids) | set(val_ids) | set(test_ids),
            set(range(10)),
        )
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertFalse(set(val_ids) & set(test_ids))

    def test_rejects_non_positive_split_sizes(self):
        with self.assertRaises(ValueError):
            wm.create_episode_splits(
                num_episodes=10,
                train_episodes=-1,
                val_episodes=5,
                test_episodes=6,
                seed=42,
            )


class ScalarWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_latents, actions):
        return torch.ones_like(input_latents) * self.weight


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, target):
        self.target = target

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "latent_window": torch.tensor(
                [[0.0], [self.target]],
            ),
            "actions": torch.zeros(1),
        }


class RecordingLinearWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.output_dtype = None

    def forward(self, input_latents, actions):
        output = self.linear(input_latents)
        self.output_dtype = output.dtype
        return output


class IncrementWorldModel(torch.nn.Module):
    def forward(self, input_latents, actions):
        return input_latents + 1.0


class TrainingLoopTest(unittest.TestCase):
    def test_run_epoch_can_report_one_step_and_rollout_losses(self):
        loader = torch.utils.data.DataLoader(
            [{
                "latent_window": torch.tensor(
                    [[[0.0]], [[1.0]], [[2.0]], [[10.0]], [[20.0]]]
                ),
                "actions": torch.zeros(4, 1),
            }],
            batch_size=1,
        )

        metrics = wm.run_epoch(
            IncrementWorldModel(),
            loader,
            torch.device("cpu"),
            rollout_steps=3,
            rollout_loss_weight=0.5,
            return_metrics=True,
        )

        self.assertEqual(metrics["one_step_loss"], 0.0)
        self.assertAlmostEqual(
            metrics["rollout_loss"],
            (0.0 + 49.0 + 256.0) / 3.0,
            places=5,
        )
        self.assertAlmostEqual(
            metrics["loss"],
            0.5 * metrics["rollout_loss"],
            places=5,
        )

    def test_multistep_loss_feeds_predictions_back_autoregressively(self):
        latent_window = torch.tensor(
            [[[[0.0]], [[1.0]], [[2.0]], [[10.0]], [[20.0]]]]
        )
        actions = torch.zeros(1, 4, 1)

        total_loss, one_step_loss, rollout_loss = (
            wm.compute_world_model_loss(
                model=IncrementWorldModel(),
                latent_window=latent_window,
                actions=actions,
                num_hist=2,
                rollout_steps=3,
                rollout_loss_weight=0.5,
            )
        )

        self.assertEqual(one_step_loss.item(), 0.0)
        self.assertAlmostEqual(
            rollout_loss.item(),
            (0.0 + 49.0 + 256.0) / 3.0,
            places=5,
        )
        self.assertAlmostEqual(
            total_loss.item(),
            0.5 * (0.0 + 49.0 + 256.0) / 3.0,
            places=5,
        )

    def test_model_only_checkpoint_loading_does_not_restore_training_state(self):
        source_model = ScalarWorldModel()
        source_model.weight.data.fill_(7.0)
        source_optimizer = torch.optim.SGD(source_model.parameters(), lr=1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "best.pt"
            wm.save_checkpoint(
                checkpoint_path,
                epoch=9,
                model=source_model,
                optimizer=source_optimizer,
                best_val_loss=0.25,
                history={"train_loss": [1.0], "val_loss": [0.25]},
                metadata={"model_configuration": {"kind": "scalar"}},
            )

            target_model = ScalarWorldModel()
            metadata = wm.load_model_weights(
                checkpoint_path,
                target_model,
                torch.device("cpu"),
            )

        self.assertEqual(target_model.weight.item(), 7.0)
        self.assertEqual(metadata["model_configuration"], {"kind": "scalar"})

    def test_run_epoch_does_not_read_every_loss_back_to_the_cpu(self):
        model = ScalarWorldModel()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([
                DictDataset(target=1.0),
                DictDataset(target=1.0),
                DictDataset(target=1.0),
            ]),
            batch_size=1,
        )

        with torch.profiler.profile() as profiler:
            loss = wm.run_epoch(
                model,
                loader,
                torch.device("cpu"),
                log_every=0,
            )

        scalar_reads = sum(
            event.count
            for event in profiler.key_averages()
            if event.key == "aten::_local_scalar_dense"
        )

        self.assertEqual(loss, 1.0)
        self.assertLessEqual(scalar_reads, 2)

    def test_run_epoch_uses_requested_bfloat16_autocast(self):
        model = RecordingLinearWorldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = torch.utils.data.DataLoader(
            DictDataset(target=1.0),
            batch_size=1,
        )

        loss = wm.run_epoch(
            model,
            loader,
            torch.device("cpu"),
            optimizer=optimizer,
            precision="bf16",
        )

        self.assertEqual(model.output_dtype, torch.bfloat16)
        self.assertTrue(np.isfinite(loss))

    def test_best_checkpoint_is_only_replaced_when_validation_improves(self):
        self.assertTrue(hasattr(wm, "train_model"))

        model = ScalarWorldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        train_loader = torch.utils.data.DataLoader(
            DictDataset(target=1.0),
            batch_size=1,
        )
        val_loader = torch.utils.data.DataLoader(
            DictDataset(target=2.0),
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            history, best_val_loss = wm.train_model(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=torch.device("cpu"),
                epochs=2,
                validate_every=1,
                output_dir=output_dir,
                log_every=0,
            )

            best = torch.load(
                output_dir / "best.pt",
                map_location="cpu",
                weights_only=True,
            )
            last = torch.load(
                output_dir / "last.pt",
                map_location="cpu",
                weights_only=True,
            )

            self.assertEqual(best["epoch"], 0)
            self.assertEqual(last["epoch"], 1)
            self.assertEqual(best["model"]["weight"].item(), 2.0)
            self.assertEqual(last["model"]["weight"].item(), 0.0)
            self.assertEqual(best_val_loss, 0.0)
            self.assertEqual(len(history["train_loss"]), 2)
            self.assertEqual(len(history["val_loss"]), 2)

    def test_last_checkpoint_can_resume_from_the_next_epoch(self):
        model = ScalarWorldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        loader = torch.utils.data.DataLoader(
            DictDataset(target=1.0),
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            history, best_val_loss = wm.train_model(
                model=model,
                optimizer=optimizer,
                train_loader=loader,
                val_loader=loader,
                device=torch.device("cpu"),
                epochs=1,
                validate_every=1,
                output_dir=output_dir,
                log_every=0,
            )

            resumed_model = ScalarWorldModel()
            resumed_optimizer = torch.optim.SGD(
                resumed_model.parameters(),
                lr=1.0,
            )

            (
                start_epoch,
                resumed_best_val_loss,
                resumed_history,
                metadata,
            ) = wm.load_training_checkpoint(
                output_dir / "last.pt",
                resumed_model,
                resumed_optimizer,
                torch.device("cpu"),
            )

            self.assertEqual(start_epoch, 1)
            self.assertEqual(
                resumed_model.weight.item(),
                model.weight.item(),
            )
            self.assertEqual(resumed_best_val_loss, best_val_loss)
            self.assertEqual(resumed_history, history)
            self.assertEqual(metadata, {})

    def test_final_epoch_is_validated_even_when_not_on_interval(self):
        model = ScalarWorldModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        loader = torch.utils.data.DataLoader(
            DictDataset(target=1.0),
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            history, _ = wm.train_model(
                model=model,
                optimizer=optimizer,
                train_loader=loader,
                val_loader=loader,
                device=torch.device("cpu"),
                epochs=3,
                validate_every=2,
                output_dir=Path(temporary_directory),
                log_every=0,
            )

        self.assertEqual(
            [entry["epoch"] for entry in history["val_loss"]],
            [1, 2],
        )

    def test_resume_metadata_rejects_changed_episode_split(self):
        saved_metadata = {
            "model_configuration": {"num_hist": 3},
            "train_ids": [0, 1],
            "val_ids": [2],
            "test_ids": [3],
            "normalize_actions": True,
            "action_mean": torch.zeros(2),
            "action_std": torch.ones(2),
        }
        current_metadata = {
            **saved_metadata,
            "train_ids": [0, 2],
            "val_ids": [1],
        }

        with self.assertRaises(ValueError):
            wm.validate_resume_metadata(
                saved_metadata,
                current_metadata,
            )

    def test_resume_metadata_rejects_changed_training_objective(self):
        saved_metadata = {
            "model_configuration": {"num_hist": 3},
            "train_ids": [0, 1],
            "val_ids": [2],
            "test_ids": [3],
            "normalize_actions": False,
            "action_mean": None,
            "action_std": None,
            "training_objective": {
                "rollout_steps": 1,
                "rollout_loss_weight": 0.0,
            },
        }
        current_metadata = {
            **saved_metadata,
            "training_objective": {
                "rollout_steps": 3,
                "rollout_loss_weight": 1.0,
            },
        }

        with self.assertRaises(ValueError):
            wm.validate_resume_metadata(saved_metadata, current_metadata)

    def test_finetuning_accepts_a_new_objective_with_same_data_contract(self):
        saved_metadata = {
            "model_configuration": {"num_hist": 3},
            "train_ids": [0, 1],
            "val_ids": [2],
            "test_ids": [3],
            "normalize_actions": True,
            "action_mean": torch.tensor([1.0]),
            "action_std": torch.tensor([2.0]),
            "training_objective": {
                "rollout_steps": 1,
                "rollout_loss_weight": 0.0,
            },
        }
        current_metadata = {
            **saved_metadata,
            "training_objective": {
                "rollout_steps": 3,
                "rollout_loss_weight": 1.0,
            },
        }

        wm.validate_finetune_metadata(saved_metadata, current_metadata)

    def test_finetuning_rejects_changed_action_normalization(self):
        saved_metadata = {
            "model_configuration": {"num_hist": 3},
            "train_ids": [0, 1],
            "val_ids": [2],
            "test_ids": [3],
            "normalize_actions": True,
            "action_mean": torch.tensor([1.0]),
            "action_std": torch.tensor([2.0]),
        }
        current_metadata = {
            **saved_metadata,
            "action_mean": torch.tensor([3.0]),
        }

        with self.assertRaises(ValueError):
            wm.validate_finetune_metadata(saved_metadata, current_metadata)


class CommandLineTrainingTest(unittest.TestCase):
    def test_small_training_run_creates_last_and_best_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            output_dir = root / "run"
            latent_dir.mkdir()
            trajectory_dir.mkdir()

            for episode_id in range(4):
                np.save(
                    latent_dir / f"episode_{episode_id:04d}.npy",
                    np.zeros((5, 2, 3), dtype=np.float16),
                )
                torch.save(
                    {"macro_actions": torch.zeros(4, 2)},
                    trajectory_dir / f"episode_{episode_id:04d}.pt",
                )

            command = [
                sys.executable,
                "-m",
                "src.mylonas_ijepa_wm.wm",
                    "--latent-dir", str(latent_dir),
                    "--trajectory-dir", str(trajectory_dir),
                    "--output-dir", str(output_dir),
                    "--num-episodes", "4",
                    "--train-episodes", "2",
                    "--val-episodes", "1",
                    "--test-episodes", "1",
                    "--num-hist", "2",
                    "--num-patches", "2",
                    "--ijepa-dim", "3",
                    "--action-dim", "2",
                    "--action-embed-dim", "4",
                    "--embed-dim", "8",
                    "--num-blocks", "1",
                    "--num-heads", "2",
                    "--mlp-dim", "16",
                    "--batch-size", "2",
                    "--epochs", "1",
                    "--device", "cpu",
                    "--num-workers", "0",
                    "--log-every", "0",
                    "--preload-latents",
            ]

            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + result.stderr,
            )
            self.assertTrue((output_dir / "last.pt").is_file())
            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertIn("test_loss=", result.stdout)

            resume_result = subprocess.run(
                command + [
                    "--epochs", "2",
                    "--resume",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                resume_result.returncode,
                0,
                msg=resume_result.stdout + resume_result.stderr,
            )
            resumed_checkpoint = torch.load(
                output_dir / "last.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(resumed_checkpoint["epoch"], 1)

            finetune_output_dir = root / "run_k3"
            finetune_command = command.copy()
            output_value_index = finetune_command.index("--output-dir") + 1
            finetune_command[output_value_index] = str(finetune_output_dir)
            finetune_command.extend([
                "--rollout-steps", "3",
                "--rollout-loss-weight", "1.0",
                "--finetune-from", str(output_dir / "best.pt"),
                "--learning-rate", "5e-5",
            ])

            finetune_result = subprocess.run(
                finetune_command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                finetune_result.returncode,
                0,
                msg=finetune_result.stdout + finetune_result.stderr,
            )
            finetune_checkpoint = torch.load(
                finetune_output_dir / "last.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(finetune_checkpoint["epoch"], 0)
            self.assertEqual(
                finetune_checkpoint["metadata"]["training_objective"],
                {"rollout_steps": 3, "rollout_loss_weight": 1.0},
            )
            self.assertEqual(
                len(finetune_checkpoint["history"]["train_loss"]),
                1,
            )
            self.assertIn("train_rollout=", finetune_result.stdout)

            test_metrics = torch.load(
                finetune_output_dir / "test_metrics.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertIn("test_one_step_loss", test_metrics)
            self.assertIn("test_rollout_loss", test_metrics)


if __name__ == "__main__":
    unittest.main()
