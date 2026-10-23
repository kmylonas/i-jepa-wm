import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import embed_frames, wm, wm_proprio
from src.mylonas_ijepa_wm.proprio_vit import ProprioViT


class EmbedFramesTimingTest(unittest.TestCase):
    def test_dense_embeddings_support_offset_training_windows(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return images[:, :1, :1, :1].reshape(-1, 1, 1)

        frames = torch.arange(16, dtype=torch.uint8).view(16, 1, 1, 1).expand(16, 224, 224, 3)
        episode = {
            "frames": frames,
            "actions": torch.arange(30, dtype=torch.float32).reshape(15, 2),
            "states": torch.arange(64, dtype=torch.float32).reshape(16, 4),
            "desired_goal": torch.zeros(2),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embed_frames.process_episode(
                episode, TinyEncoder(), torch.device("cpu"), "episode_0000", root,
                embedding_frame_stride=1,
            )
            latents = np.load(root / "latents" / "episode_0000.npy")
            trajectory = torch.load(root / "trajectories" / "episode_0000.pt", weights_only=True)
            self.assertEqual(len(latents), 16)
            self.assertEqual(trajectory["embedding_frame_stride"], 1)
            torch.testing.assert_close(trajectory["frame_indices"], torch.arange(16))
            torch.testing.assert_close(trajectory["sampled_states"], episode["states"])

            dataset = wm_proprio.ProprioWorldModelDataset(
                root / "latents", root / "trajectories", [0], num_hist=2,
                rollout_steps=1, state_dim=4,
            )
            self.assertEqual(len(dataset), 6)
            sample = dataset[1]
            np.testing.assert_array_equal(sample["latent_window"].numpy(), latents[[1, 6, 11]])
            torch.testing.assert_close(sample["state_window"], episode["states"][[1, 6, 11]])
            torch.testing.assert_close(
                sample["actions"],
                torch.stack((episode["actions"][1:6].flatten(), episode["actions"][6:11].flatten())),
            )

            visual_dataset = wm.WorldModelDataset(root / "latents", root / "trajectories", [0], num_hist=2)
            torch.testing.assert_close(visual_dataset[1]["actions"], sample["actions"])

            multistep = wm_proprio.ProprioWorldModelDataset(
                root / "latents", root / "trajectories", [0], num_hist=2,
                rollout_steps=2, state_dim=4,
            )
            self.assertEqual(len(multistep), 1)
            torch.testing.assert_close(multistep[0]["state_window"], episode["states"][[0, 5, 10, 15]])
            torch.testing.assert_close(multistep[0]["actions"][2], episode["actions"][10:15].flatten())

            expected_groups = torch.stack([episode["actions"][start:start + 5].flatten() for start in range(11)])
            mean, std = wm.compute_action_statistics(root / "trajectories", [0])
            torch.testing.assert_close(mean, expected_groups.mean(dim=0))
            torch.testing.assert_close(std, expected_groups.std(dim=0, unbiased=False))

    def test_preserves_physics_timing_and_action_alignment(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return torch.zeros((len(images), 1, 1), device=images.device)

        episode = {
            "frames": torch.zeros((101, 224, 224, 3), dtype=torch.uint8),
            "actions": torch.arange(200, dtype=torch.float32).reshape(100, 2),
            "states": torch.arange(404, dtype=torch.float32).reshape(101, 4),
            "desired_goal": torch.zeros(2),
            "physics_frame_skip": 5,
            "control_dt_seconds": 0.05,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            embed_frames.process_episode(
                episode=episode,
                encoder=TinyEncoder(),
                device=torch.device("cpu"),
                episode_name="episode_0000",
                output_directory=output,
            )
            trajectory = torch.load(
                output / "trajectories" / "episode_0000.pt",
                map_location="cpu",
                weights_only=True,
            )
            latents = np.load(output / "latents" / "episode_0000.npy")

            self.assertEqual(trajectory["physics_frame_skip"], 5)
            self.assertAlmostEqual(trajectory["control_dt_seconds"], 0.05)
            self.assertEqual(latents.shape[0], 21)
            torch.testing.assert_close(trajectory["sampled_states"], episode["states"][::5])
            torch.testing.assert_close(
                trajectory["macro_actions"][0], episode["actions"][:5].flatten()
            )


class ProprioDatasetTest(unittest.TestCase):
    def test_checkpoint_timing_can_be_read_before_loading_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            torch.save({"metadata": {"physics_frame_skip": 5}}, checkpoint)
            self.assertEqual(
                wm_proprio.read_checkpoint_physics_frame_skip(checkpoint), 5
            )

    def test_training_rejects_mixed_physics_timing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trajectory_dir = Path(temporary_directory)
            torch.save({"physics_frame_skip": 5}, trajectory_dir / "episode_0000.pt")
            torch.save({"physics_frame_skip": 5}, trajectory_dir / "episode_0001.pt")
            self.assertEqual(
                wm_proprio.read_physics_frame_skip(trajectory_dir, [0, 1]), 5
            )
            torch.save({"physics_frame_skip": 1}, trajectory_dir / "episode_0001.pt")
            with self.assertRaisesRegex(ValueError, "physics_frame_skip"):
                wm_proprio.read_physics_frame_skip(trajectory_dir, [0, 1])

    def test_sample_aligns_latents_states_and_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()
            latents = np.arange(5, dtype=np.float16).reshape(5, 1, 1)
            states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
            actions = torch.arange(4, dtype=torch.float32).reshape(4, 1)
            np.save(latent_dir / "episode_0007.npy", latents)
            torch.save(
                {"sampled_states": states, "macro_actions": actions},
                trajectory_dir / "episode_0007.pt",
            )

            dataset = wm_proprio.ProprioWorldModelDataset(
                latent_dir=latent_dir,
                trajectory_dir=trajectory_dir,
                episode_ids=[7],
                num_hist=2,
                rollout_steps=2,
                state_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                state_std=torch.ones(4),
            )
            sample = dataset[1]

            torch.testing.assert_close(
                sample["latent_window"].flatten().float(),
                torch.tensor([1.0, 2.0, 3.0, 4.0]),
            )
            torch.testing.assert_close(
                sample["state_window"],
                states[1:5] - torch.tensor([1.0, 2.0, 3.0, 4.0]),
            )
            torch.testing.assert_close(sample["actions"].flatten(), actions[1:4].flatten())

    def test_rejects_state_sequence_not_aligned_with_latents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "latents").mkdir()
            (root / "trajectories").mkdir()
            np.save(root / "latents" / "episode_0000.npy", np.zeros((5, 1, 1), dtype=np.float16))
            torch.save(
                {
                    "sampled_states": torch.zeros(4, 4),
                    "macro_actions": torch.zeros(4, 2),
                },
                root / "trajectories" / "episode_0000.pt",
            )

            with self.assertRaisesRegex(ValueError, "sampled_states"):
                wm_proprio.ProprioWorldModelDataset(
                    root / "latents",
                    root / "trajectories",
                    episode_ids=[0],
                    num_hist=2,
                )


class ProprioModelTest(unittest.TestCase):
    def test_one_step_training_uses_state_target_and_one_model_call(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, latents, states, actions):
                self.calls += 1
                return latents + 1.0, states + 1.0

        model = CountingModel()
        latents = torch.zeros(1, 3, 1, 1)
        states = torch.zeros(1, 3, 4)
        states[:, 2] = 4.0
        actions = torch.zeros(1, 2, 1)

        metrics = wm_proprio.compute_world_model_loss(
            model,
            latents,
            states,
            actions,
            num_hist=2,
            proprio_loss_weight=0.1,
        )

        self.assertEqual(model.calls, 1)
        self.assertAlmostEqual(float(metrics["one_step_visual_loss"]), 1.0)
        self.assertAlmostEqual(float(metrics["one_step_proprio_loss"]), 5.0)
        self.assertAlmostEqual(float(metrics["loss"]), 1.5, places=5)

    def test_predicts_visual_and_state_and_uses_proprio_input(self):
        torch.manual_seed(0)
        model = ProprioViT(
            num_patches=2,
            num_hist=2,
            num_t_blocks=1,
            ijepa_dim=3,
            action_dim=2,
            action_embed_dim=4,
            proprio_dim=4,
            proprio_embed_dim=4,
            embed_dim=8,
            num_heads=2,
            mlp_dim=16,
            dropout=0.0,
        ).eval()
        latents = torch.randn(1, 2, 2, 3)
        actions = torch.randn(1, 2, 2)
        states = torch.randn(1, 2, 4)

        predicted_visual, predicted_state = model(latents, states, actions)
        changed_visual, changed_state = model(latents, states + 2.0, actions)

        self.assertEqual(tuple(predicted_visual.shape), (1, 2, 2, 3))
        self.assertEqual(tuple(predicted_state.shape), (1, 2, 4))
        self.assertFalse(torch.allclose(predicted_visual, changed_visual))
        self.assertFalse(torch.allclose(predicted_state, changed_state))

    def test_rollout_uses_predicted_state_after_first_step(self):
        class StateDrivenModel(torch.nn.Module):
            def forward(self, latents, states, actions):
                next_state = states + actions[..., :1]
                next_visual = next_state[..., :1].unsqueeze(-2)
                return next_visual, next_state

        visual, state = wm_proprio.autoregressive_rollout(
            StateDrivenModel(),
            latent_history=torch.zeros(1, 1, 1, 1),
            state_history=torch.zeros(1, 1, 4),
            actions=torch.ones(1, 2, 1),
        )

        torch.testing.assert_close(visual.flatten(), torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(state[:, :, 0].flatten(), torch.tensor([1.0, 2.0]))


class ProprioRolloutEvaluationTest(unittest.TestCase):
    def test_teacher_forcing_and_autoregression_use_their_respective_state_histories(self):
        from src.mylonas_ijepa_wm import evaluate_rollouts_proprio

        class OneSequence(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {
                    "latent_window": torch.tensor([0.0, 1.0, 3.0, 6.0]).reshape(4, 1, 1),
                    "state_window": torch.tensor([
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0, 0.0],
                        [8.0, 0.0, 0.0, 0.0],
                    ]),
                    "actions": torch.tensor([[1.0], [2.0], [3.0]]),
                }

        class StateAndLatentDrivenModel(torch.nn.Module):
            def forward(self, latents, states, actions):
                visual = torch.zeros_like(latents)
                state = torch.zeros_like(states)
                visual[:, -1, 0, 0] = (
                    latents[:, -1, 0, 0]
                    + states[:, -1, 0]
                    + actions[:, -1, 0]
                )
                state[:, -1] = states[:, -1]
                state[:, -1, 0] += actions[:, -1, 0]
                return visual, state

        metrics = evaluate_rollouts_proprio.evaluate_model(
            model=StateAndLatentDrivenModel(),
            data_loader=torch.utils.data.DataLoader(OneSequence(), batch_size=1),
            device=torch.device("cpu"),
            num_hist=2,
            proprio_loss_weight=2.0,
        )

        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(metrics["per_horizon"], [
            {
                "horizon": 1,
                "autoregressive_mse": 1.0,
                "teacher_forced_mse": 1.0,
                "persistence_mse": 4.0,
                "autoregressive_proprio_mse": 1.0,
                "teacher_forced_proprio_mse": 1.0,
                "persistence_proprio_mse": 4.0,
                "autoregressive_combined_mse": 3.0,
                "teacher_forced_combined_mse": 3.0,
                "persistence_combined_mse": 12.0,
            },
            {
                "horizon": 2,
                "autoregressive_mse": 16.0,
                "teacher_forced_mse": 25.0,
                "persistence_mse": 25.0,
                "autoregressive_proprio_mse": 1.0,
                "teacher_forced_proprio_mse": 0.0,
                "persistence_proprio_mse": 12.25,
                "autoregressive_combined_mse": 18.0,
                "teacher_forced_combined_mse": 25.0,
                "persistence_combined_mse": 49.5,
            },
        ])
        self.assertEqual(metrics["proprio_loss_weight"], 2.0)

    def test_cli_saves_metrics_and_latent_plot_for_checkpoint_test_split(self):
        from src.mylonas_ijepa_wm import evaluate_rollouts_proprio

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()
            np.save(latent_dir / "episode_0007.npy", np.zeros((5, 2, 3), dtype=np.float16))
            torch.save(
                {
                    "sampled_states": torch.zeros(5, 4),
                    "macro_actions": torch.zeros(4, 2),
                },
                trajectory_dir / "episode_0007.pt",
            )
            configuration = dict(
                num_patches=2, num_hist=2, num_t_blocks=1, ijepa_dim=3,
                action_dim=2, action_embed_dim=4, proprio_dim=4,
                proprio_embed_dim=4, embed_dim=8, num_heads=2,
                mlp_dim=16, dropout=0.0,
            )
            model = ProprioViT(**configuration)
            checkpoint = root / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": 2,
                    "metadata": {
                        "model_type": "proprio_v1",
                        "camera": "topdown",
                        "proprio_fields": ("x", "y", "vx", "vy"),
                        "model_configuration": configuration,
                        "test_ids": [7],
                        "normalize_actions": False,
                        "state_mean": torch.zeros(4),
                        "state_std": torch.ones(4),
                        "proprio_loss_weight": 2.0,
                    },
                },
                checkpoint,
            )
            output = root / "metrics.json"
            plot = root / "metrics.png"
            evaluate_rollouts_proprio.main([
                "--checkpoint", str(checkpoint),
                "--latent-dir", str(latent_dir),
                "--trajectory-dir", str(trajectory_dir),
                "--split", "test",
                "--horizon", "2",
                "--batch-size", "1",
                "--num-workers", "0",
                "--device", "cpu",
                "--precision", "fp32",
                "--output", str(output),
                "--plot-output", str(plot),
            ])

            saved = json.loads(output.read_text())
            self.assertEqual(saved["episode_ids"], [7])
            self.assertEqual(saved["checkpoint_epoch"], 2)
            self.assertEqual([row["horizon"] for row in saved["per_horizon"]], [1, 2])
            self.assertEqual(saved["proprio_loss_weight"], 2.0)
            for row in saved["per_horizon"]:
                for method in ("autoregressive", "teacher_forced", "persistence"):
                    self.assertAlmostEqual(
                        row[f"{method}_combined_mse"],
                        row[f"{method}_mse"] + 2.0 * row[f"{method}_proprio_mse"],
                    )
            self.assertEqual(plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


class ProprioTrajectoryVisualizationTest(unittest.TestCase):
    def test_state_paths_use_real_teacher_forced_and_predicted_histories_in_maze_units(self):
        from src.mylonas_ijepa_wm import visualize_proprio_rollouts

        class StateDrivenModel(torch.nn.Module):
            def forward(self, latents, states, actions):
                visual = torch.zeros_like(latents)
                predicted_states = torch.zeros_like(states)
                predicted_states[:, -1] = states[:, -1]
                predicted_states[:, -1, (0, 2)] += actions[:, -1, 0].unsqueeze(-1)
                return visual, predicted_states

        sample = {
            "latent_window": torch.zeros(4, 1, 1),
            "state_window": torch.tensor([
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [5.0, 0.0, 5.0, 0.0],
                [8.0, 0.0, 8.0, 0.0],
            ]),
            "actions": torch.tensor([[1.0], [2.0], [3.0]]),
        }
        trajectory = visualize_proprio_rollouts.predict_state_trajectory(
            model=StateDrivenModel(),
            sample=sample,
            num_hist=2,
            episode_id=7,
            state_mean=torch.tensor([10.0, 20.0, 30.0, 40.0]),
            state_std=torch.tensor([2.0, 3.0, 4.0, 5.0]),
            device=torch.device("cpu"),
        )

        self.assertEqual(trajectory["episode_id"], 7)
        self.assertEqual(trajectory["start"], 0)
        torch.testing.assert_close(
            trajectory["true"][:, (0, 2)],
            torch.tensor([[12.0, 34.0], [20.0, 50.0], [26.0, 62.0]]),
        )
        torch.testing.assert_close(
            trajectory["autoregressive"][:, (0, 2)],
            torch.tensor([[12.0, 34.0], [16.0, 42.0], [22.0, 54.0]]),
        )
        torch.testing.assert_close(
            trajectory["teacher_forced"][:, (0, 2)],
            torch.tensor([[12.0, 34.0], [16.0, 42.0], [26.0, 62.0]]),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "positions.png"
            visualize_proprio_rollouts.save_trajectory_plot([trajectory], output)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
