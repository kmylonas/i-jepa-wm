import argparse
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.mylonas_ijepa_wm import embed_frames, generate_episodes, plan_proprio, wm_proprio
from src.mylonas_ijepa_wm.environment_adapters import StepResult
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig, load_environment_config
from src.mylonas_ijepa_wm.planning import CEMResult


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        return torch.zeros(len(images), 1, 1, device=images.device)


class FakeWall:
    def __init__(self):
        self.goal = np.array([2, 0], dtype=np.float32)
        self.layout = (32, 30)
        self.action_low = np.array([-1, -1], dtype=np.float32)
        self.action_high = np.array([1, 1], dtype=np.float32)
        self.control_dt_seconds = None
        self.position = np.array([0, 0], dtype=np.float32)

    def frame(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def reset(self, seed):
        self.position[:] = 0
        return StepResult(self.position.copy(), self.frame(), False)

    def step(self, action):
        self.position += np.asarray(action, dtype=np.float32)
        return StepResult(self.position.copy(), self.frame(), self.position[0] >= 2)

    def render_goal(self):
        return self.frame()

    def render_video(self):
        return self.frame()

    def close(self):
        pass


class MultiEnvironmentTest(unittest.TestCase):
    def test_dino_wall_actions_have_truncated_lengths_and_persistent_directions(self):
        actions = generate_episodes.sample_dino_wall_actions(
            np.random.default_rng(12), 300
        )
        lengths = np.linalg.norm(actions, axis=1)
        headings = np.arctan2(actions[:, 1], actions[:, 0])
        self.assertEqual(actions.shape, (300, 2))
        self.assertTrue(np.all((lengths >= 0.2) & (lengths <= 1.8)))
        self.assertGreater(float(np.abs(actions).max()), 1.0)
        self.assertGreater(float(np.cos(np.diff(headings)).mean()), 0.7)

    def test_dino_wall_policy_is_recorded_in_raw_episode_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(generate_episodes, "create_adapter", return_value=FakeWall()):
                generate_episodes.main([
                    "--env", "wall", "--output-dir", directory,
                    "--num-episodes", "1", "--steps-per-episode", "50",
                    "--wall-policy", "dino",
                ])
            root = Path(directory)
            episode = torch.load(root / "episode_0000.pt", weights_only=True)
            self.assertEqual(episode["collection_policy"], "dino")
            self.assertEqual(episode["actions"].shape, (50, 2))
            self.assertEqual(episode["frames"].shape[0], 51)
            self.assertEqual(episode["states"].shape[0], 51)
            manifest = json.loads((root / "metadata.json").read_text())
            self.assertEqual(manifest["wall_policy"], "dino")
            self.assertEqual(
                episode["collection_policy_parameters"], manifest["wall_policy_parameters"]
            )
            self.assertEqual(manifest["wall_policy_parameters"]["length_max"], 1.8)
            self.assertIsNone(manifest["wall_policy_parameters"]["component_clip"])
            embed_frames.process_episode(
                episode, TinyEncoder(), torch.device("cpu"), "episode_0000",
                root / "cache", environment="wall", actions_per_model_step=5,
                embedding_frame_stride=1, encoder_batch_size=8,
            )
            trajectory = torch.load(
                root / "cache/trajectories/episode_0000.pt", weights_only=True,
            )
            self.assertEqual(
                trajectory["collection_policy_parameters"], manifest["wall_policy_parameters"]
            )

    def test_unbounded_wall_planning_can_execute_actions_above_training_limit(self):
        class ActionSumModel(torch.nn.Module):
            num_hist = 1

            def forward(self, latents, states, actions):
                predicted = states.clone()
                predicted[:, -1, :] += actions[:, -1].reshape(-1, 2, 2).sum(dim=1)
                return latents, predicted

        adapter = FakeWall()
        adapter.goal = np.array([5., 0.], dtype=np.float32)
        config = EnvironmentConfig(env="wall", actions_per_model_step=2)
        metadata = {
            "state_mean": torch.zeros(2), "state_std": torch.ones(2),
            "action_mean": None, "action_std": None,
        }
        args = argparse.Namespace(
            cost="position", max_macro_steps=1, planning_horizon=1,
            actions_per_plan=1, population=256, num_elites=32,
            cem_iterations=5, candidate_batch_size=64, min_std=0.05,
            unbounded_actions=True,
        )
        with patch.object(plan_proprio.visual, "encode_frame", return_value=torch.zeros(1, 1)):
            record = plan_proprio.run_episode(
                adapter, config, object(), ActionSumModel(), metadata, args,
                torch.device("cpu"), "fp32", 7, None,
            )
        self.assertGreater(np.abs(record["macro_actions"]).max(), 1.0)

    def test_wall_planning_limits_each_executed_action_length(self):
        class ActionSumModel(torch.nn.Module):
            num_hist = 1

            def forward(self, latents, states, actions):
                predicted = states.clone()
                predicted[:, -1, :] += actions[:, -1].reshape(-1, 2, 2).sum(dim=1)
                return latents, predicted

        args = plan_proprio.parse_args([
            "--env", "wall", "--world-model-checkpoint", "unused.pt",
            "--output", "unused.json", "--cost", "position",
            "--unbounded-actions", "--max-action-length", "0.7",
            "--max-macro-steps", "1", "--planning-horizon", "1",
            "--population", "256", "--num-elites", "32",
            "--cem-iterations", "5", "--candidate-batch-size", "64",
        ])
        adapter = FakeWall()
        adapter.goal = np.array([5., 0.], dtype=np.float32)
        config = EnvironmentConfig(env="wall", actions_per_model_step=2)
        metadata = {
            "state_mean": torch.zeros(2), "state_std": torch.ones(2),
            "action_mean": None, "action_std": None,
        }
        with patch.object(plan_proprio.visual, "encode_frame", return_value=torch.zeros(1, 1)):
            record = plan_proprio.run_episode(
                adapter, config, object(), ActionSumModel(), metadata, args,
                torch.device("cpu"), "fp32", 7, None,
            )

        lengths = np.linalg.norm(np.asarray(record["macro_actions"]).reshape(-1, 2), axis=1)
        self.assertLessEqual(float(lengths.max()), 0.700001)
        self.assertGreater(float(lengths.max()), 0.5)
        self.assertAlmostEqual(
            record["final_planner_cost"], record["final_goal_distance"], places=5,
        )

    def test_action_length_limit_must_be_positive_and_finite(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit):
                    plan_proprio.parse_args([
                        "--world-model-checkpoint", "unused.pt", "--output", "unused.json",
                        "--max-action-length", value,
                    ])
                self.assertIn("positive finite", stderr.getvalue())

    def test_wall_cli_can_restore_best_sample_selection(self):
        args = plan_proprio.parse_args([
            "--env", "wall", "--world-model-checkpoint", "unused.pt",
            "--output", "unused.json", "--cost", "latent",
            "--cem-selection", "best",
        ])
        self.assertEqual(args.cem_selection, "best")

    def test_collector_can_fill_a_dataset_in_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            common = [
                "--env", "wall", "--output-dir", directory,
                "--num-episodes", "3", "--steps-per-episode", "5",
                "--wall-policy", "mixed",
            ]
            with patch.object(generate_episodes, "create_adapter", return_value=FakeWall()):
                generate_episodes.main([*common, "--start-index", "0", "--end-index", "1"])
                self.assertEqual(len(list(Path(directory).glob("episode_*.pt"))), 1)
                generate_episodes.main([
                    *common, "--start-index", "1", "--end-index", "3", "--resume",
                ])
            self.assertEqual(len(list(Path(directory).glob("episode_*.pt"))), 3)
            policies = [
                torch.load(Path(directory) / f"episode_{index:04d}.pt", weights_only=True)["collection_policy"]
                for index in range(3)
            ]
            self.assertEqual(policies, ["random", "door", "random"])

    def test_config_rejects_mismatched_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.json"
            path.write_text('{"env":"wall","door_location":20}')
            with self.assertRaisesRegex(ValueError, "disagree"):
                load_environment_config("pointmaze", path)

    def test_wall_episode_embedding_and_training_dataset_align(self):
        config = EnvironmentConfig(env="wall", actions_per_model_step=2)
        raw = generate_episodes.collect_episode(FakeWall(), config, 0, 4, 0)
        self.assertEqual(raw["states"].shape, (5, 2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = TinyEncoder()
            embed_frames.process_episode(
                raw, encoder, torch.device("cpu"), "episode_0000", root,
                environment="wall", actions_per_model_step=2, encoder_batch_size=2,
            )
            self.assertEqual(encoder.batch_sizes, [2, 1])
            trajectory = torch.load(
                root / "trajectories/episode_0000.pt", weights_only=True
            )
            self.assertEqual(trajectory["macro_actions"].shape, (2, 4))
            self.assertEqual(trajectory["sampled_states"].shape, (3, 2))
            dataset = wm_proprio.ProprioWorldModelDataset(
                root / "latents", root / "trajectories", [0],
                num_hist=2, state_dim=2,
            )
            self.assertEqual(dataset[0]["state_window"].shape, (3, 2))

    def test_generic_planner_uses_wall_action_and_state_dimensions(self):
        adapter = FakeWall()
        config = EnvironmentConfig(env="wall", actions_per_model_step=2)
        model = argparse.Namespace(num_hist=2)
        metadata = {
            "state_mean": torch.zeros(2), "state_std": torch.ones(2),
            "action_mean": None, "action_std": None,
        }
        args = argparse.Namespace(
            cost="position", max_macro_steps=2, planning_horizon=1,
            actions_per_plan=1, population=2, num_elites=1,
            cem_iterations=1, candidate_batch_size=2, min_std=0.05,
        )
        with patch.object(plan_proprio.visual, "encode_frame", return_value=torch.zeros(1, 1)):
            with patch.object(plan_proprio.planner, "plan_action") as plan:
                plan.return_value = CEMResult(
                    action_sequence=torch.tensor([[1., 0., 1., 0.]]), cost=0.0
                )
                record = plan_proprio.run_episode(
                    adapter, config, object(), model, metadata, args,
                    torch.device("cpu"), "fp32", 0, None,
                )
        self.assertTrue(record["success"])
        self.assertEqual(record["proprio_states"][-1], [2., 0.])
        self.assertEqual(plan.call_args.kwargs["action_history"].shape, (1, 4))

    def test_wall_planner_replans_with_one_frame_history(self):
        class OneFrameModel(torch.nn.Module):
            num_hist = 1

            def forward(self, latents, states, actions):
                next_states = states.clone()
                next_states[:, -1, 0] += actions[:, -1, 0]
                return latents, next_states

        adapter = FakeWall()
        adapter.goal = np.array([100., 0.], dtype=np.float32)
        config = EnvironmentConfig(env="wall", actions_per_model_step=2)
        metadata = {
            "state_mean": torch.zeros(2), "state_std": torch.ones(2),
            "action_mean": None, "action_std": None,
        }
        args = argparse.Namespace(
            cost="position", max_macro_steps=2, planning_horizon=1,
            actions_per_plan=1, population=2, num_elites=1,
            cem_iterations=1, candidate_batch_size=2, min_std=0.05,
        )
        with patch.object(plan_proprio.visual, "encode_frame", return_value=torch.zeros(1, 1)):
            record = plan_proprio.run_episode(
                adapter, config, object(), OneFrameModel(), metadata, args,
                torch.device("cpu"), "fp32", 0, None,
            )

        self.assertEqual(record["macro_steps"], 2)


if __name__ == "__main__":
    unittest.main()
