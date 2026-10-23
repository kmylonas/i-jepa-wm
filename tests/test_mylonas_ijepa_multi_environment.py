import argparse
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


if __name__ == "__main__":
    unittest.main()
