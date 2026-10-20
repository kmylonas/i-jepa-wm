import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze
from src.mylonas_ijepa_wm import plan_pointmaze_proprio
from src.mylonas_ijepa_wm.planning import CEMResult
from src.mylonas_ijepa_wm.proprio_vit import ProprioViT


class StateDrivenModel(torch.nn.Module):
    num_hist = 1

    def forward(self, latents, states, actions):
        next_state = states + actions[..., :1]
        next_visual = next_state[..., :1].unsqueeze(-2)
        return next_visual, next_state


class ProprioPlanningTest(unittest.TestCase):
    def test_cli_loads_checkpoint_and_writes_episode_results(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images), 256, 1280, device=images.device)

        configuration = dict(
            num_patches=256, num_hist=1, num_t_blocks=1,
            ijepa_dim=1280, action_dim=10, action_embed_dim=4,
            proprio_dim=4, proprio_embed_dim=4, embed_dim=8,
            num_heads=2, mlp_dim=16, dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            encoder_checkpoint = root / "encoder.pt"
            output = root / "planning.json"
            encoder_checkpoint.touch()
            torch.save({
                "epoch": 0,
                "model": ProprioViT(**configuration).state_dict(),
                "metadata": {
                    "model_type": "proprio_v1",
                    "camera": "topdown",
                    "proprio_fields": ("x", "y", "vx", "vy"),
                    "model_configuration": configuration,
                    "state_mean": torch.zeros(4),
                    "state_std": torch.ones(4),
                    "action_mean": None,
                    "action_std": None,
                    "physics_frame_skip": 5,
                },
            }, checkpoint)

            for cost in ("latent", "position"):
                with self.subTest(cost=cost):
                    with patch.object(plan_pointmaze, "load_frozen_ijepa_encoder", return_value=TinyEncoder()):
                        plan_pointmaze_proprio.main([
                            "--encoder-checkpoint", str(encoder_checkpoint),
                            "--world-model-checkpoint", str(checkpoint),
                            "--output", str(output), "--cost", cost,
                            "--num-episodes", "1", "--max-macro-steps", "1",
                            "--planning-horizon", "1", "--population", "2",
                            "--num-elites", "1", "--cem-iterations", "1",
                            "--candidate-batch-size", "2", "--device", "cpu",
                        ])

                    result = json.loads(output.read_text())
                    self.assertEqual(result["configuration"]["camera"], "topdown")
                    self.assertEqual(result["configuration"]["physics_frame_skip"], 5)
                    self.assertEqual(result["configuration"]["cost"], cost)
                    self.assertEqual(result["episodes"][0]["cost_mode"], cost)
                    self.assertEqual(len(result["episodes"]), 1)

    def test_position_cost_uses_terminal_unnormalized_xy(self):
        predicted_states = torch.tensor([
            [[9.0, 9.0, 0.0, 0.0], [1.0, 1.0, 100.0, -100.0]],
            [[1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 0.0, 0.0]],
        ])
        costs = plan_pointmaze_proprio.compute_terminal_position_cost(
            predicted_states=predicted_states,
            goal_position=torch.tensor([12.0, 24.0]),
            state_mean=torch.tensor([10.0, 20.0, 0.0, 0.0]),
            state_std=torch.tensor([2.0, 4.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(costs, torch.tensor([0.0, 2.0]))

    def test_position_planner_scores_world_model_states(self):
        plan = plan_pointmaze_proprio.plan_action(
            model=StateDrivenModel(),
            latent_history=torch.zeros(1, 1, 1),
            state_history=torch.zeros(1, 4),
            action_history=torch.empty(0, 1),
            target_latent=None,
            action_low=torch.tensor([0.0]),
            action_high=torch.tensor([2.0]),
            device=torch.device("cpu"),
            planning_horizon=1,
            population=64,
            num_elites=8,
            num_iterations=4,
            candidate_batch_size=16,
            cost="position",
            goal_position=torch.tensor([13.0, 26.0]),
            state_mean=torch.tensor([10.0, 20.0, 0.0, 0.0]),
            state_std=torch.tensor([2.0, 4.0, 1.0, 1.0]),
            generator=torch.Generator().manual_seed(0),
        )
        self.assertAlmostEqual(plan.action_sequence[0, 0].item(), 1.5, delta=0.2)

    def test_candidate_rollout_updates_predicted_proprioception(self):
        visual, states = plan_pointmaze_proprio.rollout_action_sequences(
            model=StateDrivenModel(),
            latent_history=torch.zeros(1, 1, 1, 1),
            state_history=torch.zeros(1, 1, 4),
            action_history=torch.empty(0, 1),
            candidate_actions=torch.tensor([[[1.0], [1.0]], [[2.0], [0.0]]]),
            device=torch.device("cpu"),
            chunk_size=1,
        )

        torch.testing.assert_close(
            visual[:, :, 0, 0], torch.tensor([[1.0, 2.0], [2.0, 2.0]])
        )
        torch.testing.assert_close(states[:, :, 0], torch.tensor([[1.0, 2.0], [2.0, 2.0]]))

    def test_episode_uses_real_proprioception_and_topdown_observations(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1).expand(-1, 2, 3)

        class TinyModel(torch.nn.Module):
            num_hist = 2

            def forward(self, latents, states, actions):
                visual = latents + 0.001 * actions[..., :1].unsqueeze(-1)
                state = states + 0.001 * actions[..., :1]
                return visual, state

        environment = plan_pointmaze.create_environment(
            "PointMaze_UMaze-v3", max_micro_steps=5, camera="topdown"
        )
        frames = []
        try:
            record = plan_pointmaze_proprio.run_planning_episode(
                env=environment,
                encoder=TinyEncoder(),
                model=TinyModel(),
                device=torch.device("cpu"),
                seed=0,
                max_macro_steps=1,
                planning_horizon=1,
                population=4,
                num_elites=2,
                num_iterations=1,
                candidate_batch_size=2,
                state_mean=torch.zeros(4),
                state_std=torch.ones(4),
                frame_callback=frames.append,
            )
        finally:
            environment.close()

        self.assertEqual(len(record["proprio_states"]), 2)
        self.assertEqual(len(record["macro_actions"]), 1)
        self.assertEqual(len(frames), 6)
        np.testing.assert_allclose(
            np.asarray(record["proprio_states"])[:, :2],
            np.asarray(record["achieved_positions"]),
            atol=1e-6,
        )

    def test_episode_executes_multiple_actions_before_replanning_with_total_step_limit(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images), 2, 3, device=images.device)

        class TinyModel(torch.nn.Module):
            num_hist = 2

            def forward(self, latents, states, actions):
                next_states = states.clone()
                next_states[:, -1, 0] += actions[:, -1, 0]
                return latents, next_states

        def action(value):
            return torch.full((10,), value)

        plans = [
            CEMResult(torch.stack([action(0.1), action(0.2), action(0.3)]), 1.0),
            CEMResult(torch.stack([action(0.4), action(0.5), action(0.6)]), 2.0),
        ]
        environment = plan_pointmaze.create_environment(
            "PointMaze_UMaze-v3", max_micro_steps=15, camera="topdown"
        )
        try:
            with patch.object(plan_pointmaze_proprio, "plan_action", side_effect=plans) as planner:
                record = plan_pointmaze_proprio.run_planning_episode(
                    env=environment,
                    encoder=TinyEncoder(),
                    model=TinyModel(),
                    device=torch.device("cpu"),
                    seed=0,
                    max_macro_steps=3,
                    planning_horizon=3,
                    actions_per_plan=2,
                    population=2,
                    num_elites=1,
                    num_iterations=1,
                    candidate_batch_size=2,
                    state_mean=torch.zeros(4),
                    state_std=torch.ones(4),
                    cost="position",
                    collect_plan_traces=True,
                )
                walls = plan_pointmaze.extract_wall_rectangles(environment)
        finally:
            environment.close()

        self.assertEqual(planner.call_count, 2)
        self.assertEqual(record["macro_steps"], 3)
        self.assertEqual(record["micro_steps"], 15)
        np.testing.assert_allclose(
            np.asarray(record["macro_actions"])[:, 0], [0.1, 0.2, 0.4]
        )
        self.assertEqual(len(record["proprio_states"]), 4)
        self.assertEqual(len(record["achieved_positions"]), 4)
        self.assertEqual(record["final_planner_cost"], 2.0)
        traces = record["plan_traces"]
        self.assertEqual(len(traces), 2)
        self.assertEqual([len(trace["actual_states"]) for trace in traces], [3, 2])
        self.assertEqual([len(trace["predicted_states"]) for trace in traces], [3, 3])
        self.assertEqual([trace["start_macro_step"] for trace in traces], [0, 2])
        np.testing.assert_allclose(
            np.asarray(traces[0]["predicted_states"])[:, 0]
            - traces[0]["start_state"][0],
            [0.1, 0.3, 0.6], atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(traces[0]["actual_states"]),
            np.asarray(record["proprio_states"][:3]), atol=1e-6,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = plan_pointmaze_proprio.save_plan_trace_diagnostics(
                record, walls, Path(directory), episode_index=0,
            )
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertTrue((Path(directory) / "episode_0000_seed_000000_traces.json").is_file())


if __name__ == "__main__":
    unittest.main()
