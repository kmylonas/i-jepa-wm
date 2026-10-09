import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze
from src.mylonas_ijepa_wm import planning


class FakeData:
    def __init__(self):
        self.qpos = np.array([0.2, 0.3], dtype=np.float64)
        self.qvel = np.array([1.0, 2.0], dtype=np.float64)


class FakePointEnvironment:
    def __init__(self):
        self.data = FakeData()

    def set_state(self, qpos, qvel):
        self.data.qpos = np.asarray(qpos).copy()
        self.data.qvel = np.asarray(qvel).copy()


class FakeRenderEnvironment:
    def __init__(self, fail_render=False):
        self.unwrapped = self
        self.goal = np.array([0.8, -0.4], dtype=np.float64)
        self.point_env = FakePointEnvironment()
        self.fail_render = fail_render
        self.rendered_qpos = None
        self.rendered_qvel = None

    def render(self):
        self.rendered_qpos = self.point_env.data.qpos.copy()
        self.rendered_qvel = self.point_env.data.qvel.copy()
        if self.fail_render:
            raise RuntimeError("render failed")
        return np.zeros((224, 224, 3), dtype=np.uint8)


class TargetObservationTest(unittest.TestCase):
    def test_renders_ball_at_goal_and_restores_simulator_state(self):
        environment = FakeRenderEnvironment()
        original_qpos = environment.point_env.data.qpos.copy()
        original_qvel = environment.point_env.data.qvel.copy()

        frame = plan_pointmaze.render_target_observation(environment)

        self.assertEqual(frame.shape, (224, 224, 3))
        np.testing.assert_allclose(
            environment.rendered_qpos,
            environment.goal,
        )
        np.testing.assert_allclose(
            environment.rendered_qvel,
            np.zeros(2),
        )
        np.testing.assert_allclose(
            environment.point_env.data.qpos,
            original_qpos,
        )
        np.testing.assert_allclose(
            environment.point_env.data.qvel,
            original_qvel,
        )

    def test_restores_simulator_state_when_rendering_fails(self):
        environment = FakeRenderEnvironment(fail_render=True)
        original_qpos = environment.point_env.data.qpos.copy()
        original_qvel = environment.point_env.data.qvel.copy()

        with self.assertRaisesRegex(RuntimeError, "render failed"):
            plan_pointmaze.render_target_observation(environment)

        np.testing.assert_allclose(
            environment.point_env.data.qpos,
            original_qpos,
        )
        np.testing.assert_allclose(
            environment.point_env.data.qvel,
            original_qvel,
        )


class FakeStepEnvironment:
    def __init__(self, stop_kind=None, stop_after=None):
        self.stop_kind = stop_kind
        self.stop_after = stop_after
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        step = len(self.actions)
        stopped = step == self.stop_after
        observation = {
            "observation": np.array(
                [float(step), 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            "achieved_goal": np.array(
                [float(step), 0.0],
                dtype=np.float32,
            ),
            "desired_goal": np.array([5.0, 0.0], dtype=np.float32),
        }
        return (
            observation,
            0.0,
            stopped and self.stop_kind == "terminated",
            stopped and self.stop_kind == "truncated",
            {"success": stopped and self.stop_kind == "success"},
        )


class MacroActionExecutionTest(unittest.TestCase):
    def test_executes_five_ordered_two_dimensional_actions(self):
        environment = FakeStepEnvironment()
        macro_action = torch.arange(10, dtype=torch.float32)

        result = plan_pointmaze.execute_macro_action(
            environment,
            macro_action,
        )

        self.assertEqual(result.micro_steps, 5)
        self.assertFalse(result.success)
        self.assertEqual(len(environment.actions), 5)
        np.testing.assert_allclose(
            np.stack(environment.actions),
            np.arange(10, dtype=np.float32).reshape(5, 2),
        )
        self.assertEqual(result.observation["achieved_goal"][0], 5.0)

    def test_stops_early_on_success_termination_or_truncation(self):
        for stop_kind in ["success", "terminated", "truncated"]:
            with self.subTest(stop_kind=stop_kind):
                environment = FakeStepEnvironment(
                    stop_kind=stop_kind,
                    stop_after=2,
                )
                result = plan_pointmaze.execute_macro_action(
                    environment,
                    torch.zeros(10),
                )

                self.assertEqual(result.micro_steps, 2)
                self.assertEqual(len(environment.actions), 2)
                self.assertEqual(
                    result.success,
                    stop_kind == "success",
                )
                self.assertEqual(
                    result.terminated,
                    stop_kind == "terminated",
                )
                self.assertEqual(
                    result.truncated,
                    stop_kind == "truncated",
                )

    def test_rejects_wrong_macro_action_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            plan_pointmaze.execute_macro_action(
                FakeStepEnvironment(),
                torch.zeros(5, 2),
            )


class MPCHistoryTest(unittest.TestCase):
    def test_initializes_and_updates_aligned_real_history(self):
        z0 = torch.tensor([[0.0], [1.0]])
        z1 = torch.tensor([[2.0], [3.0]])
        z2 = torch.tensor([[4.0], [5.0]])
        a0 = torch.arange(10, dtype=torch.float32)
        a1 = torch.arange(10, 20, dtype=torch.float32)

        latent_history, action_history = (
            plan_pointmaze.initialize_mpc_history(
                initial_latent=z0,
                num_hist=3,
                action_dim=10,
            )
        )
        latent_history, action_history = (
            plan_pointmaze.update_mpc_history(
                latent_history=latent_history,
                action_history=action_history,
                executed_action=a0,
                next_latent=z1,
            )
        )
        latent_history, action_history = (
            plan_pointmaze.update_mpc_history(
                latent_history=latent_history,
                action_history=action_history,
                executed_action=a1,
                next_latent=z2,
            )
        )

        torch.testing.assert_close(
            latent_history,
            torch.stack([z0, z1, z2]),
        )
        torch.testing.assert_close(
            action_history,
            torch.stack([a0, a1]),
        )

    def test_update_does_not_mutate_input_histories(self):
        z0 = torch.zeros(2, 1)
        latent_history, action_history = (
            plan_pointmaze.initialize_mpc_history(z0, 3, 10)
        )
        original_latents = latent_history.clone()
        original_actions = action_history.clone()

        plan_pointmaze.update_mpc_history(
            latent_history,
            action_history,
            torch.ones(10),
            torch.ones(2, 1),
        )

        torch.testing.assert_close(latent_history, original_latents)
        torch.testing.assert_close(action_history, original_actions)


class FirstActionWorldModel(torch.nn.Module):
    def forward(self, latent_history, actions):
        predictions = torch.zeros_like(latent_history)
        predictions[:, -1] = (
            latent_history[:, -1]
            + actions[:, -1, :1].unsqueeze(-1)
        )
        return predictions


class PlanActionTest(unittest.TestCase):
    def test_optimizes_and_returns_complete_bounded_macro_sequence(self):
        latent_history = torch.zeros(3, 1, 1)
        action_history = torch.zeros(2, 10)
        target_latent = torch.tensor([[0.5]])

        result = plan_pointmaze.plan_action(
            model=FirstActionWorldModel(),
            latent_history=latent_history,
            action_history=action_history,
            target_latent=target_latent,
            device=torch.device("cpu"),
            cost_mode="latent",
            action_low=torch.full((10,), -1.0),
            action_high=torch.full((10,), 1.0),
            planning_horizon=1,
            population=256,
            num_elites=32,
            num_iterations=5,
            candidate_batch_size=17,
            generator=torch.Generator().manual_seed(4),
        )

        self.assertEqual(result.action_sequence.shape, (1, 10))
        self.assertGreaterEqual(result.action_sequence.min().item(), -1.0)
        self.assertLessEqual(result.action_sequence.max().item(), 1.0)
        self.assertLess(result.cost, 0.01)

        predictions = planning.rollout_action_sequences(
            model=FirstActionWorldModel(),
            latent_history=latent_history.unsqueeze(0),
            action_history=action_history,
            candidate_actions=result.action_sequence.unsqueeze(0),
            device=torch.device("cpu"),
            chunk_size=1,
        )
        rescored = planning.compute_terminal_cost(
            predicted_latents=predictions,
            target_latent=target_latent,
            mode="latent",
        )
        self.assertAlmostEqual(result.cost, rescored.item(), places=6)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.saw_grad_enabled = None

    def forward(self, images):
        self.saw_grad_enabled = torch.is_grad_enabled()
        if images.shape != (1, 3, 224, 224):
            raise ValueError("unexpected preprocessed image shape")
        return torch.ones(1, 4, 2, device=images.device) * self.scale


class FrozenEncoderTest(unittest.TestCase):
    def test_loads_only_encoder_weights_and_freezes_model(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "encoder.pt"
            torch.save(
                {
                    "encoder": {
                        "module.scale": torch.tensor(3.0),
                    },
                    "predictor": {"not_used": torch.tensor(7.0)},
                    "epoch": 10,
                },
                checkpoint_path,
            )

            encoder = plan_pointmaze.load_frozen_ijepa_encoder(
                checkpoint_path=checkpoint_path,
                device=torch.device("cpu"),
                encoder_factory=lambda **kwargs: TinyEncoder(),
            )

        self.assertFalse(encoder.training)
        self.assertEqual(encoder.scale.item(), 3.0)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in encoder.parameters())
        )

    def test_encodes_one_uint8_hwc_frame_without_gradients(self):
        encoder = TinyEncoder()
        frame = np.zeros((224, 224, 3), dtype=np.uint8)

        latent = plan_pointmaze.encode_frame(
            encoder=encoder,
            frame=frame,
            device=torch.device("cpu"),
        )

        self.assertEqual(latent.shape, (4, 2))
        self.assertEqual(latent.dtype, torch.float32)
        self.assertFalse(latent.requires_grad)
        self.assertFalse(encoder.saw_grad_enabled)


class FakePlanningPointEnvironment:
    def __init__(self):
        self.unwrapped = self
        self.goal = np.array([2.0, 0.0], dtype=np.float64)
        self.target_site_id = 0
        self.point_env = FakePointEnvironment()
        self.point_env.model = SimpleNamespace(
            site_rgba=np.ones((1, 4), dtype=np.float64),
        )
        self.action_space = SimpleNamespace(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

    def observation(self):
        position = self.point_env.data.qpos.astype(np.float32)
        velocity = self.point_env.data.qvel.astype(np.float32)
        return {
            "observation": np.concatenate([position, velocity]),
            "achieved_goal": position.copy(),
            "desired_goal": self.goal.astype(np.float32),
        }

    def reset(self, seed=None):
        self.point_env.set_state(np.zeros(2), np.zeros(2))
        return self.observation(), {"success": False}

    def render(self):
        value = int(np.clip(self.point_env.data.qpos[0] + 2.0, 0, 4) * 50)
        return np.full((224, 224, 3), value, dtype=np.uint8)

    def step(self, action):
        qpos = self.point_env.data.qpos.copy()
        qpos[0] += 0.2
        self.point_env.set_state(qpos, np.zeros(2))
        observation = self.observation()
        success = bool(
            np.linalg.norm(observation["achieved_goal"] - self.goal) < 0.05
        )
        return observation, 0.0, False, False, {"success": success}


class CountingEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        value = images.mean()
        return value.expand(1, 1, 1)


class PlanningEpisodeTest(unittest.TestCase):
    def test_replans_from_real_images_until_success(self):
        environment = FakePlanningPointEnvironment()
        encoder = CountingEncoder()
        planner_calls = []

        def fake_planner(**arguments):
            planner_calls.append(arguments)
            return planning.CEMResult(
                action_sequence=torch.zeros(
                    arguments["planning_horizon"],
                    10,
                ),
                cost=float(len(planner_calls)),
            )

        result = plan_pointmaze.run_planning_episode(
            env=environment,
            encoder=encoder,
            model=torch.nn.Identity(),
            device=torch.device("cpu"),
            seed=9,
            cost_mode="probe",
            max_macro_steps=3,
            planning_horizon=2,
            population=4,
            num_elites=2,
            num_iterations=1,
            candidate_batch_size=2,
            position_probe=torch.nn.Identity(),
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
            generator=torch.Generator().manual_seed(9),
            planner_fn=fake_planner,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["seed"], 9)
        self.assertEqual(result["macro_steps"], 2)
        self.assertEqual(result["micro_steps"], 10)
        self.assertEqual(len(result["macro_actions"]), 2)
        self.assertEqual(len(result["achieved_positions"]), 3)
        self.assertEqual(len(planner_calls), 2)
        self.assertEqual(encoder.calls, 4)
        self.assertEqual(
            environment.point_env.model.site_rgba[0, 3],
            0.0,
        )
        for arguments in planner_calls:
            self.assertNotIn("observation", arguments)
            self.assertNotIn("state", arguments)
            self.assertIsNone(arguments["oracle_goal"])


class PlanningParserTest(unittest.TestCase):
    def test_uses_colab_configurable_planning_defaults(self):
        args = plan_pointmaze.parse_args([])

        self.assertEqual(args.cost, "probe")
        self.assertEqual(args.planning_horizon, 3)
        self.assertEqual(args.population, 256)
        self.assertEqual(args.num_elites, 32)
        self.assertEqual(args.cem_iterations, 5)
        self.assertEqual(args.candidate_batch_size, 16)
        self.assertEqual(args.max_macro_steps, 20)

    def test_latent_mode_does_not_require_probe_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            encoder_path = root / "encoder.pt"
            model_path = root / "world.pt"
            encoder_path.touch()
            model_path.touch()
            args = plan_pointmaze.parse_args([
                "--cost", "latent",
                "--encoder-checkpoint", str(encoder_path),
                "--world-model-checkpoint", str(model_path),
                "--position-probe-checkpoint", str(root / "missing.pt"),
            ])

            plan_pointmaze.validate_arguments(args)

    def test_probe_and_oracle_modes_require_probe_checkpoint(self):
        for mode in ["probe", "oracle"]:
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    encoder_path = root / "encoder.pt"
                    model_path = root / "world.pt"
                    encoder_path.touch()
                    model_path.touch()
                    args = plan_pointmaze.parse_args([
                        "--cost", mode,
                        "--encoder-checkpoint", str(encoder_path),
                        "--world-model-checkpoint", str(model_path),
                        "--position-probe-checkpoint",
                        str(root / "missing.pt"),
                    ])

                    with self.assertRaisesRegex(
                        FileNotFoundError,
                        "probe",
                    ):
                        plan_pointmaze.validate_arguments(args)


class PlanningSummaryTest(unittest.TestCase):
    def test_summarizes_success_distance_and_steps(self):
        records = [
            {"success": True, "final_goal_distance": 1.0, "macro_steps": 2},
            {"success": False, "final_goal_distance": 0.5, "macro_steps": 4},
            {"success": True, "final_goal_distance": 0.25, "macro_steps": 6},
        ]

        summary = plan_pointmaze.summarize_results(records)

        self.assertEqual(summary["num_episodes"], 3)
        self.assertAlmostEqual(summary["success_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["mean_final_goal_distance"], 7.0 / 12.0)
        self.assertEqual(summary["median_final_goal_distance"], 0.5)
        self.assertEqual(summary["mean_macro_steps"], 4.0)

if __name__ == "__main__":
    unittest.main()
