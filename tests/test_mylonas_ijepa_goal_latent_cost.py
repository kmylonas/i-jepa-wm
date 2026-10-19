"""Checks for the fixed-goal latent-cost diagnostic."""

import csv
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze


MODULE_NAME = "src.mylonas_ijepa_wm.goal_latent_cost"


def experiment_module(test_case):
    test_case.assertIsNotNone(
        importlib.util.find_spec(MODULE_NAME),
        "The fixed-goal experiment module is missing",
    )
    return importlib.import_module(MODULE_NAME)


class TinyVisualEncoder(torch.nn.Module):
    def forward(self, images):
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (16, 16))
        return pooled.permute(0, 2, 3, 1).reshape(len(images), 256, 3)


class FixedGoalLatentCostTest(unittest.TestCase):
    def test_goal_seeds_cover_separated_goal_regions(self):
        experiment = experiment_module(self)

        class GoalSampler:
            goals = {
                0: (0.0, 0.0),
                1: (0.01, 0.0),
                2: (1.0, 0.0),
                3: (0.0, 1.0),
            }

            def reset(self, seed):
                return {"desired_goal": np.asarray(self.goals[seed])}, {}

        self.assertTrue(hasattr(experiment, "select_goal_seeds"))
        self.assertEqual(
            experiment.select_goal_seeds(GoalSampler(), 3, base_seed=0, pool_size=4),
            [0, 2, 3],
        )

    def test_candidate_render_at_goal_matches_planner_goal_and_restores_state(self):
        experiment = experiment_module(self)
        env = plan_pointmaze.create_environment("PointMaze_UMaze-v3", 1)
        try:
            env.reset(seed=0)
            plan_pointmaze.hide_goal_marker(env)
            point_env = env.unwrapped.point_env
            before_qpos = point_env.data.qpos.copy()
            before_qvel = point_env.data.qvel.copy()
            target = plan_pointmaze.render_target_observation(env)

            frames = experiment.render_positions(
                env, np.asarray([env.unwrapped.goal], dtype=np.float32)
            )

            np.testing.assert_array_equal(frames[0], target)
            np.testing.assert_array_equal(point_env.data.qpos, before_qpos)
            np.testing.assert_array_equal(point_env.data.qvel, before_qvel)
        finally:
            env.close()

    def test_sampling_keeps_goal_and_selects_only_eligible_grid_points(self):
        experiment = experiment_module(self)
        valid = np.asarray(
            [[0.04, 0.0], [0.08, 0.0], [0.15, 0.0], [0.35, 0.0]],
            dtype=np.float32,
        )
        positions, bins = experiment.sample_positions(
            goal=np.asarray([0.0, 0.0], dtype=np.float32),
            valid_positions=valid,
            edges=(0.0, 0.1, 0.2, 0.4),
            per_bin=1,
            rng=np.random.default_rng(0),
        )

        np.testing.assert_array_equal(positions[0], [0.0, 0.0])
        self.assertEqual(bins, [-1, 0, 1, 2])
        self.assertEqual(len(positions), 4)
        for position in positions[1:]:
            self.assertTrue(np.any(np.all(valid == position, axis=1)))

    def test_small_run_saves_real_goal_comparison_with_zero_self_mse(self):
        experiment = experiment_module(self)
        env = plan_pointmaze.create_environment("PointMaze_UMaze-v3", 1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                output_dir = Path(directory)
                result = experiment.run_experiment(
                    env=env,
                    encoder=TinyVisualEncoder().eval(),
                    output_dir=output_dir,
                    num_goals=1,
                    base_seed=0,
                    grid_resolution=21,
                    wall_clearance=0.12,
                    per_bin=1,
                    batch_size=4,
                    device=torch.device("cpu"),
                    precision="fp32",
                )

                with (output_dir / "samples.csv").open(newline="") as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(result["num_goals"], 1)
                self.assertGreater(len(rows), 1)
                self.assertEqual(sum(int(row["bin_index"]) == -1 for row in rows), 1)
                self.assertLess(float(rows[0]["latent_mse"]), 1e-8)
                self.assertGreater(max(float(row["latent_mse"]) for row in rows), 0)
                self.assertTrue((output_dir / "goal_00.png").is_file())
                self.assertTrue((output_dir / "per_goal.png").is_file())
                self.assertTrue((output_dir / "summary.json").is_file())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
