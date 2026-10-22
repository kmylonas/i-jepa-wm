import importlib
import importlib.util
import unittest

import numpy as np


class MazeTransferCollisionDesignTest(unittest.TestCase):
    def setUp(self):
        try:
            self.diagnostic = importlib.import_module(
                "src.mylonas_ijepa_wm.maze_transfer_collisions_proprio"
            )
        except ImportError as error:
            self.fail(f"maze transfer diagnostic is unavailable: {error}")

    def test_rotated_map_moves_the_wall_without_changing_shared_starts(self):
        original = self.diagnostic.maze_map("original")
        vertical = self.diagnostic.maze_map("vertical")

        self.assertEqual(original[1][2], 0)
        self.assertEqual(vertical[1][2], 1)
        self.assertEqual(original[2][1], 1)
        self.assertEqual(vertical[2][1], 0)
        self.assertEqual(original[1][1], vertical[1][1])
        self.assertEqual(original[3][1], vertical[3][1])
        self.assertEqual(original[1][1], 0)
        self.assertEqual(original[3][1], 0)

    def test_cases_push_from_shared_free_cells_toward_changed_cells(self):
        cases = {case.name: case for case in self.diagnostic.COLLISION_CASES}

        new_wall_start, new_wall_action = self.diagnostic.trial_inputs(
            cases["new_wall"], offset=0.25, force=0.8
        )
        old_wall_start, old_wall_action = self.diagnostic.trial_inputs(
            cases["removed_wall"], offset=0.25, force=0.8
        )

        np.testing.assert_allclose(new_wall_start, [-0.75, 1.0])
        np.testing.assert_allclose(new_wall_action.reshape(5, 2), [[0.8, 0.0]] * 5)
        np.testing.assert_allclose(old_wall_start, [-1.0, -0.75])
        np.testing.assert_allclose(old_wall_action.reshape(5, 2), [[0.0, 0.8]] * 5)

    def test_progress_uses_the_push_direction_for_each_case(self):
        cases = {case.name: case for case in self.diagnostic.COLLISION_CASES}

        self.assertAlmostEqual(
            self.diagnostic.forward_progress(
                cases["new_wall"], start_xy=[-0.75, 1.0], end_xy=[-0.55, 1.0]
            ),
            0.2,
        )
        self.assertAlmostEqual(
            self.diagnostic.forward_progress(
                cases["removed_wall"], start_xy=[-1.0, -0.75], end_xy=[-1.0, -0.5]
            ),
            0.25,
        )

    @unittest.skipUnless(importlib.util.find_spec("gymnasium_robotics"), "Gymnasium Robotics unavailable")
    def test_physics5_replay_changes_at_both_wall_locations(self):
        from src.mylonas_ijepa_wm import plan_pointmaze as base

        for case in self.diagnostic.COLLISION_CASES:
            progress = {}
            start, action = self.diagnostic.trial_inputs(case, offset=0.25, force=0.8)
            for layout in ("original", "vertical"):
                env = base.create_environment(
                    "PointMaze_UMaze-v3", max_micro_steps=10,
                    physics_frame_skip=5, maze_map=self.diagnostic.maze_map(layout),
                )
                try:
                    state = self.diagnostic.reset_to_state(env, start, seed=42)
                    np.testing.assert_allclose(state, [*start, 0, 0], atol=1e-6)
                    states = self.diagnostic.simulate_macro(env, action)
                    progress[layout] = self.diagnostic.forward_progress(
                        case, start, states[-1, :2]
                    )
                finally:
                    env.close()
            if case.name == "new_wall":
                self.assertGreater(progress["original"] - progress["vertical"], 0.3)
            else:
                self.assertGreater(progress["vertical"] - progress["original"], 0.3)


if __name__ == "__main__":
    unittest.main()
