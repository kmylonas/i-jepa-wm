import unittest

import numpy as np

from src.mylonas_ijepa_wm import validate_planning_costs_proprio as diagnostic


class ValidatePlanningCostsTest(unittest.TestCase):
    def test_candidate_selection_keeps_both_elites_and_unbiased_controls(self):
        latent = np.arange(10, dtype=float)
        position = latent[::-1].copy()
        selection = diagnostic.select_candidate_ids(
            latent, position, top_per_cost=2, random_count=3, seed=42
        )
        self.assertEqual(selection["latent_elites"], {0, 1})
        self.assertEqual(selection["position_elites"], {8, 9})
        self.assertEqual(len(selection["random_ids"]), 3)
        self.assertTrue(selection["random_ids"].isdisjoint({0, 1, 8, 9}))
        self.assertEqual(len(selection["selected_ids"]), 7)

    def test_true_latent_cost_averages_matching_patch_features(self):
        actual = np.array([[[1.0, 0.0]], [[0.0, 2.0]]])
        goal = np.zeros((1, 2))
        np.testing.assert_allclose(
            diagnostic.compute_true_latent_cost(actual, goal), [0.5, 2.0]
        )

    def test_summary_separates_cost_geometry_from_world_model_error(self):
        rows = [
            {"actual_distance": float(value), "true_latent_cost": float(value),
             "predicted_latent_cost": float(5 - value),
             "predicted_position_cost": float(value)}
            for value in (1, 2, 3, 4)
        ]
        result = diagnostic.summarize_case(rows)
        self.assertAlmostEqual(result["true_latent_vs_actual_distance_rho"], 1.0)
        self.assertAlmostEqual(result["predicted_latent_vs_true_latent_rho"], -1.0)
        self.assertAlmostEqual(result["predicted_position_vs_actual_distance_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
