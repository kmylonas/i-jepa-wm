import unittest

from src.mylonas_ijepa_wm import compare_encoder_goal_geometry as comparison


class CompareEncoderGoalGeometryTest(unittest.TestCase):
    def test_sample_candidate_ids_are_reproducible_unique_and_in_range(self):
        first = comparison.sample_candidate_ids(10, 4, seed=7)
        second = comparison.sample_candidate_ids(10, 4, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(all(0 <= index < 10 for index in first))
        with self.assertRaises(ValueError):
            comparison.sample_candidate_ids(10, 11, seed=7)

    def test_summary_compares_each_encoders_cost_with_actual_distance(self):
        rows = [
            {"actual_distance": distance, "ijepa_mse": ijepa, "dinov2_mse": dino}
            for distance, ijepa, dino in (
                (1.0, 4.0, 1.0),
                (2.0, 3.0, 2.0),
                (3.0, 2.0, 3.0),
                (4.0, 1.0, 4.0),
            )
        ]
        result = comparison.summarize_geometry(rows)
        self.assertEqual(result["candidate_count"], 4)
        self.assertAlmostEqual(result["ijepa_distance_spearman"], -1.0)
        self.assertAlmostEqual(result["dinov2_distance_spearman"], 1.0)


if __name__ == "__main__":
    unittest.main()
