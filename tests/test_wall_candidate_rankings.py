import unittest

import numpy as np
import torch

from src.mylonas_ijepa_wm import evaluate_wall_candidate_rankings as diagnostic


class CandidateRankingTest(unittest.TestCase):
    def test_requires_multi_wall_checkpoint_with_requested_training_layout(self):
        metadata = {
            "environment": "wall",
            "experiment_manifest_sha256": "abc123",
            "layout_splits": {"train": [[24, 18], [32, 30]]},
        }
        diagnostic.validate_multi_wall_checkpoint(metadata, 32, 30)
        with self.assertRaisesRegex(ValueError, "multi-wall"):
            diagnostic.validate_multi_wall_checkpoint(
                {"environment": "wall"}, 32, 30
            )
        with self.assertRaisesRegex(ValueError, "training layout"):
            diagnostic.validate_multi_wall_checkpoint(metadata, 28, 24)

    def test_selection_preserves_scored_action_and_prioritizes_final_elites(self):
        rounds = [
            (torch.tensor([[[0.0]], [[1.0]], [[2.0]], [[3.0]]]),
             torch.tensor([3.0, 2.0, 1.0, 0.0])),
            (torch.tensor([[[10.0]], [[11.0]], [[12.0]], [[13.0]]]),
             torch.tensor([2.0, 0.0, 3.0, 1.0])),
        ]
        selected = diagnostic.select_candidates(
            rounds, random_count=2, elite_count=2, seed=9
        )
        final = [item for item in selected if item["source"] == "final_elite"]
        self.assertEqual([item["candidate_id"] for item in final], [1, 3])
        self.assertEqual([item["predicted_cost"] for item in final], [0.0, 1.0])
        self.assertEqual([float(item["actions"][0, 0]) for item in final], [11.0, 13.0])
        self.assertEqual(len([item for item in selected if item["source"] == "initial_random"]), 2)

    def test_summary_detects_reversed_model_ranking_and_selection_regret(self):
        rows = [
            {"predicted_cost": 0.0, "true_cost": 3.0, "actual_distance": 3.0},
            {"predicted_cost": 1.0, "true_cost": 2.0, "actual_distance": 2.0},
            {"predicted_cost": 2.0, "true_cost": 1.0, "actual_distance": 1.0},
        ]
        result = diagnostic.summarize_rankings(rows)
        self.assertAlmostEqual(result["predicted_vs_true_rho"], -1.0)
        self.assertAlmostEqual(result["true_vs_distance_rho"], 1.0)
        self.assertAlmostEqual(result["predicted_best_true_cost"], 3.0)
        self.assertAlmostEqual(result["oracle_best_true_cost"], 1.0)

    def test_replay_resets_simulator_before_every_action_sequence(self):
        class Adapter:
            def __init__(self):
                self.position = 0.0
                self.resets = 0
                self.goal = np.array([10.0, 0.0], dtype=np.float32)

            def reset(self, seed):
                self.position = 0.0
                self.resets += 1
                return type("Initial", (), {"state": np.array([0.0, 0.0])})()

            def step(self, action):
                self.position += float(action[0])
                return type("Outcome", (), {
                    "state": np.array([self.position, 0.0]),
                    "success": False,
                })()

            def render(self):
                return np.array([[[self.position, 0.0, 0.0]]], dtype=np.float32)

        adapter = Adapter()
        candidates = [
            {"actions": np.array([[1.0, 0.0, 1.0, 0.0]])},
            {"actions": np.array([[2.0, 0.0, 2.0, 0.0]])},
        ]
        outcomes = diagnostic.replay_candidates(
            adapter, seed=4, candidates=candidates, actions_per_model_step=2
        )
        self.assertEqual(adapter.resets, 2)
        self.assertEqual([float(item["state"][0]) for item in outcomes], [2.0, 4.0])
        self.assertEqual([float(item["frame"][0, 0, 0]) for item in outcomes], [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
