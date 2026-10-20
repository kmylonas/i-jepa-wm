import unittest

import numpy as np
import torch

from src.mylonas_ijepa_wm import planning_cost_rankings_proprio as diagnostic


class StateDrivenModel(torch.nn.Module):
    num_hist = 1

    def forward(self, latents, states, actions):
        next_state = states + actions[..., :1]
        next_visual = next_state[..., :1].unsqueeze(-2)
        return next_visual, next_state


class PlanningCostRankingsTest(unittest.TestCase):
    def test_same_candidate_rollouts_can_get_opposite_cost_rankings(self):
        result = diagnostic.score_candidate_sequences(
            model=StateDrivenModel(),
            latent_history=torch.zeros(1, 1, 1),
            state_history=torch.zeros(1, 4),
            action_history=torch.empty(0, 1),
            candidate_actions=torch.tensor([[[0.0]], [[1.0]]]),
            target_latent=torch.tensor([[1.0]]),
            goal_position=torch.tensor([0.0, 0.0]),
            state_mean=torch.zeros(4),
            state_std=torch.ones(4),
            device=torch.device("cpu"),
            candidate_batch_size=1,
        )
        np.testing.assert_allclose(result["latent_cost"], [1.0, 0.0])
        np.testing.assert_allclose(
            result["position_cost"], [0.0, np.sqrt(2.0)], rtol=1e-6
        )
        np.testing.assert_allclose(result["predicted_xy"], [[0.0, 0.0], [1.0, 1.0]])

    def test_rank_comparison_reports_misleading_latent_elites(self):
        result = diagnostic.compare_rankings(
            latent_costs=np.array([1.0, 2.0, 3.0, 4.0]),
            position_costs=np.array([4.0, 3.0, 2.0, 1.0]),
            num_elites=2,
        )
        self.assertAlmostEqual(result["spearman"], -1.0)
        self.assertEqual(result["elite_overlap_count"], 0)
        self.assertEqual(result["latent_best_position_distance"], 4.0)
        self.assertEqual(result["position_best_distance"], 1.0)
        self.assertEqual(result["random_overlap_expected"], 1.0)


if __name__ == "__main__":
    unittest.main()
