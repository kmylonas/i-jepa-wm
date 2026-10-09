import unittest

import torch

from src.mylonas_ijepa_wm import planning


class AddLastActionWorldModel(torch.nn.Module):
    def forward(self, latent_history, actions):
        predictions = torch.zeros_like(latent_history)
        predictions[:, -1] = (
            latent_history[:, -1]
            + actions[:, -1].unsqueeze(-1)
        )
        return predictions


class ScalarPositionProbe(torch.nn.Module):
    def forward(self, latents):
        x_coordinate = latents[:, 0, 0]
        return torch.stack(
            [x_coordinate, torch.zeros_like(x_coordinate)],
            dim=1,
        )


class ActionAlignmentTest(unittest.TestCase):
    def test_prepends_history_and_preserves_raw_candidates(self):
        history = torch.tensor([[10.0], [20.0]])
        candidates = torch.tensor([
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ])

        actions = planning.assemble_rollout_actions(
            action_history=history,
            candidate_actions=candidates,
        )

        torch.testing.assert_close(
            actions,
            torch.tensor([
                [[10.0], [20.0], [1.0], [2.0], [3.0]],
                [[10.0], [20.0], [4.0], [5.0], [6.0]],
            ]),
        )

    def test_normalizes_model_actions_without_changing_candidates(self):
        history = torch.tensor([[10.0], [20.0]])
        candidates = torch.tensor([[[1.0], [2.0]]])
        original = candidates.clone()

        actions = planning.assemble_rollout_actions(
            action_history=history,
            candidate_actions=candidates,
            action_mean=torch.tensor([1.0]),
            action_std=torch.tensor([2.0]),
        )

        torch.testing.assert_close(
            actions,
            torch.tensor([[[4.5], [9.5], [0.0], [0.5]]]),
        )
        torch.testing.assert_close(candidates, original)


class CandidateRolloutTest(unittest.TestCase):
    def test_chunked_rollout_preserves_candidate_order(self):
        history = torch.tensor([[[[0.0]], [[1.0]], [[3.0]]]])
        action_history = torch.tensor([[1.0], [2.0]])
        candidates = torch.tensor([
            [[3.0], [4.0]],
            [[5.0], [6.0]],
            [[7.0], [8.0]],
        ])

        predictions = planning.rollout_action_sequences(
            model=AddLastActionWorldModel(),
            latent_history=history,
            action_history=action_history,
            candidate_actions=candidates,
            device=torch.device("cpu"),
            chunk_size=2,
        )

        torch.testing.assert_close(
            predictions[:, :, 0, 0],
            torch.tensor([
                [6.0, 10.0],
                [8.0, 14.0],
                [10.0, 18.0],
            ]),
        )

    def test_requires_one_shared_latent_history(self):
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            planning.rollout_action_sequences(
                model=AddLastActionWorldModel(),
                latent_history=torch.zeros(2, 3, 1, 1),
                action_history=torch.zeros(2, 1),
                candidate_actions=torch.zeros(1, 2, 1),
                device=torch.device("cpu"),
                chunk_size=1,
            )

    def test_rejects_non_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            planning.rollout_action_sequences(
                model=AddLastActionWorldModel(),
                latent_history=torch.zeros(1, 3, 1, 1),
                action_history=torch.zeros(2, 1),
                candidate_actions=torch.zeros(1, 2, 1),
                device=torch.device("cpu"),
                chunk_size=0,
            )

    def test_chunked_scoring_releases_predictions_between_chunks(self):
        history = torch.tensor([[[[0.0]], [[1.0]], [[3.0]]]])
        candidates = torch.tensor([
            [[3.0], [4.0]],
            [[5.0], [6.0]],
            [[7.0], [8.0]],
        ])
        cost_batch_sizes = []

        def terminal_value(predictions):
            cost_batch_sizes.append(predictions.shape[0])
            return predictions[:, -1, 0, 0]

        costs = planning.score_action_sequences(
            model=AddLastActionWorldModel(),
            latent_history=history,
            action_history=torch.tensor([[1.0], [2.0]]),
            candidate_actions=candidates,
            device=torch.device("cpu"),
            chunk_size=2,
            cost_fn=terminal_value,
        )

        torch.testing.assert_close(costs, torch.tensor([10.0, 14.0, 18.0]))
        self.assertEqual(cost_batch_sizes, [2, 1])


class TerminalCostTest(unittest.TestCase):
    def setUp(self):
        self.predictions = torch.tensor([
            [[[0.0], [2.0]]],
            [[[1.0], [3.0]]],
        ])
        self.target_latent = torch.tensor([[2.0], [0.0]])

    def test_latent_cost_is_terminal_patch_mse(self):
        goal = torch.tensor([[1.0], [1.0]])

        costs = planning.compute_terminal_cost(
            predicted_latents=self.predictions,
            target_latent=goal,
            mode="latent",
        )

        torch.testing.assert_close(costs, torch.tensor([1.0, 2.0]))

    def test_probe_cost_decodes_target_image_latent(self):
        costs = planning.compute_terminal_cost(
            predicted_latents=self.predictions,
            target_latent=self.target_latent,
            mode="probe",
            position_probe=ScalarPositionProbe(),
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
        )

        torch.testing.assert_close(costs, torch.tensor([2.0, 1.0]))

    def test_oracle_cost_uses_numeric_goal(self):
        costs = planning.compute_terminal_cost(
            predicted_latents=self.predictions,
            target_latent=self.target_latent,
            mode="oracle",
            position_probe=ScalarPositionProbe(),
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
            oracle_goal=torch.tensor([3.0, 0.0]),
        )

        torch.testing.assert_close(costs, torch.tensor([3.0, 2.0]))

    def test_oracle_cost_requires_numeric_goal(self):
        with self.assertRaisesRegex(ValueError, "oracle_goal"):
            planning.compute_terminal_cost(
                predicted_latents=self.predictions,
                target_latent=self.target_latent,
                mode="oracle",
                position_probe=ScalarPositionProbe(),
                position_mean=torch.zeros(2),
                position_std=torch.ones(2),
            )

    def test_probe_cost_requires_probe_and_statistics(self):
        with self.assertRaisesRegex(ValueError, "position probe"):
            planning.compute_terminal_cost(
                predicted_latents=self.predictions,
                target_latent=self.target_latent,
                mode="probe",
            )

    def test_rejects_unknown_cost_mode(self):
        with self.assertRaisesRegex(ValueError, "cost mode"):
            planning.compute_terminal_cost(
                predicted_latents=self.predictions,
                target_latent=self.target_latent,
                mode="other",
            )


class CEMValidationTest(unittest.TestCase):
    def setUp(self):
        self.arguments = {
            "score_fn": lambda actions: actions.square().sum(dim=(1, 2)),
            "horizon": 3,
            "action_dim": 2,
            "action_low": torch.tensor([-1.0, -1.0]),
            "action_high": torch.tensor([1.0, 1.0]),
            "population": 8,
            "num_elites": 2,
            "num_iterations": 2,
        }

    def test_rejects_non_positive_dimensions_and_counts(self):
        for name in [
            "horizon",
            "action_dim",
            "population",
            "num_elites",
            "num_iterations",
        ]:
            with self.subTest(name=name):
                arguments = dict(self.arguments)
                arguments[name] = 0
                with self.assertRaisesRegex(ValueError, name):
                    planning.cem_optimize(**arguments)

    def test_rejects_more_elites_than_population(self):
        arguments = dict(self.arguments)
        arguments["num_elites"] = 9

        with self.assertRaisesRegex(ValueError, "num_elites"):
            planning.cem_optimize(**arguments)

    def test_rejects_non_positive_minimum_standard_deviation(self):
        arguments = dict(self.arguments)
        arguments["min_std"] = 0.0

        with self.assertRaisesRegex(ValueError, "min_std"):
            planning.cem_optimize(**arguments)

    def test_rejects_bounds_with_wrong_shape_or_order(self):
        wrong_shape = dict(self.arguments)
        wrong_shape["action_low"] = torch.tensor([-1.0])
        with self.assertRaisesRegex(ValueError, "bounds"):
            planning.cem_optimize(**wrong_shape)

        wrong_order = dict(self.arguments)
        wrong_order["action_low"] = torch.tensor([-1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "bounds"):
            planning.cem_optimize(**wrong_order)


class CEMOptimizationTest(unittest.TestCase):
    def run_optimizer(self, seed):
        target = torch.tensor([0.35, -0.45]).view(1, 1, 2)

        def quadratic_cost(actions):
            return (actions - target).square().sum(dim=(1, 2))

        return planning.cem_optimize(
            score_fn=quadratic_cost,
            horizon=2,
            action_dim=2,
            action_low=torch.tensor([-1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0]),
            population=512,
            num_elites=64,
            num_iterations=6,
            generator=torch.Generator().manual_seed(seed),
        )

    def test_converges_to_bounded_quadratic_optimum(self):
        result = self.run_optimizer(seed=7)
        expected = torch.tensor([
            [0.35, -0.45],
            [0.35, -0.45],
        ])

        torch.testing.assert_close(
            result.action_sequence,
            expected,
            atol=0.08,
            rtol=0.0,
        )
        self.assertGreaterEqual(result.action_sequence.min().item(), -1.0)
        self.assertLessEqual(result.action_sequence.max().item(), 1.0)
        self.assertLess(result.cost, 0.03)

    def test_is_reproducible_with_independently_seeded_generators(self):
        first = self.run_optimizer(seed=11)
        second = self.run_optimizer(seed=11)

        torch.testing.assert_close(
            first.action_sequence,
            second.action_sequence,
        )
        self.assertEqual(first.cost, second.cost)


if __name__ == "__main__":
    unittest.main()
