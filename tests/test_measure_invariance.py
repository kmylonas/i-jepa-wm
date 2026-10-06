import unittest

import torch


try:
    from experiments.measure_invariance import (
        apply_corruption,
        compute_invariance_metrics,
        mean_pool_and_normalize,
    )
except ImportError as exc:
    apply_corruption = None
    compute_invariance_metrics = None
    mean_pool_and_normalize = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class MeasureInvarianceTest(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"invariance helpers are unavailable: {IMPORT_ERROR}")

    def test_mean_pooling_produces_unit_global_embeddings(self):
        patch_tokens = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 2.0], [0.0, 4.0]],
            ]
        )

        embeddings = mean_pool_and_normalize(patch_tokens)

        torch.testing.assert_close(
            embeddings, torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )

    def test_noise_is_reproducible_for_the_same_seed(self):
        image = torch.full((3, 8, 8), 0.5)

        first = apply_corruption(image, "gaussian_noise", 0.05, seed=7)
        second = apply_corruption(image, "gaussian_noise", 0.05, seed=7)

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, image))

    def test_metrics_reward_retrieving_the_corresponding_clean_image(self):
        clean = torch.eye(3)
        transformed = clean.clone()

        metrics = compute_invariance_metrics(clean, transformed)

        self.assertAlmostEqual(metrics["positive_cosine"], 1.0)
        self.assertAlmostEqual(metrics["hardest_negative_cosine"], 0.0)
        self.assertAlmostEqual(metrics["margin"], 1.0)
        self.assertAlmostEqual(metrics["retrieval_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
