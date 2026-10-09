import unittest

import torch

import src.mylonas_ijepa_wm.vit as vit


class LayerNormTest(unittest.TestCase):
    def test_float16_statistics_do_not_overflow(self):
        layer_norm = vit.LayerNorm(embed_dim=4).half()
        X = torch.tensor(
            [[[1000.0, -1000.0, 500.0, -500.0]]],
            dtype=torch.float16,
        )

        output = layer_norm(X)

        expected = torch.tensor(
            [[[1.2646, -1.2646, 0.6323, -0.6323]]],
            dtype=torch.float16,
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(output, expected)


class BlockCausalMaskTest(unittest.TestCase):
    def test_allows_same_and_previous_frames_only(self):
        self.assertTrue(hasattr(vit, "create_block_causal_mask"))

        mask = vit.create_block_causal_mask(
            num_hist=3,
            num_patches=2,
        )

        expected = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )

        torch.testing.assert_close(mask, expected)


class MultiHeadAttentionTest(unittest.TestCase):
    def test_uses_scaled_dot_product_attention_operator(self):
        attention = vit.MultiHeadAttention(
            embed_dim=8,
            num_heads=2,
            dropout=0.0,
        )
        inputs = torch.randn(1, 6, 8)
        mask = vit.create_block_causal_mask(
            num_hist=3,
            num_patches=2,
        )

        with torch.profiler.profile() as profiler:
            output = attention(inputs, attn_mask=mask)

        operator_names = {
            event.key
            for event in profiler.key_averages()
        }

        self.assertEqual(output.shape, inputs.shape)
        self.assertIn(
            "aten::scaled_dot_product_attention",
            operator_names,
        )


class WorldModelViTTest(unittest.TestCase):
    def make_model(self):
        return vit.ViT(
            num_patches=4,
            num_hist=3,
            num_t_blocks=1,
            ijepa_dim=8,
            action_dim=2,
            action_embed_dim=4,
            embed_dim=8,
            num_heads=2,
            mlp_dim=16,
        )

    def test_returns_one_prediction_for_every_input_patch(self):
        required_parameters = {
            "num_patches",
            "num_hist",
            "num_t_blocks",
            "ijepa_dim",
            "action_dim",
            "action_embed_dim",
            "embed_dim",
            "num_heads",
            "mlp_dim",
        }

        actual_parameters = set(
            __import__("inspect").signature(vit.ViT).parameters
        )

        if not required_parameters.issubset(actual_parameters):
            self.fail("ViT does not yet expose the world-model interface")

        model = self.make_model()

        latents = torch.randn(2, 3, 4, 8)
        actions = torch.randn(2, 3, 2)

        predictions = model(latents, actions)

        self.assertEqual(predictions.shape, latents.shape)

    def test_actions_condition_the_predictions(self):
        torch.manual_seed(0)
        model = self.make_model()
        model.eval()

        latents = torch.zeros(1, 3, 4, 8)
        first_actions = torch.zeros(1, 3, 2)
        second_actions = torch.ones(1, 3, 2)

        first_predictions = model(latents, first_actions)
        second_predictions = model(latents, second_actions)

        self.assertFalse(
            torch.allclose(first_predictions, second_predictions)
        )

    def test_spatial_positions_condition_the_predictions(self):
        torch.manual_seed(0)
        model = self.make_model()
        model.eval()

        latents = torch.zeros(1, 3, 4, 8)
        actions = torch.zeros(1, 3, 2)

        predictions = model(latents, actions)

        self.assertFalse(
            torch.allclose(
                predictions[:, :, 0],
                predictions[:, :, 1],
            )
        )

    def test_temporal_positions_condition_the_predictions(self):
        torch.manual_seed(0)
        model = self.make_model()
        model.eval()

        latents = torch.zeros(1, 3, 4, 8)
        actions = torch.zeros(1, 3, 2)

        predictions = model(latents, actions)

        self.assertFalse(
            torch.allclose(
                predictions[:, 0],
                predictions[:, 1],
            )
        )

    def test_patches_in_the_same_frame_can_interact(self):
        torch.manual_seed(0)
        model = self.make_model()
        model.eval()

        actions = torch.zeros(1, 3, 2)
        first_latents = torch.zeros(1, 3, 4, 8)
        second_latents = first_latents.clone()
        second_latents[:, 0, 1] = 10.0

        first_predictions = model(first_latents, actions)
        second_predictions = model(second_latents, actions)

        self.assertFalse(
            torch.allclose(
                first_predictions[:, 0, 0],
                second_predictions[:, 0, 0],
            )
        )

    def test_future_frames_do_not_change_earlier_predictions(self):
        torch.manual_seed(0)
        model = self.make_model()
        model.eval()

        first_latents = torch.randn(1, 3, 4, 8)
        first_actions = torch.randn(1, 3, 2)

        second_latents = first_latents.clone()
        second_actions = first_actions.clone()
        second_latents[:, 2] = second_latents[:, 2] + 10.0
        second_actions[:, 2] = second_actions[:, 2] + 10.0

        first_predictions = model(first_latents, first_actions)
        second_predictions = model(second_latents, second_actions)

        torch.testing.assert_close(
            first_predictions[:, :2],
            second_predictions[:, :2],
        )


if __name__ == "__main__":
    unittest.main()
