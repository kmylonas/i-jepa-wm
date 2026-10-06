import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


try:
    with (
        patch("src.helper.init_model", return_value=(MagicMock(), MagicMock())),
        patch(
            "torch.load",
            return_value={"epoch": 0, "encoder": {}},
        ),
        patch("builtins.breakpoint"),
    ):
        from mylonas.mylonas import (
            build_embedding_payload,
            extract_embeddings,
            load_frozen_encoder,
            save_embedding_payload,
        )
except ImportError as exc:
    build_embedding_payload = None
    extract_embeddings = None
    load_frozen_encoder = None
    save_embedding_payload = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class DeterministicPatchEncoder(torch.nn.Module):
    def forward(self, images):
        values = images[:, 0, 0, 0]
        token = torch.stack([values, torch.ones_like(values)], dim=1)
        return torch.stack([token, token], dim=1)


class MylonasEmbeddingTest(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"embedding helpers are unavailable: {IMPORT_ERROR}")

    def test_extract_embeddings_batches_mean_pools_and_normalizes(self):
        images = torch.zeros(3, 3, 2, 2)
        images[:, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
        encoder = DeterministicPatchEncoder()

        embeddings = extract_embeddings(
            encoder,
            images,
            batch_size=2,
            device="cpu",
        )

        expected = torch.tensor(
            [
                [1.0 / 2**0.5, 1.0 / 2**0.5],
                [2.0 / 5**0.5, 1.0 / 5**0.5],
                [3.0 / 10**0.5, 1.0 / 10**0.5],
            ]
        )
        torch.testing.assert_close(embeddings, expected)
        self.assertFalse(encoder.training)
        self.assertFalse(embeddings.requires_grad)

    def test_checkpoint_loader_strips_distributed_prefix_and_freezes_encoder(self):
        expected_weight = torch.tensor([[2.0, 3.0]])
        expected_bias = torch.tensor([4.0])

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            torch.save(
                {
                    "epoch": 7,
                    "encoder": {
                        "module.weight": expected_weight,
                        "module.bias": expected_bias,
                    },
                },
                checkpoint_path,
            )
            encoder, epoch = load_frozen_encoder(
                checkpoint_path,
                device="cpu",
                encoder_factory=lambda: torch.nn.Linear(2, 1),
            )

        self.assertEqual(epoch, 7)
        torch.testing.assert_close(encoder.weight, expected_weight)
        torch.testing.assert_close(encoder.bias, expected_bias)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_build_payload_preserves_sample_alignment(self):
        images = torch.zeros(3, 3, 2, 2)
        images[:, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
        prepared_dataset = {
            "images": images,
            "labels": torch.tensor([0, 1, 0]),
            "states": ["closed", "open", "closed"],
            "nouns": ["book", "door", "gate"],
            "paths": ["closed book/a.jpg", "open door/b.jpg", "closed gate/c.jpg"],
            "label_names": ["closed", "open"],
            "preprocessing": {"center_crop": 224},
        }

        payload = build_embedding_payload(
            prepared_dataset,
            DeterministicPatchEncoder(),
            batch_size=2,
            device="cpu",
        )

        self.assertEqual(tuple(payload["embeddings"].shape), (3, 2))
        torch.testing.assert_close(payload["labels"], torch.tensor([0, 1, 0]))
        self.assertEqual(payload["states"], ["closed", "open", "closed"])
        self.assertEqual(payload["nouns"], ["book", "door", "gate"])
        self.assertEqual(
            payload["paths"],
            ["closed book/a.jpg", "open door/b.jpg", "closed gate/c.jpg"],
        )
        self.assertEqual(payload["embedding_config"]["pooling"], "mean")
        self.assertTrue(payload["embedding_config"]["l2_normalized"])

    def test_saved_payload_can_be_loaded_for_logistic_regression(self):
        payload = {
            "embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "labels": torch.tensor([0, 1]),
            "states": ["closed", "open"],
            "nouns": ["book", "door"],
            "paths": ["closed book/a.jpg", "open door/b.jpg"],
            "label_names": ["closed", "open"],
            "preprocessing": {},
            "embedding_config": {
                "pooling": "mean",
                "l2_normalized": True,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "embeddings.pt"
            save_embedding_payload(payload, output_path)
            loaded = torch.load(output_path, map_location="cpu", weights_only=True)

        torch.testing.assert_close(loaded["embeddings"], payload["embeddings"])
        torch.testing.assert_close(loaded["labels"], payload["labels"])
        self.assertEqual(loaded["nouns"], payload["nouns"])


if __name__ == "__main__":
    unittest.main()
