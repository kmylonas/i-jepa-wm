import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import wm_proprio
from src.mylonas_ijepa_wm.proprio_vit import ProprioViT


class ProprioDatasetTest(unittest.TestCase):
    def test_sample_aligns_latents_states_and_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            latent_dir = root / "latents"
            trajectory_dir = root / "trajectories"
            latent_dir.mkdir()
            trajectory_dir.mkdir()
            latents = np.arange(5, dtype=np.float16).reshape(5, 1, 1)
            states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
            actions = torch.arange(4, dtype=torch.float32).reshape(4, 1)
            np.save(latent_dir / "episode_0007.npy", latents)
            torch.save(
                {"sampled_states": states, "macro_actions": actions},
                trajectory_dir / "episode_0007.pt",
            )

            dataset = wm_proprio.ProprioWorldModelDataset(
                latent_dir=latent_dir,
                trajectory_dir=trajectory_dir,
                episode_ids=[7],
                num_hist=2,
                rollout_steps=2,
                state_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
                state_std=torch.ones(4),
            )
            sample = dataset[1]

            torch.testing.assert_close(
                sample["latent_window"].flatten().float(),
                torch.tensor([1.0, 2.0, 3.0, 4.0]),
            )
            torch.testing.assert_close(
                sample["state_window"],
                states[1:5] - torch.tensor([1.0, 2.0, 3.0, 4.0]),
            )
            torch.testing.assert_close(sample["actions"].flatten(), actions[1:4].flatten())

    def test_rejects_state_sequence_not_aligned_with_latents(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "latents").mkdir()
            (root / "trajectories").mkdir()
            np.save(root / "latents" / "episode_0000.npy", np.zeros((5, 1, 1), dtype=np.float16))
            torch.save(
                {
                    "sampled_states": torch.zeros(4, 4),
                    "macro_actions": torch.zeros(4, 2),
                },
                root / "trajectories" / "episode_0000.pt",
            )

            with self.assertRaisesRegex(ValueError, "sampled_states"):
                wm_proprio.ProprioWorldModelDataset(
                    root / "latents",
                    root / "trajectories",
                    episode_ids=[0],
                    num_hist=2,
                )


class ProprioModelTest(unittest.TestCase):
    def test_one_step_training_uses_state_target_and_one_model_call(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, latents, states, actions):
                self.calls += 1
                return latents + 1.0, states + 1.0

        model = CountingModel()
        latents = torch.zeros(1, 3, 1, 1)
        states = torch.zeros(1, 3, 4)
        states[:, 2] = 4.0
        actions = torch.zeros(1, 2, 1)

        metrics = wm_proprio.compute_world_model_loss(
            model,
            latents,
            states,
            actions,
            num_hist=2,
            proprio_loss_weight=0.1,
        )

        self.assertEqual(model.calls, 1)
        self.assertAlmostEqual(float(metrics["one_step_visual_loss"]), 1.0)
        self.assertAlmostEqual(float(metrics["one_step_proprio_loss"]), 5.0)
        self.assertAlmostEqual(float(metrics["loss"]), 1.5, places=5)

    def test_predicts_visual_and_state_and_uses_proprio_input(self):
        torch.manual_seed(0)
        model = ProprioViT(
            num_patches=2,
            num_hist=2,
            num_t_blocks=1,
            ijepa_dim=3,
            action_dim=2,
            action_embed_dim=4,
            proprio_dim=4,
            proprio_embed_dim=4,
            embed_dim=8,
            num_heads=2,
            mlp_dim=16,
            dropout=0.0,
        ).eval()
        latents = torch.randn(1, 2, 2, 3)
        actions = torch.randn(1, 2, 2)
        states = torch.randn(1, 2, 4)

        predicted_visual, predicted_state = model(latents, states, actions)
        changed_visual, changed_state = model(latents, states + 2.0, actions)

        self.assertEqual(tuple(predicted_visual.shape), (1, 2, 2, 3))
        self.assertEqual(tuple(predicted_state.shape), (1, 2, 4))
        self.assertFalse(torch.allclose(predicted_visual, changed_visual))
        self.assertFalse(torch.allclose(predicted_state, changed_state))

    def test_rollout_uses_predicted_state_after_first_step(self):
        class StateDrivenModel(torch.nn.Module):
            def forward(self, latents, states, actions):
                next_state = states + actions[..., :1]
                next_visual = next_state[..., :1].unsqueeze(-2)
                return next_visual, next_state

        visual, state = wm_proprio.autoregressive_rollout(
            StateDrivenModel(),
            latent_history=torch.zeros(1, 1, 1, 1),
            state_history=torch.zeros(1, 1, 4),
            actions=torch.ones(1, 2, 1),
        )

        torch.testing.assert_close(visual.flatten(), torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(state[:, :, 0].flatten(), torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
