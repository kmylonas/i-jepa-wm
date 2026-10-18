import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze
from src.mylonas_ijepa_wm import plan_pointmaze_proprio
from src.mylonas_ijepa_wm.proprio_vit import ProprioViT


class StateDrivenModel(torch.nn.Module):
    num_hist = 1

    def forward(self, latents, states, actions):
        next_state = states + actions[..., :1]
        next_visual = next_state[..., :1].unsqueeze(-2)
        return next_visual, next_state


class ProprioPlanningTest(unittest.TestCase):
    def test_cli_loads_checkpoint_and_writes_episode_results(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return torch.zeros(len(images), 256, 1280, device=images.device)

        configuration = dict(
            num_patches=256, num_hist=1, num_t_blocks=1,
            ijepa_dim=1280, action_dim=10, action_embed_dim=4,
            proprio_dim=4, proprio_embed_dim=4, embed_dim=8,
            num_heads=2, mlp_dim=16, dropout=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            encoder_checkpoint = root / "encoder.pt"
            output = root / "planning.json"
            encoder_checkpoint.touch()
            torch.save({
                "epoch": 0,
                "model": ProprioViT(**configuration).state_dict(),
                "metadata": {
                    "model_type": "proprio_v1",
                    "camera": "topdown",
                    "proprio_fields": ("x", "y", "vx", "vy"),
                    "model_configuration": configuration,
                    "state_mean": torch.zeros(4),
                    "state_std": torch.ones(4),
                    "action_mean": None,
                    "action_std": None,
                },
            }, checkpoint)

            with patch.object(plan_pointmaze, "load_frozen_ijepa_encoder", return_value=TinyEncoder()):
                plan_pointmaze_proprio.main([
                    "--encoder-checkpoint", str(encoder_checkpoint),
                    "--world-model-checkpoint", str(checkpoint),
                    "--output", str(output),
                    "--num-episodes", "1", "--max-macro-steps", "1",
                    "--planning-horizon", "1", "--population", "2",
                    "--num-elites", "1", "--cem-iterations", "1",
                    "--candidate-batch-size", "2", "--device", "cpu",
                ])

            result = json.loads(output.read_text())
            self.assertEqual(result["configuration"]["camera"], "topdown")
            self.assertEqual(result["configuration"]["cost"], "latent")
            self.assertEqual(len(result["episodes"]), 1)

    def test_candidate_rollout_updates_predicted_proprioception(self):
        visual, states = plan_pointmaze_proprio.rollout_action_sequences(
            model=StateDrivenModel(),
            latent_history=torch.zeros(1, 1, 1, 1),
            state_history=torch.zeros(1, 1, 4),
            action_history=torch.empty(0, 1),
            candidate_actions=torch.tensor([[[1.0], [1.0]], [[2.0], [0.0]]]),
            device=torch.device("cpu"),
            chunk_size=1,
        )

        torch.testing.assert_close(
            visual[:, :, 0, 0], torch.tensor([[1.0, 2.0], [2.0, 2.0]])
        )
        torch.testing.assert_close(states[:, :, 0], torch.tensor([[1.0, 2.0], [2.0, 2.0]]))

    def test_episode_uses_real_proprioception_and_topdown_observations(self):
        class TinyEncoder(torch.nn.Module):
            def forward(self, images):
                return images.mean(dim=(1, 2, 3)).reshape(-1, 1, 1).expand(-1, 2, 3)

        class TinyModel(torch.nn.Module):
            num_hist = 2

            def forward(self, latents, states, actions):
                visual = latents + 0.001 * actions[..., :1].unsqueeze(-1)
                state = states + 0.001 * actions[..., :1]
                return visual, state

        environment = plan_pointmaze.create_environment(
            "PointMaze_UMaze-v3", max_micro_steps=5, camera="topdown"
        )
        frames = []
        try:
            record = plan_pointmaze_proprio.run_planning_episode(
                env=environment,
                encoder=TinyEncoder(),
                model=TinyModel(),
                device=torch.device("cpu"),
                seed=0,
                max_macro_steps=1,
                planning_horizon=1,
                population=4,
                num_elites=2,
                num_iterations=1,
                candidate_batch_size=2,
                state_mean=torch.zeros(4),
                state_std=torch.ones(4),
                frame_callback=frames.append,
            )
        finally:
            environment.close()

        self.assertEqual(len(record["proprio_states"]), 2)
        self.assertEqual(len(record["macro_actions"]), 1)
        self.assertEqual(len(frames), 6)
        np.testing.assert_allclose(
            np.asarray(record["proprio_states"])[:, :2],
            np.asarray(record["achieved_positions"]),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
