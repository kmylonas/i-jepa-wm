import numpy as np
import torch
import tempfile
import unittest
from pathlib import Path

from mylonas_ijepa_wm.embed_frames import process_episode


class FakeEncoder(torch.nn.Module):
    def forward(self, images):
        return torch.zeros(
            (images.shape[0], 256, 1280),
            dtype=torch.float32,
            device=images.device,
        )


class ProcessEpisodeTest(unittest.TestCase):
    def test_uses_requested_output_directory_and_episode_name(self):
        episode = {
            "frames": torch.zeros((101, 224, 224, 3), dtype=torch.uint8),
            "actions": torch.arange(200, dtype=torch.float32).reshape(100, 2),
            "states": torch.arange(404, dtype=torch.float32).reshape(101, 4),
            "desired_goal": torch.tensor([1.0, 2.0]),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)

            process_episode(
                episode=episode,
                encoder=FakeEncoder(),
                device=torch.device("cpu"),
                episode_name="episode_0007",
                output_directory=output_directory,
            )

            latent_path = output_directory / "latents" / "episode_0007.npy"
            trajectory_path = output_directory / "trajectories" / "episode_0007.pt"

            self.assertTrue(latent_path.is_file())
            self.assertTrue(trajectory_path.is_file())

            latents = np.load(latent_path, mmap_mode="r")
            trajectory = torch.load(trajectory_path, weights_only=True)

            self.assertEqual(latents.shape, (21, 256, 1280))
            self.assertEqual(latents.dtype, np.float16)
            self.assertEqual(trajectory["macro_actions"].shape, (20, 10))
            self.assertEqual(trajectory["sampled_states"].shape, (21, 4))


if __name__ == "__main__":
    unittest.main()
