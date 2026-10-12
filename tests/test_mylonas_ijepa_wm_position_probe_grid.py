import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import evaluate_position_probe_grid


class MeanEncoder(torch.nn.Module):
    def forward(self, images):
        values = images.mean(dim=(1, 2, 3))
        return values[:, None, None].expand(-1, 1, 2)


class FirstTokenProbe(torch.nn.Module):
    def forward(self, tokens):
        return tokens[:, 0, :2]


class FakePointEnvironment:
    def __init__(self):
        self.data = SimpleNamespace(
            qpos=np.array([0.25, -0.5], dtype=np.float64),
            qvel=np.array([0.1, -0.2], dtype=np.float64),
        )

    def set_state(self, qpos, qvel):
        self.data.qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.data.qvel = np.asarray(qvel, dtype=np.float64).copy()


class FakeRenderEnvironment:
    def __init__(self):
        self.unwrapped = self
        self.point_env = FakePointEnvironment()

    def render(self):
        value = int((self.point_env.data.qpos[0] + 2.0) * 40)
        return np.full((224, 224, 3), value, dtype=np.uint8)


class PositionGridTest(unittest.TestCase):
    def test_excludes_points_inside_wall_clearance(self):
        positions = evaluate_position_probe_grid.generate_valid_grid_positions(
            x_limits=(-1.0, 1.0),
            y_limits=(-1.0, 1.0),
            wall_rectangles=np.array(
                [[-0.1, -1.0, 0.2, 2.0]],
                dtype=np.float32,
            ),
            grid_resolution=3,
            wall_clearance=0.0,
        )

        np.testing.assert_allclose(
            positions,
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 1.0],
                [1.0, 1.0],
            ],
        )

    def test_decodes_every_rendered_grid_position_and_restores_state(self):
        environment = FakeRenderEnvironment()
        original_qpos = environment.point_env.data.qpos.copy()
        original_qvel = environment.point_env.data.qvel.copy()
        positions = np.array(
            [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )

        predictions = evaluate_position_probe_grid.decode_grid_positions(
            env=environment,
            positions=positions,
            encoder=MeanEncoder(),
            position_probe=FirstTokenProbe(),
            position_mean=torch.zeros(2),
            position_std=torch.ones(2),
            device=torch.device("cpu"),
            batch_size=2,
            precision="fp32",
        )

        self.assertEqual(predictions.shape, (3, 2))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertLess(predictions[0, 0], predictions[-1, 0])
        np.testing.assert_allclose(
            environment.point_env.data.qpos,
            original_qpos,
        )
        np.testing.assert_allclose(
            environment.point_env.data.qvel,
            original_qvel,
        )

    def test_saves_heatmap_measurements_and_summary(self):
        positions = np.array(
            [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        predictions = np.array(
            [[-0.9, 0.0], [0.0, 0.2], [0.6, 0.0]],
            dtype=np.float32,
        )
        wall_rectangles = np.array(
            [[-0.5, 0.4, 1.0, 0.2]],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths, summary = (
                evaluate_position_probe_grid.save_grid_diagnostics(
                    positions=positions,
                    predictions=predictions,
                    wall_rectangles=wall_rectangles,
                    output_dir=temporary_directory,
                )
            )

            self.assertTrue(Path(paths["figure_path"]).is_file())
            self.assertTrue(Path(paths["data_path"]).is_file())
            self.assertTrue(Path(paths["summary_path"]).is_file())
            self.assertAlmostEqual(summary["max_error"], 0.4, places=6)
            with np.load(paths["data_path"]) as saved:
                np.testing.assert_allclose(saved["positions"], positions)
                np.testing.assert_allclose(
                    saved["predictions"],
                    predictions,
                )
                np.testing.assert_allclose(
                    saved["errors"],
                    [0.1, 0.2, 0.4],
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
