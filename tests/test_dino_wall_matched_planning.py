"""Checks for the matched DINO-WM Wall planning comparison."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.mylonas_ijepa_wm import compare_dino_wall_planning as matched


class MatchedWallPlanningTests(unittest.TestCase):
    def test_selects_only_failed_episodes_with_saved_start_and_goal(self):
        payload = {"episodes": [
            {"seed": 1, "success": True, "achieved_positions": [[1, 2]], "goal_position": [3, 4]},
            {"seed": 2, "success": False, "achieved_positions": [[5, 6], [7, 8]], "goal_position": [9, 10]},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planning.json"
            path.write_text(json.dumps(payload))
            cases = matched.load_failed_cases(path)
        self.assertEqual(cases, [{"seed": 2, "start": [5, 6], "goal": [9, 10]}])

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "data/dino_wm_reference/outputs/wall_single/checkpoints/model_latest.pth").is_file(),
        "released DINO-WM checkpoint is unavailable",
    )
    def test_chunkable_rollout_matches_released_model_rollout(self):
        torch.set_num_threads(2)
        model, _ = matched.load_dino_model()
        visual = matched.render_observation_at(
            matched.make_wall_adapter(10003), [24.15535545349121, 55.59099197387695]
        )
        state = torch.tensor([24.15535545349121, 55.59099197387695])
        mean = torch.tensor([31.894489288330078, 31.81402587890625])
        std = torch.tensor([17.663522720336914, 17.27516746520996])
        obs = {"visual": visual[None, None], "proprio": ((state - mean) / std)[None, None]}
        actions = torch.tensor([
            [[0.1] * 10, [-0.2] * 10],
            [[0.3] * 10, [0.4] * 10],
        ])
        with torch.inference_mode():
            encoded = model.encode_obs(obs)
            expected, _ = model.rollout(
                {key: value.expand(2, *value.shape[1:]) for key, value in obs.items()},
                actions,
            )
            actual = matched.rollout_encoded(model, encoded, actions)
        torch.testing.assert_close(actual["visual"], expected["visual"][:, -1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual["proprio"], expected["proprio"][:, -1], atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_released_attention_masks_move_with_model_to_mps(self):
        model, _ = matched.load_dino_model()
        matched.place_model(model, "mps")
        masks = [module.bias for module in model.modules()
                 if type(module).__module__ == "models.vit" and type(module).__name__ == "Attention"]
        self.assertTrue(masks)
        self.assertTrue(all(mask.device.type == "mps" for mask in masks))

    def test_optional_action_limit_applies_in_raw_wall_units(self):
        candidates = torch.tensor([[[4.0, -4.0] * 5]])
        limited = matched.limit_normalized_actions(
            candidates, torch.tensor([0.2, -0.2]), torch.tensor([0.5, 0.5]), 1.0
        )
        torch.testing.assert_close(limited, torch.tensor([[[1.6, -1.6] * 5]]))

    def test_native_run_can_resume_output_saved_before_action_limit_option(self):
        old = {"horizon": 5, "population": 300}
        current = {"horizon": 5, "population": 300, "action_limit": None}
        self.assertTrue(matched.settings_match(old, current))
        self.assertFalse(matched.settings_match(old, {**current, "action_limit": 1.0}))


if __name__ == "__main__":
    unittest.main()
