import unittest
import subprocess
import sys
from pathlib import Path

import torch


try:
    from experiments.compare_latents import (
        build_center_masks,
        compare_representations,
        strip_module_prefix,
    )
except ImportError as exc:
    build_center_masks = None
    compare_representations = None
    strip_module_prefix = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class CompareLatentsTest(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"experiment helpers are unavailable: {IMPORT_ERROR}")

    def test_center_masks_select_expected_square(self):
        context, target = build_center_masks(grid_size=4, target_size=2)

        self.assertEqual(target.tolist(), [[5, 6, 9, 10]])
        self.assertEqual(
            context.tolist(),
            [[0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15]],
        )

    def test_comparison_separates_correct_from_shifted_targets(self):
        prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target = prediction.clone()

        metrics = compare_representations(prediction, target)

        self.assertAlmostEqual(metrics["cosine_correct_mean"], 1.0)
        self.assertAlmostEqual(metrics["cosine_incorrect_mean"], 0.0)
        self.assertAlmostEqual(metrics["cosine_margin"], 1.0)
        self.assertAlmostEqual(metrics["smooth_l1"], 0.0)

    def test_module_prefix_is_removed_without_changing_unprefixed_keys(self):
        state = {
            "module.layer.weight": torch.tensor([1.0]),
            "norm.bias": torch.tensor([2.0]),
        }

        normalized = strip_module_prefix(state)

        self.assertEqual(set(normalized), {"layer.weight", "norm.bias"})
        self.assertIs(normalized["layer.weight"], state["module.layer.weight"])

    def test_script_can_be_invoked_by_path(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "experiments/compare_latents.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
