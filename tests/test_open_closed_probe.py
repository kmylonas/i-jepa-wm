import subprocess
import sys
import unittest
from pathlib import Path

import torch


try:
    from experiments.train_open_closed_probe import evaluate_leave_one_noun_out
except ImportError as exc:
    evaluate_leave_one_noun_out = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class OpenClosedProbeTest(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"open/closed probe is unavailable: {IMPORT_ERROR}")

    def test_leave_one_noun_out_probe_generalizes_shared_state_direction(self):
        embeddings = []
        labels = []
        nouns = []
        noun_offsets = {"book": 0.0, "door": 10.0, "gate": 20.0}
        for noun, offset in noun_offsets.items():
            for label, state_value in ((0, -2.0), (1, 2.0)):
                for variation in (-0.1, 0.1):
                    embeddings.append([state_value + variation, offset])
                    labels.append(label)
                    nouns.append(noun)

        payload = {
            "embeddings": torch.tensor(embeddings),
            "labels": torch.tensor(labels),
            "nouns": nouns,
        }

        results = evaluate_leave_one_noun_out(payload)

        self.assertEqual(
            [fold["test_noun"] for fold in results["folds"]],
            ["book", "door", "gate"],
        )
        for fold in results["folds"]:
            self.assertNotIn(fold["test_noun"], fold["train_nouns"])
            self.assertEqual(fold["train_size"], 8)
            self.assertEqual(fold["test_size"], 4)
            self.assertEqual(fold["balanced_accuracy"], 1.0)
            self.assertEqual(fold["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(results["aggregate"]["balanced_accuracy"], 1.0)
        self.assertEqual(results["aggregate"]["closed_recall"], 1.0)
        self.assertEqual(results["aggregate"]["open_recall"], 1.0)

    def test_rejects_metadata_that_is_not_aligned_with_embeddings(self):
        payload = {
            "embeddings": torch.zeros(3, 2),
            "labels": torch.tensor([0, 1]),
            "nouns": ["book", "door", "gate"],
        }

        with self.assertRaisesRegex(ValueError, "same number of samples"):
            evaluate_leave_one_noun_out(payload)

    def test_probe_script_can_be_invoked_by_path(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "experiments/train_open_closed_probe.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--embeddings", result.stdout)


if __name__ == "__main__":
    unittest.main()
