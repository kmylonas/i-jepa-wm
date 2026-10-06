import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image


try:
    from src.datasets.mit_states_open_closed import (
        OpenClosedMITStatesDataset,
        save_preprocessed_dataset,
    )
except ImportError as exc:
    OpenClosedMITStatesDataset = None
    save_preprocessed_dataset = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class OpenClosedMITStatesDatasetTest(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.fail(f"MIT States dataset helpers are unavailable: {IMPORT_ERROR}")

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.image_root = Path(self.temp_dir.name) / "images"

        for noun in ("book", "door", "gate"):
            for state in ("closed", "open"):
                folder = self.image_root / f"{state} {noun}"
                folder.mkdir(parents=True)
                color = (255, 0, 0) if state == "open" else (0, 0, 255)
                Image.new("RGB", (256, 256), color).save(folder / "sample.png")

        ignored = self.image_root / "open window"
        ignored.mkdir()
        Image.new("RGB", (256, 256), (0, 255, 0)).save(ignored / "ignored.png")

    def test_discovers_only_requested_noun_state_folders_with_binary_labels(self):
        dataset = OpenClosedMITStatesDataset(self.image_root)

        self.assertEqual(len(dataset), 6)
        observed = {
            (sample["noun"], sample["state"], sample["label"])
            for sample in dataset.samples
        }
        self.assertEqual(
            observed,
            {
                ("book", "closed", 0),
                ("book", "open", 1),
                ("door", "closed", 0),
                ("door", "open", 1),
                ("gate", "closed", 0),
                ("gate", "open", 1),
            },
        )

    def test_getitem_returns_normalized_ijepa_input_and_metadata(self):
        dataset = OpenClosedMITStatesDataset(self.image_root)

        item = next(
            dataset[index]
            for index, sample in enumerate(dataset.samples)
            if sample["noun"] == "book" and sample["state"] == "open"
        )

        self.assertEqual(tuple(item["image"].shape), (3, 224, 224))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertEqual(item["label"], 1)
        self.assertEqual(item["state"], "open")
        self.assertEqual(item["noun"], "book")
        torch.testing.assert_close(
            item["image"][:, 0, 0],
            torch.tensor(
                [
                    (1.0 - 0.485) / 0.229,
                    (0.0 - 0.456) / 0.224,
                    (0.0 - 0.406) / 0.225,
                ]
            ),
        )

    def test_save_materializes_images_labels_and_metadata(self):
        dataset = OpenClosedMITStatesDataset(self.image_root)
        output_path = Path(self.temp_dir.name) / "prepared" / "open_closed.pt"

        save_preprocessed_dataset(dataset, output_path)

        saved = torch.load(output_path, map_location="cpu", weights_only=True)
        self.assertEqual(tuple(saved["images"].shape), (6, 3, 224, 224))
        self.assertEqual(saved["labels"].dtype, torch.long)
        self.assertEqual(sorted(saved["labels"].tolist()), [0, 0, 0, 1, 1, 1])
        self.assertEqual(set(saved["nouns"]), {"book", "door", "gate"})
        self.assertEqual(saved["label_names"], ["closed", "open"])
        self.assertEqual(len(saved["paths"]), 6)

    def test_builder_script_can_be_invoked_by_path(self):
        repo_root = Path(__file__).resolve().parents[1]

        result = subprocess.run(
            [sys.executable, "experiments/build_open_closed_dataset.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--image-root", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
