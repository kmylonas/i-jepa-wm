from pathlib import Path
import json
import torch
import av
# from PIL import Image


def load_first_last_frames(video_path):
    video_path = Path(video_path)

    first_frame = None
    last_frame = None

    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_image().convert("RGB")

            if first_frame is None:
                first_frame = image.copy()

            last_frame = image.copy()

    if first_frame is None:
        raise ValueError(f"No video frames could be decoded from {video_path}")

    return first_frame, last_frame


def normalize_template(template):
    return template.replace("[", "").replace("]", "")


class SomethingSomethingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        transform,
        selected_label_ids=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        self.video_root = (
            self.dataset_root
            / "videos"
            / "20bn-something-something-v2"
        )

        with open(self.dataset_root / "labels" / "labels.json") as file:
            raw_label_to_id = json.load(file)

        self.label_to_id = {
            template: int(label_id)
            for template, label_id in raw_label_to_id.items()
        }

        with open(self.dataset_root / "labels" / f"{split}.json") as file:
                annotations = json.load(file)

        if selected_label_ids is None:
            selected_label_ids = sorted(self.label_to_id.values())

        # Preserve the order supplied by the user.
        self.selected_label_ids = list(selected_label_ids)

        # Original Something-Something ID -> contiguous classifier target.
        self.original_to_target = {
            original_id: target
            for target, original_id in enumerate(self.selected_label_ids)
        }

        # Filter before decoding any videos.
        self.annotations = []

        for annotation in annotations:
            template = normalize_template(annotation["template"])
            original_label_id = self.label_to_id[template]

            if original_label_id in self.original_to_target:
                self.annotations.append(annotation)


    def __len__(self):
            return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]

        video_id = annotation["id"]
        template = normalize_template(annotation["template"])

        original_label_id = self.label_to_id[template]
        target = self.original_to_target[original_label_id]

        video_path = self.video_root / f"{video_id}.webm"

        first_frame, last_frame = load_first_last_frames(video_path)

        frames = torch.stack([
            self.transform(first_frame),
            self.transform(last_frame),
        ])

        return {
            "frames": frames,
            "target": target,
            "original_label_id": original_label_id,
            "video_id": video_id,
            "template": template,
        }