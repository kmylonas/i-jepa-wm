"""Measure invariance of global I-JEPA representations on Imagenette Mini."""

import argparse
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from experiments.compare_latents import (
    DEFAULT_CHECKPOINT,
    load_component,
    make_encoder,
)


DEFAULT_DATASET = REPO_ROOT / "data/imagenette-mini"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONDITIONS = (
    ("gaussian_noise", 0.02),
    ("gaussian_noise", 0.05),
    ("gaussian_noise", 0.10),
    ("gaussian_blur", 0.50),
    ("gaussian_blur", 1.00),
    ("gaussian_blur", 2.00),
    ("darkening", 0.90),
    ("darkening", 0.70),
    ("darkening", 0.50),
    ("brightening", 1.10),
    ("brightening", 1.30),
    ("brightening", 1.50),
    ("contrast_reduction", 0.90),
    ("contrast_reduction", 0.70),
    ("contrast_reduction", 0.50),
)


def mean_pool_and_normalize(patch_tokens):
    return F.normalize(patch_tokens.mean(dim=1), dim=-1)


def apply_corruption(image, name, severity, seed=0):
    """Apply one deterministic corruption to a CHW tensor in [0, 1]."""
    if name == "gaussian_noise":
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
        return (image + severity * noise).clamp(0.0, 1.0)
    if name == "gaussian_blur":
        kernel_size = max(3, 2 * math.ceil(3 * severity) + 1)
        return TF.gaussian_blur(
            image, kernel_size=[kernel_size, kernel_size], sigma=[severity, severity]
        )
    if name in {"darkening", "brightening"}:
        return TF.adjust_brightness(image, severity).clamp(0.0, 1.0)
    if name == "contrast_reduction":
        return TF.adjust_contrast(image, severity).clamp(0.0, 1.0)
    raise ValueError(f"unknown corruption: {name}")


def compute_invariance_metrics(clean_embeddings, transformed_embeddings):
    if clean_embeddings.shape != transformed_embeddings.shape:
        raise ValueError("clean and transformed embedding shapes must match")
    if clean_embeddings.shape[0] < 2:
        raise ValueError("at least two images are required for negative comparisons")

    similarities = transformed_embeddings @ clean_embeddings.T
    positive = similarities.diagonal()
    negative_mask = ~torch.eye(
        similarities.shape[0], dtype=torch.bool, device=similarities.device
    )
    hardest_negative = similarities.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
    retrieval = similarities.argmax(dim=1) == torch.arange(similarities.shape[0])

    return {
        "positive_cosine": positive.mean().item(),
        "hardest_negative_cosine": hardest_negative.mean().item(),
        "margin": (positive - hardest_negative).mean().item(),
        "retrieval_accuracy": retrieval.float().mean().item(),
    }


def load_canonical_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = TF.resize(image, 256)
        image = TF.center_crop(image, [224, 224])
        return TF.to_tensor(image)


def normalize_image(image):
    return TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def select_images(dataset_path):
    class_directories = sorted(path for path in dataset_path.iterdir() if path.name.startswith("n"))
    paths = []
    for class_directory in class_directories:
        candidates = sorted(class_directory.glob("*.JPEG"))
        if not candidates:
            raise FileNotFoundError(f"no JPEG images in {class_directory}")
        paths.append(candidates[0])
    if len(paths) < 2:
        raise ValueError(f"expected at least two class directories in {dataset_path}")
    return paths


def extract_embeddings(encoder, images, batch_size):
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack(images[start:start + batch_size])
            embeddings.append(mean_pool_and_normalize(encoder(batch)))
    return torch.cat(embeddings)


def run(checkpoint_path, dataset_path, batch_size=2, seed=0):
    image_paths = select_images(dataset_path)
    canonical_images = [load_canonical_image(path) for path in image_paths]

    print(f"Loading I-JEPA encoder from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "encoder" not in checkpoint:
        raise KeyError("checkpoint does not contain an encoder")
    encoder = load_component(make_encoder(), checkpoint["encoder"], "encoder")

    clean_images = [normalize_image(image) for image in canonical_images]
    clean_embeddings = extract_embeddings(encoder, clean_images, batch_size)

    print(f"Images: {len(image_paths)} (one per Imagenette class)")
    print("\ntransformation       severity  positive   hardest-neg  margin     retrieval")
    print("-" * 75)
    for name, severity in CONDITIONS:
        transformed_images = [
            normalize_image(
                apply_corruption(image, name, severity, seed=seed + image_index)
            )
            for image_index, image in enumerate(canonical_images)
        ]
        transformed_embeddings = extract_embeddings(
            encoder, transformed_images, batch_size
        )
        metrics = compute_invariance_metrics(clean_embeddings, transformed_embeddings)
        print(
            f"{name:<20} {severity:>7.2f}  "
            f"{metrics['positive_cosine']:>8.4f}   "
            f"{metrics['hardest_negative_cosine']:>11.4f}  "
            f"{metrics['margin']:>8.4f}   "
            f"{metrics['retrieval_accuracy']:>8.1%}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.checkpoint, args.dataset, args.batch_size, args.seed)
