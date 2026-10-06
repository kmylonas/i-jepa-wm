"""Compare I-JEPA predictions with target-encoder representations on CPU."""

import argparse
import gc
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import src.models.vision_transformer as vit
from src.masks.utils import apply_masks


DEFAULT_CHECKPOINT = REPO_ROOT / "src/checkpoints/IN1K-vit.h.14-300e.pth.tar"
DEFAULT_IMAGE = (
    REPO_ROOT
    / "data/imagenette-mini/n03445777/ILSVRC2012_val_00008161.JPEG"
)


def strip_module_prefix(state_dict):
    """Return a state dict compatible with a model not wrapped in DDP."""
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def build_center_masks(grid_size=16, target_size=4):
    """Build one centered square target mask and its context complement."""
    if target_size < 1 or target_size > grid_size:
        raise ValueError("target_size must be between 1 and grid_size")

    start = (grid_size - target_size) // 2
    target_indices = [
        row * grid_size + col
        for row in range(start, start + target_size)
        for col in range(start, start + target_size)
    ]
    target_set = set(target_indices)
    context_indices = [
        index for index in range(grid_size**2) if index not in target_set
    ]
    return (
        torch.tensor([context_indices], dtype=torch.long),
        torch.tensor([target_indices], dtype=torch.long),
    )


def compare_representations(prediction, target):
    """Compare aligned predictions to targets and to shifted target patches."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if prediction.shape[1] < 2:
        raise ValueError("at least two target patches are required for comparison")

    correct = F.cosine_similarity(prediction, target, dim=-1)
    incorrect_target = torch.roll(target, shifts=1, dims=1)
    incorrect = F.cosine_similarity(prediction, incorrect_target, dim=-1)

    return {
        "smooth_l1": F.smooth_l1_loss(prediction, target).item(),
        "cosine_correct_mean": correct.mean().item(),
        "cosine_correct_min": correct.min().item(),
        "cosine_correct_max": correct.max().item(),
        "cosine_incorrect_mean": incorrect.mean().item(),
        "cosine_margin": (correct.mean() - incorrect.mean()).item(),
    }


def make_encoder():
    return vit.vit_huge(img_size=[224], patch_size=14)


def make_predictor(encoder):
    return vit.vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        depth=12,
        num_heads=encoder.num_heads,
    )


def load_image(path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def load_component(model, state_dict, name):
    result = model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"failed to load {name}: {result}")
    model.eval()
    return model


def run(checkpoint_path, image_path, target_size=4):
    print(f"Loading checkpoint index: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    required = {"encoder", "predictor", "target_encoder"}
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"checkpoint is missing components: {sorted(missing)}")

    image = load_image(image_path)
    context_mask, target_mask = build_center_masks(target_size=target_size)

    print("Loading and running target encoder...")
    target_encoder = load_component(
        make_encoder(), checkpoint["target_encoder"], "target_encoder"
    )
    with torch.inference_mode():
        full_target = target_encoder(image)
        full_target = F.layer_norm(full_target, (full_target.size(-1),))
        target = apply_masks(full_target, [target_mask])
    del full_target, target_encoder
    gc.collect()

    print("Loading and running context encoder and predictor...")
    encoder = load_component(make_encoder(), checkpoint["encoder"], "encoder")
    predictor = load_component(
        make_predictor(encoder), checkpoint["predictor"], "predictor"
    )
    with torch.inference_mode():
        context = encoder(image, [context_mask])
        prediction = predictor(context, [context_mask], [target_mask])

    metrics = compare_representations(prediction, target)
    print(f"Image: {image_path}")
    print(f"Input shape:      {tuple(image.shape)}")
    print(f"Context shape:    {tuple(context.shape)}")
    print(f"Prediction shape: {tuple(prediction.shape)}")
    print(f"Target shape:     {tuple(target.shape)}")
    print(f"Smooth-L1 loss:          {metrics['smooth_l1']:.6f}")
    print(f"Correct cosine mean:     {metrics['cosine_correct_mean']:.6f}")
    print(f"Correct cosine min/max:  {metrics['cosine_correct_min']:.6f} / "
          f"{metrics['cosine_correct_max']:.6f}")
    print(f"Incorrect cosine mean:   {metrics['cosine_incorrect_mean']:.6f}")
    print(f"Correct-minus-incorrect: {metrics['cosine_margin']:.6f}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--target-size", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.checkpoint, args.image, args.target_size)
