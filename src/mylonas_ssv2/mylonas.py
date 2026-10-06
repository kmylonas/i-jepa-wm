from pathlib import Path

from src.helper import init_model
from mylonas_ssv2.something_something_v2 import SomethingSomethingDataset
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from tqdm import tqdm



device="cpu"
patch_size=14
crop_size=224
pred_depth= 12
pred_emb_dim=384
model_name="vit_huge"


checkpoint_path = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "IN1K-vit.h.14-300e.pth.tar"
)



## Transform
IJEPA_MEAN = (0.485, 0.456, 0.406)
IJEPA_STD = (0.229, 0.224, 0.225)


ijepa_transform = transforms.Compose([
    # Resize the shorter image side while preserving aspect ratio
    transforms.Resize(
        256,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ),

    # Required input size for your ViT-H/14 checkpoint
    transforms.CenterCrop(224),

    # Convert PIL RGB image to float tensor in [0, 1]
    transforms.ToTensor(),

    # Normalization used by I-JEPA training
    transforms.Normalize(
        mean=IJEPA_MEAN,
        std=IJEPA_STD,
    ),
])



def load_checkpoint(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    pretrained_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["encoder"].items()
    }

    epoch = checkpoint['epoch']
    
    msg = encoder.load_state_dict(pretrained_dict)
    print(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    return encoder




def extract_pair_embeddings(encoder, loader, device):
    first_embeddings = []
    last_embeddings = []
    targets = []
    original_label_ids = []
    video_ids = []
    templates = []

    encoder.eval()
    encoder.requires_grad_(False)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            frames = batch["frames"]

            # frames: [B, 2, 3, 224, 224]
            batch_size, pair_size, channels, height, width = frames.shape

            # [B, 2, 3, 224, 224] -> [2B, 3, 224, 224]
            images = frames.reshape(
                batch_size * pair_size,
                channels,
                height,
                width,
            ).to(device)

            # [2B, 256, 1280]
            tokens = encoder(images)

            # Average the 256 patch tokens.
            pooled = tokens.mean(dim=1)

            # Optional but sensible for a linear probe.
            pooled = F.normalize(pooled, dim=1)

            # Restore first/last pairing: [B, 2, 1280]
            pairs = pooled.reshape(batch_size, pair_size, -1).cpu()

            first_embeddings.append(pairs[:, 0])
            last_embeddings.append(pairs[:, 1])

            targets.append(batch["target"])
            original_label_ids.append(batch["original_label_id"])
            video_ids.extend(batch["video_id"])
            templates.extend(batch["template"])
            # break

    return {
        "z_first": torch.cat(first_embeddings),
        "z_last": torch.cat(last_embeddings),
        "targets": torch.cat(targets),
        "original_label_ids": torch.cat(original_label_ids),
        "video_ids": video_ids,
        "templates": templates,
    }


def main():
    selected_label_ids = [
        34,35,36,37,38,39,40,41,42,43,44,45
        # Add the remaining selected classes here.
    ]

    split = "test"    

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )   

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)


    encoder = load_checkpoint(encoder, checkpoint_path)
    encoder.eval()



    dataset_root = (
        Path(__file__).resolve().parent
        / "datasets"
        / "something-something-v2"
    )

    dataset = SomethingSomethingDataset(
        dataset_root=dataset_root,
        split=split,
        transform=ijepa_transform,
        selected_label_ids=selected_label_ids,
    )

    print(f"Selected training samples: {len(dataset)}")
    print(f"Selected label IDs: {selected_label_ids}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,  # Start with 0; increase after verifying.
    )

    payload = extract_pair_embeddings(
        encoder=encoder,
        loader=loader,
        device=device,
    )

    

    payload["selected_label_ids"] = selected_label_ids
    payload["original_to_target"] = dataset.original_to_target
    payload["embedding_config"] = {
        "pooling": "mean",
        "l2_normalized": True,
        "crop_size": 224,
        "checkpoint": checkpoint_path.name,
    }

    output_path = Path(f"data/something_something_{split}_embeddings.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(payload, output_path)

    print(f"Saved embeddings to: {output_path}")
    print(f"First embeddings: {payload['z_first'].shape}")
    print(f"Last embeddings:  {payload['z_last'].shape}")
    print(f"Targets:          {payload['targets'].shape}")


if __name__ == "__main__":
    main()





# # -- init model
# encoder, predictor = init_model(
#     device=device,
#     patch_size=patch_size,
#     crop_size=crop_size,
#     pred_depth=pred_depth,
#     pred_emb_dim=pred_emb_dim,
#     model_name=model_name)


# encoder = load_checkpoint(encoder, checkpoint_path)
# encoder.eval()



# dataset = SomethingSomethingDataset(
#     dataset_root="src/datasets/something-something-v2",
#     split="train",
#     transform=ijepa_transform,
# )

# loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=4,
#     shuffle=False,
#     num_workers=0,
#     # persistent_workers=True,
# )


# for batch in loader:
#     print("Hello")
#     print(type(batch))
#     breakpoint()

