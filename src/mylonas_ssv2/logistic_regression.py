import torch
from torch import nn
import copy
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# from src.mylonas.vit import ViT

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)



def load_embedding_dataset(path):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    
    # Preserve endpoint order: first followed by last.
    features = torch.cat(
        [
            payload["z_first"],
            payload["z_last"],
        ],
        dim=1,
    ).float()

    targets = payload["targets"].long()

    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Feature/target count mismatch in {path}: "
            f"{features.shape[0]} != {targets.shape[0]}"
        )

    metadata = {
        "selected_label_ids": payload["selected_label_ids"],
        "original_to_target": payload["original_to_target"],
        "embedding_config": payload["embedding_config"],
        "input_dim": features.shape[1],
    }

    return TensorDataset(features, targets), metadata


def make_embedding_loaders(
    train_path,
    validation_path,
    batch_size=256,
):
    train_dataset, train_metadata = load_embedding_dataset(train_path)
    validation_dataset, validation_metadata = load_embedding_dataset(
        validation_path
    )

    # These must match, otherwise target 0 could represent different
    # actions in the two splits.
    if (
        train_metadata["selected_label_ids"]
        != validation_metadata["selected_label_ids"]
    ):
        raise ValueError(
            "Train and validation selected_label_ids do not match"
        )

    if (
        train_metadata["original_to_target"]
        != validation_metadata["original_to_target"]
    ):
        raise ValueError(
            "Train and validation class mappings do not match"
        )

    if (
        train_metadata["input_dim"]
        != validation_metadata["input_dim"]
    ):
        raise ValueError(
            "Train and validation input dimensions do not match"
        )

    # Makes training shuffling reproducible.
    generator = torch.Generator().manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, validation_loader, train_metadata

class LogisticRegression(nn.Module):
    def __init__(self, embed_dim, C):
        super().__init__()
        self.fc = nn.Linear(embed_dim, C)

    def forward(self, X):
        return self.fc(X)


def train(model, optim, train_loader, val_loader, epochs, device="cpu"):

    loss_fn = nn.CrossEntropyLoss()
    best_checkpoint_loss = float("inf")
    best_epoch = -1
    best_state = None
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        val_loss = 0.0
        
        model.train()
        for X,y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optim.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)
            
            loss.backward()
            optim.step()

            epoch_loss += loss.item()

        val_correct = 0
        val_samples = 0

        model.eval()
        with torch.no_grad():    
            for X,y in val_loader:
                X = X.to(device)
                y = y.to(device)

                batch_size = y.size(0)

                logits = model(X)
                loss = loss_fn(logits, y)
                
                val_loss += loss.item()
                val_samples += batch_size


                predictions = logits.argmax(dim=1)
                val_correct += (predictions == y).sum().item()


        average_val_loss = val_loss / len(val_loader)
        average_epoch_loss = epoch_loss / len(train_loader)
        val_accuracy = val_correct / val_samples
        
        if average_val_loss < best_checkpoint_loss:
            best_checkpoint_loss = average_val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"epoch: {epoch} train_loss: {average_epoch_loss:.4f} val_loss: {average_val_loss:.4f} val_accuracy: {val_accuracy}")

    print(f"Best model epoch was {best_epoch}. Loading model")
    model.load_state_dict(best_state)
    return model

        
# def evaluate(model, test_loader, device="cpu"):
   
#     model.eval()
#     loss_fn = nn.CrossEntropyLoss()
#     test_loss = 0.0
   
#     with torch.no_grad():
#         for X,y in test_loader:
#             X = X.to(device)
#             y = y.to(device)
            
#             logits = model(X)
#             loss = loss_fn(logits, y)
            
#             test_loss += loss.item()

#         print(f"test_loss: {test_loss / len(test_loader)}")

#####
def evaluate(
    model,
    test_loader,
    num_classes,
    class_ids,
    device="cpu",
    confusion_matrix_path=None,
):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    sample_count = 0

    all_targets = []
    all_predictions = []

    with torch.inference_mode():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device, dtype=torch.long)

            logits = model(X)
            loss = loss_fn(logits, y)

            predictions = logits.argmax(dim=1)
            batch_size = y.size(0)

            loss_sum += loss.item() * batch_size
            sample_count += batch_size

            all_targets.extend(y.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = loss_sum / sample_count

    accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        all_targets,
        all_predictions,
    )

    target_labels = list(range(num_classes))

    raw_confusion = confusion_matrix(
        all_targets,
        all_predictions,
        labels=target_labels,
    )

    normalized_confusion = confusion_matrix(
        all_targets,
        all_predictions,
        labels=target_labels,
        normalize="true",
    )

    print(f"Evaluation loss:     {average_loss:.4f}")
    print(f"Accuracy:            {100 * accuracy:.2f}%")
    print(
        f"Balanced accuracy:   "
        f"{100 * balanced_accuracy:.2f}%"
    )

    print("\nRaw confusion matrix:")
    print(raw_confusion)

    print("\nAxis labels:")
    for target, original_label_id in enumerate(class_ids):
        print(
            f"target {target:2d} -> "
            f"Something-Something label {original_label_id}"
        )

    if confusion_matrix_path is not None:
        confusion_matrix_path = Path(confusion_matrix_path)
        confusion_matrix_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Show the original Something-Something label IDs.
        display_labels = [
            str(label_id)
            for label_id in class_ids
        ]

        figure, axes = plt.subplots(
            1,
            2,
            figsize=(18, 8),
        )

        ConfusionMatrixDisplay(
            confusion_matrix=raw_confusion,
            display_labels=display_labels,
        ).plot(
            ax=axes[0],
            cmap="Blues",
            values_format="d",
            colorbar=False,
        )

        axes[0].set_title("Confusion matrix: counts")
        axes[0].set_xlabel("Predicted original label ID")
        axes[0].set_ylabel("True original label ID")
        axes[0].tick_params(axis="x", rotation=45)

        ConfusionMatrixDisplay(
            confusion_matrix=normalized_confusion,
            display_labels=display_labels,
        ).plot(
            ax=axes[1],
            cmap="Blues",
            values_format=".2f",
            colorbar=False,
        )

        axes[1].set_title("Confusion matrix: recall-normalized")
        axes[1].set_xlabel("Predicted original label ID")
        axes[1].set_ylabel("True original label ID")
        axes[1].tick_params(axis="x", rotation=45)

        figure.tight_layout()
        figure.savefig(
            confusion_matrix_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(
            f"Saved confusion matrix to "
            f"{confusion_matrix_path}"
        )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": raw_confusion,
        "normalized_confusion_matrix": normalized_confusion,
    }


###




def main():
    repo_root = Path(__file__).resolve().parents[2]

    train_path = (
        repo_root
        / "data"
        / "something_something_train_embeddings.pt"
    )

    validation_path = (
        repo_root
        / "data"
        / "something_something_validation_embeddings.pt"
    )

    train_loader, validation_loader, metadata = (
        make_embedding_loaders(
            train_path=train_path,
            validation_path=validation_path,
            batch_size=256,
        )
    )

    input_dim = metadata["input_dim"]
    num_classes = len(metadata["selected_label_ids"])

    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Input dimension:    {input_dim}")
    print(f"Number of classes:  {num_classes}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = LogisticRegression(
        embed_dim=input_dim,
        C=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    model = train(
        model=model,
        optim=optimizer,
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=100,
        device=device,
    )

    metrics = evaluate(
        model=model,
        test_loader=validation_loader,
        num_classes=num_classes,
        class_ids=metadata["selected_label_ids"],
        device=device,
        confusion_matrix_path=(
            repo_root
            / "data"
            / "validation_confusion_matrix.png"
        ),
    )


    checkpoint_path = repo_root / "data" / "something_something_logreg.pt"

    torch.save(
        {
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "input_dim": input_dim,
            "num_classes": num_classes,
            "selected_label_ids": metadata["selected_label_ids"],
            "original_to_target": metadata["original_to_target"],
        },
        checkpoint_path,
    )

    print(f"Saved classifier to {checkpoint_path}")


if __name__ == "__main__":
    main()


