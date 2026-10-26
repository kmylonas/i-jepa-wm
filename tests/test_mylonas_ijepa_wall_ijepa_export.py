import pytest
import torch


def _best(path, encoder=None, *, val_loss=0.1, best_val_loss=0.1):
    if encoder is None:
        encoder = {"module.weight": torch.full((2, 2), 0.25),
                   "module.bias": torch.full((2,), 0.5)}
    checkpoint = {
        "encoder": encoder,
        "target_encoder": {"module.weight": torch.zeros(2, 2),
                           "module.bias": torch.zeros(2)},
        "epoch": 3,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "wall_run_identity": {"manifest_sha256": "abc", "source_checkpoint_sha256": "def"},
    }
    torch.save(checkpoint, path)


def test_export_uses_online_encoder_and_existing_embedding_format(tmp_path):
    from src.mylonas_ijepa_wm.embed_frames import load_checkpoint
    from src.mylonas_ijepa_wm.export_wall_ijepa_encoder import export_encoder

    best = tmp_path / "jepa-best.pth.tar"
    output = tmp_path / "jepa-best.encoder-only.pth.tar"
    _best(best)
    export_encoder(best, output)
    exported = torch.load(output, weights_only=True)
    assert exported["epoch"] == 3
    assert set(exported["encoder"]) == {"weight", "bias"}
    torch.testing.assert_close(exported["encoder"]["weight"], torch.full((2, 2), 0.25))
    torch.testing.assert_close(exported["encoder"]["bias"], torch.full((2,), 0.5))
    assert exported["wall_run_identity"]["manifest_sha256"] == "abc"
    model = torch.nn.Linear(2, 2)
    load_checkpoint(model, output)
    torch.testing.assert_close(model.weight, torch.full((2, 2), 0.25))


@pytest.mark.parametrize("encoder", [{}, {"module.weight": "bad"}])
def test_export_rejects_malformed_encoder(tmp_path, encoder):
    from src.mylonas_ijepa_wm.export_wall_ijepa_encoder import export_encoder

    best = tmp_path / "jepa-best.pth.tar"
    _best(best, encoder=encoder)
    with pytest.raises(ValueError, match="encoder"):
        export_encoder(best, tmp_path / "out.pt")


def test_export_rejects_nonbest_checkpoint(tmp_path):
    from src.mylonas_ijepa_wm.export_wall_ijepa_encoder import export_encoder

    best = tmp_path / "jepa-best.pth.tar"
    _best(best, val_loss=0.2, best_val_loss=0.1)
    with pytest.raises(ValueError, match="best"):
        export_encoder(best, tmp_path / "out.pt")
