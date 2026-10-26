from hashlib import sha256

import pytest
import torch


def _released(path, *, omit=None, wrong_shape=False):
    source = torch.nn.Linear(2, 2)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(0.5)
    state = {f"module.{key}": value for key, value in source.state_dict().items()}
    if wrong_shape:
        state = {**state, "module.weight": torch.zeros(3, 2)}
    checkpoint = {
        "encoder": state,
        "predictor": state,
        "target_encoder": state,
        "epoch": 299,
        "opt": {"unrelated": True},
    }
    if omit:
        del checkpoint[omit]
    torch.save(checkpoint, path)


def test_released_checkpoint_initializes_all_models_without_optimizer_state(tmp_path):
    from src.helper import initialize_from_released_checkpoint

    path = tmp_path / "released.pt"
    _released(path)
    modules = [torch.nn.Linear(2, 2) for _ in range(3)]
    identity = initialize_from_released_checkpoint(path, *modules)
    for module in modules:
        torch.testing.assert_close(module.weight, torch.full((2, 2), 0.25))
        torch.testing.assert_close(module.bias, torch.full((2,), 0.5))
    assert identity["epoch"] == 299
    assert identity["sha256"] == sha256(path.read_bytes()).hexdigest()
    assert "opt" not in identity


@pytest.mark.parametrize("missing", ["encoder", "predictor", "target_encoder"])
def test_released_checkpoint_requires_all_components(tmp_path, missing):
    from src.helper import initialize_from_released_checkpoint

    path = tmp_path / "incomplete.pt"
    _released(path, omit=missing)
    with pytest.raises(ValueError, match=missing):
        initialize_from_released_checkpoint(path, *(torch.nn.Linear(2, 2) for _ in range(3)))


def test_released_checkpoint_rejects_architecture_mismatch(tmp_path):
    from src.helper import initialize_from_released_checkpoint

    path = tmp_path / "wrong_shape.pt"
    _released(path, wrong_shape=True)
    with pytest.raises((RuntimeError, ValueError), match="weight"):
        initialize_from_released_checkpoint(path, *(torch.nn.Linear(2, 2) for _ in range(3)))
