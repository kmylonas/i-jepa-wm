import copy
import json

import pytest
import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, masks=None):
        x = images.mean(dim=(1, 2, 3)).view(-1, 1, 1).expand(-1, 16, 2) * self.scale
        if masks is None:
            return x
        from src.masks.utils import apply_masks
        return apply_masks(x, masks)


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.8))

    def forward(self, tokens, masks_enc, masks_pred):
        return tokens.mean(dim=1, keepdim=True).expand(-1, masks_pred[0].shape[1], -1) * self.scale


class FixedMaskCollator:
    def __init__(self, **kwargs):
        self.count = 0

    def __call__(self, batch):
        from torch.utils.data import default_collate
        self.count += 1
        collated = default_collate(batch)
        n = len(batch)
        return collated, [torch.tensor([[0, 1]] * n)], [torch.tensor([[2]] * n)]


class TinyWallFrames:
    constructed_splits = []

    def __init__(self, raw_dir, index, split):
        self.constructed_splits.append(split)
        self.length = 2 if split == "train" else 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index + 1)), 0


def _config(tmp_path, source_path):
    return {
        "meta": {"use_bfloat16": False, "model_name": "vit_huge", "load_checkpoint": False,
                 "read_checkpoint": None, "copy_data": False, "pred_depth": 12,
                 "pred_emb_dim": 384, "init_checkpoint": str(source_path)},
        "data": {"dataset": "wall", "raw_dir": str(tmp_path / "raw"),
                 "episode_manifest": str(tmp_path / "manifest.json"), "frame_stride": 5,
                 "batch_size": 1, "pin_mem": False, "num_workers": 0, "crop_size": 4,
                 "use_gaussian_blur": False, "use_horizontal_flip": False,
                 "use_color_distortion": False, "color_jitter_strength": 0.0,
                 "crop_scale": [1.0, 1.0]},
        "mask": {"allow_overlap": False, "patch_size": 2, "num_enc_masks": 1,
                 "min_keep": 1, "enc_mask_scale": [0.5, 0.5],
                 "num_pred_masks": 1, "pred_mask_scale": [0.25, 0.25],
                 "aspect_ratio": [1.0, 1.0]},
        "optimization": {"ema": [0.5, 0.5], "ipe_scale": 1.0, "weight_decay": 0.04,
                         "final_weight_decay": 0.04, "epochs": 2, "warmup": 0,
                         "start_lr": 0.01, "lr": 0.01, "final_lr": 0.001,
                         "accumulation_steps": 2},
        "logging": {"folder": str(tmp_path / "run"), "write_tag": "jepa"},
    }


def _setup(tmp_path, monkeypatch):
    from src import train

    source = tmp_path / "released.pt"
    encoder, predictor = TinyEncoder(), TinyPredictor()
    torch.save({"encoder": {"module.scale": encoder.scale.detach().clone()},
                "predictor": {"module.scale": predictor.scale.detach().clone()},
                "target_encoder": {"module.scale": encoder.scale.detach().clone()},
                "epoch": 299, "opt": {"junk": True}}, source)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(train, "init_model", lambda **kwargs: (TinyEncoder(), TinyPredictor()))
    monkeypatch.setattr(train, "MBMaskCollator", FixedMaskCollator)
    monkeypatch.setattr(train, "build_frame_index", lambda raw, manifest, frame_stride: {
        "manifest_sha256": "manifest-hash", "frame_stride": frame_stride,
        "splits": {"train": [], "val": [], "test": []},
        "sample_counts": {"train": 2, "val": 1, "test": 0},
    }, raising=False)
    monkeypatch.setattr(train, "WallFrames", TinyWallFrames, raising=False)
    monkeypatch.setattr(train, "load_manifest", lambda path: {}, raising=False)
    monkeypatch.setattr(train, "make_imagenet1k", lambda **kwargs: pytest.fail("ImageNet loader used for Wall"))
    monkeypatch.setattr(train, "init_distributed", lambda: (1, 0))
    return train, _config(tmp_path, source)


def test_wall_mode_uses_existing_train_loop_and_selects_validation_best(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    TinyWallFrames.constructed_splits.clear()
    train.main(config)
    run = tmp_path / "run"
    assert TinyWallFrames.constructed_splits == ["train", "val"]
    best = torch.load(run / "jepa-best.pth.tar", weights_only=True)
    latest = torch.load(run / "jepa-latest.pth.tar", weights_only=True)
    assert best["epoch"] in (1, 2)
    assert latest["epoch"] == 2
    assert best["encoder"]["scale"] != torch.tensor(1.0)
    assert best["predictor"]["scale"] != torch.tensor(0.8)
    assert latest["target_encoder"]["scale"] != torch.tensor(1.0)
    assert len(json.loads((run / "history.json").read_text())) == 2
    assert latest["wall_run_identity"]["manifest_sha256"] == "manifest-hash"
    assert latest["optimizer_updates"] == 2


def test_wall_resume_rejects_changed_frame_stride(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    train.main(config)
    original_params = (tmp_path / "run" / "params-ijepa.yaml").read_bytes()
    changed = copy.deepcopy(config)
    changed["meta"]["load_checkpoint"] = True
    changed["data"]["frame_stride"] = 1
    with pytest.raises(ValueError, match="identity"):
        train.main(changed)
    assert (tmp_path / "run" / "params-ijepa.yaml").read_bytes() == original_params


def test_fresh_wall_run_rejects_existing_output_without_changes(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    train.main(config)
    run = tmp_path / "run"
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    changed = copy.deepcopy(config)
    changed["data"]["frame_stride"] = 1
    with pytest.raises(FileExistsError, match="existing Wall run"):
        train.main(changed)
    assert {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()} == before


def test_wall_resume_rejects_changed_world_size(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    train.main(config)
    monkeypatch.setattr(train, "init_distributed", lambda: (2, 0))
    resumed = copy.deepcopy(config)
    resumed["meta"]["load_checkpoint"] = True
    with pytest.raises(ValueError, match="identity"):
        train.main(resumed)


def test_wall_mode_rejects_nondeterministic_mask_workers(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    config["data"]["num_workers"] = 2
    with pytest.raises(ValueError, match="num_workers"):
        train.main(config)


def test_wall_resume_reproduces_uninterrupted_final_weights(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    original_save = train._atomic_torch_save
    interrupted = False

    def stop_after_first_latest(value, path):
        nonlocal interrupted
        original_save(value, path)
        if not interrupted and str(path).endswith("jepa-latest.pth.tar"):
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(train, "_atomic_torch_save", stop_after_first_latest)
    with pytest.raises(RuntimeError, match="interruption"):
        train.main(config)
    monkeypatch.setattr(train, "_atomic_torch_save", original_save)
    resumed_config = copy.deepcopy(config)
    resumed_config["meta"]["load_checkpoint"] = True
    train.main(resumed_config)
    resumed = torch.load(tmp_path / "run" / "jepa-latest.pth.tar", weights_only=True)

    control_config = copy.deepcopy(config)
    control_config["logging"]["folder"] = str(tmp_path / "control")
    train.main(control_config)
    control = torch.load(tmp_path / "control" / "jepa-latest.pth.tar", weights_only=True)
    for name in ("encoder", "predictor", "target_encoder"):
        for key in control[name]:
            torch.testing.assert_close(resumed[name][key], control[name][key], rtol=0, atol=0)
    assert resumed["history"] == control["history"]


def test_resume_uses_best_if_final_latest_save_is_interrupted(tmp_path, monkeypatch):
    train, config = _setup(tmp_path, monkeypatch)
    original_save = train._atomic_torch_save
    interrupted = False
    validation_losses = iter((2.0, 1.0))
    monkeypatch.setattr(train, "evaluate_wall_loss", lambda *args, **kwargs: next(validation_losses))

    def fail_before_final_latest(value, path):
        nonlocal interrupted
        if (not interrupted and str(path).endswith("jepa-latest.pth.tar")
                and value["epoch"] == config["optimization"]["epochs"]):
            interrupted = True
            raise RuntimeError("simulated final save interruption")
        original_save(value, path)

    monkeypatch.setattr(train, "_atomic_torch_save", fail_before_final_latest)
    with pytest.raises(RuntimeError, match="final save interruption"):
        train.main(config)
    monkeypatch.setattr(train, "_atomic_torch_save", original_save)
    best = torch.load(tmp_path / "run" / "jepa-best.pth.tar", weights_only=True)
    assert best["epoch"] == 2
    resumed_config = copy.deepcopy(config)
    resumed_config["meta"]["load_checkpoint"] = True
    train.main(resumed_config)
    latest = torch.load(tmp_path / "run" / "jepa-latest.pth.tar", weights_only=True)
    assert latest["epoch"] == 2
    assert latest["best_val_loss"] == best["val_loss"]
    assert latest["history"] == best["history"]
    assert latest["encoder"]["scale"] == best["encoder"]["scale"]


def test_wall_validation_reuses_the_same_masks_and_preserves_rng():
    from src.train import evaluate_wall_loss

    class Images:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.full((3, 8, 8), float(index + 1)), 0

    settings = dict(input_size=8, patch_size=2, pred_mask_scale=[0.25, 0.25],
                    enc_mask_scale=[0.5, 0.5], aspect_ratio=[1.0, 1.0],
                    nenc=1, npred=1, allow_overlap=False, min_keep=0)
    encoder, predictor = TinyEncoder(), TinyPredictor()
    target = copy.deepcopy(encoder)
    before = torch.random.get_rng_state()
    first = evaluate_wall_loss(Images(), 1, 0, False, settings,
                               encoder, predictor, target, torch.device("cpu"), seed=7)
    second = evaluate_wall_loss(Images(), 1, 0, False, settings,
                                encoder, predictor, target, torch.device("cpu"), seed=7)
    assert first == second
    assert torch.equal(torch.random.get_rng_state(), before)


def test_imagenet_mode_keeps_existing_loader_path(tmp_path, monkeypatch):
    from torch.utils.data import DataLoader, DistributedSampler

    train, config = _setup(tmp_path, monkeypatch)
    config["data"]["dataset"] = "imagenet"
    config["data"]["root_path"] = str(tmp_path)
    config["data"]["image_folder"] = "unused"
    config["meta"].pop("init_checkpoint")
    config["optimization"]["epochs"] = 1
    called = []

    class Images:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.ones(3, 4, 4), 0

    def image_loader(**kwargs):
        called.append(kwargs["root_path"])
        dataset = Images()
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                            collate_fn=kwargs["collator"])
        return dataset, loader, sampler

    monkeypatch.setattr(train, "make_imagenet1k", image_loader)
    monkeypatch.setattr(train, "make_transforms", lambda **kwargs: None)
    train.main(config)
    assert called == [str(tmp_path)]
    assert (tmp_path / "run" / "jepa-latest.pth.tar").is_file()
