import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest


def _dino_source():
    configured = os.environ.get("IJEPA_DINO_WM_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(parent / "data/dino_wm_reference/source" for parent in Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "env/wall/envs/wall.py").is_file():
            return candidate
    pytest.skip("Pinned DINO-WM Wall source is unavailable")


@pytest.mark.parametrize("layout", [(26, 22), (38, 38)])
def test_probe_cases_match_real_dotwall_geometry(layout):
    from src.mylonas_ijepa_wm.evaluate_wall_collision_probes import build_probes

    config = EnvironmentConfig(env="wall", dino_wm_root=str(_dino_source()))
    adapter = WallAdapter(config, max_micro_steps=5)
    probes = build_probes(layout)
    assert {(probe["kind"], probe["direction"]) for probe in probes} == {
        (kind, direction)
        for kind in ("solid_wall", "doorway", "old_wall_open")
        for direction in ("left_to_right", "right_to_left")
    }
    for probe in probes:
        adapter.reset(17, layout=layout)
        adapter.env.set_init_state(np.asarray(probe["start"], dtype=np.float32))
        adapter.env.reset()
        start_x = adapter.state()[0]
        for action in probe["actions"]:
            end = adapter.step(action).state
        reference_x = probe["reference_x"]
        crossed = (start_x < reference_x < end[0]) or (end[0] < reference_x < start_x)
        assert crossed == (probe["kind"] != "solid_wall"), probe


def test_probe_prediction_denormalizes_state_and_records_crossing(monkeypatch):
    from src.mylonas_ijepa_wm import evaluate_wall_collision_probes as probes

    class FakeModel(torch.nn.Module):
        num_hist = 1

        def forward(self, latents, states, actions):
            predicted = states.clone()
            predicted[:, -1] = torch.tensor([0.5, 0.0])
            return latents, predicted

    class FakeEnv:
        def __init__(self, adapter):
            self.adapter = adapter

        def set_init_state(self, position):
            self.adapter.position = np.asarray(position, dtype=np.float32).copy()

        def reset(self):
            return None, torch.tensor(self.adapter.position)

    class FakeAdapter:
        def __init__(self):
            self.position = np.zeros(2, dtype=np.float32)
            self.env = FakeEnv(self)

        def reset(self, seed, layout=None):
            self.layout = tuple(layout)

        def state(self):
            return self.position.copy()

        def render(self):
            return np.zeros((224, 224, 3), dtype=np.uint8)

        def step(self, action):
            self.position += np.asarray(action, dtype=np.float32) * 2
            return type("Outcome", (), {"state": self.state(), "frame": self.render()})()

    monkeypatch.setattr(probes.visual, "encode_frame", lambda *args: torch.zeros(1, 1))
    probe = dict(
        kind="doorway", direction="left_to_right", reference_x=11,
        start=[10., 20.], actions=[[1., 0.]] * 5,
    )
    metadata = dict(
        state_mean=torch.tensor([10., 20.]), state_std=torch.tensor([4., 2.]),
        normalize_actions=False, action_mean=None, action_std=None,
    )
    record = probes.evaluate_single_probe(
        FakeAdapter(), object(), FakeModel(), metadata, probe, (32, 30),
        seed=7, device=torch.device("cpu"), precision="fp32",
    )
    assert record["predicted_final_position"] == pytest.approx([12., 20.])
    assert record["actual_final_position"] == pytest.approx([20., 20.])
    assert record["predicted_crossed"]
    assert record["actual_crossed"]
    assert record["token_mse"] == 0.0


def test_probes_reject_checkpoint_from_other_manifest_before_encoding(tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm import evaluate_wall_collision_probes as probes

    config_path = tmp_path / "environment.json"
    config_path.write_text('{"env": "wall"}')
    monkeypatch.setattr(
        probes.wm_proprio, "load_model_and_metadata",
        lambda *args: (type("Model", (), {"num_hist": 1})(),
                       {"environment": "wall", "experiment_manifest_sha256": "other"}, 0),
    )
    monkeypatch.setattr(
        probes.visual, "load_frozen_ijepa_encoder",
        lambda *args: pytest.fail("encoder should not load for mismatched manifest"),
    )
    with pytest.raises(ValueError, match="manifest"):
        probes.evaluate_probes(
            tmp_path / "checkpoint.pt", tmp_path / "encoder.pt",
            prepare_manifest(train_episodes=90, val_episodes=20, test_episodes=40),
            [10], tmp_path / "probes.json", config_path, device_name="cpu",
        )
