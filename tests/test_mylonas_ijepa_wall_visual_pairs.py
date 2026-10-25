import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
from src.mylonas_ijepa_wm.wall_layout_manifest import TEST_LAYOUTS


def _dino_source():
    configured = os.environ.get("IJEPA_DINO_WM_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend(parent / "data/dino_wm_reference/source" for parent in Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / "env/wall/envs/wall.py").is_file():
            return candidate
    pytest.skip("Pinned DINO-WM Wall source is unavailable")


def _crossed(start_x, end_x, reference_x):
    return (start_x < reference_x < end_x) or (end_x < reference_x < start_x)


def test_matched_visual_pairs_have_opposite_real_crossing_outcomes():
    from src.mylonas_ijepa_wm.evaluate_wall_visual_pairs import build_matched_pairs

    adapter = WallAdapter(EnvironmentConfig(env="wall", dino_wm_root=str(_dino_source())), 5)
    pairs = build_matched_pairs()
    assert len(pairs) == 4
    for pair in pairs:
        assert len(pair["layouts"]) == 2
        assert all(layout in TEST_LAYOUTS for layout in pair["layouts"])
        assert len(pair["actions"]) == 5
        actual = []
        for layout in pair["layouts"]:
            adapter.reset(10000, layout=layout)
            adapter.env.set_init_state(np.asarray(pair["start"], dtype=np.float32))
            adapter.env.reset()
            start = adapter.state().copy()
            assert start.tolist() == pair["start"]
            for action in pair["actions"]:
                end = adapter.step(action).state
            actual.append(_crossed(start[0], end[0], pair["reference_x"]))
        assert actual == pair["expected_crossed"], pair["name"]
        assert actual == [True, False] or actual == [False, True]


def test_model_sees_same_state_and_actions_but_different_visual_tokens(monkeypatch):
    from src.mylonas_ijepa_wm import evaluate_wall_visual_pairs as visual_pairs

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, latents, states, actions):
            self.inputs.append((latents.clone(), states.clone(), actions.clone()))
            return latents, states

    adapter = WallAdapter(EnvironmentConfig(env="wall", dino_wm_root=str(_dino_source())), 5)
    model = RecordingModel()
    monkeypatch.setattr(
        visual_pairs.visual, "encode_frame",
        lambda encoder, frame, device, precision: torch.from_numpy(
            np.asarray(frame, dtype=np.float32).copy()
        ).flatten(),
    )
    pair = visual_pairs.build_matched_pairs()[0]
    record = visual_pairs.evaluate_pair(
        adapter, object(), model,
        {"state_mean": torch.zeros(2), "state_std": torch.ones(2), "normalize_actions": False},
        pair, 10000, torch.device("cpu"), "fp32",
    )
    assert len(model.inputs) == 2
    assert torch.equal(model.inputs[0][1], model.inputs[1][1])
    assert torch.equal(model.inputs[0][2], model.inputs[1][2])
    assert not torch.equal(model.inputs[0][0], model.inputs[1][0])
    assert [case["actual_crossed"] for case in record["cases"]] == [True, False]
    assert all(len(case["actual_positions"]) == 6 for case in record["cases"])
