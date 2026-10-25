import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
from src.mylonas_ijepa_wm.wall_layout_manifest import TEST_LAYOUTS, prepare_manifest


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


def test_rejects_pair_whose_real_outcomes_do_not_contrast(monkeypatch):
    from src.mylonas_ijepa_wm import evaluate_wall_visual_pairs as visual_pairs

    class StationaryModel(torch.nn.Module):
        def forward(self, latents, states, actions):
            return latents, states

    monkeypatch.setattr(visual_pairs.visual, "encode_frame", lambda *args: torch.zeros(1, 1))
    adapter = WallAdapter(EnvironmentConfig(env="wall", dino_wm_root=str(_dino_source())), 5)
    pair = visual_pairs.build_matched_pairs()[0]
    pair["layouts"] = [[26, 22], [26, 22]]
    with pytest.raises(ValueError, match="contrast"):
        visual_pairs.evaluate_pair(
            adapter, object(), StationaryModel(),
            {"state_mean": torch.zeros(2), "state_std": torch.ones(2),
             "normalize_actions": False},
            pair, 10000, torch.device("cpu"), "fp32",
        )


def test_rejects_other_manifest_before_loading_encoder(tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm import evaluate_wall_visual_pairs as visual_pairs

    config = tmp_path / "environment.json"
    config.write_text('{"env": "wall"}')
    monkeypatch.setattr(
        visual_pairs.wm_proprio, "load_model_and_metadata",
        lambda *args: (type("Model", (), {"num_hist": 1})(),
                       {"environment": "wall", "experiment_manifest_sha256": "other"}, 0),
    )
    monkeypatch.setattr(
        visual_pairs.visual, "load_frozen_ijepa_encoder",
        lambda *args: pytest.fail("encoder loaded before manifest validation"),
    )
    with pytest.raises(ValueError, match="manifest"):
        visual_pairs.evaluate_pairs(
            tmp_path / "checkpoint.pt", tmp_path / "encoder.pt",
            prepare_manifest(train_episodes=90, val_episodes=20, test_episodes=40),
            tmp_path / "result.json", config, "cpu", "fp32",
        )


def test_pair_figure_shows_actual_paths_predicted_endpoints_and_wall_guides(tmp_path):
    import matplotlib.pyplot as plt
    from src.mylonas_ijepa_wm.evaluate_wall_visual_pairs import (
        build_pair_figure, save_pair_figure,
    )

    record = {
        "name": "door_y22_left_to_right", "reference_x": 26,
        "start": [21.0, 22.0], "actions": [[1.0, 0.0]] * 5,
        "cases": [
            {"layout": [26, 22], "actual_positions": [[21, 22], [22, 22],
              [23, 22], [24, 22], [25, 22], [27, 22]],
             "predicted_final_position": [26.5, 22.2],
             "actual_crossed": True, "predicted_crossed": True},
            {"layout": [26, 38], "actual_positions": [[21, 22], [22, 22],
              [22.8, 22], [22.8, 22], [22.8, 22], [22.8, 22]],
             "predicted_final_position": [22.5, 22.3],
             "actual_crossed": False, "predicted_crossed": False},
        ],
    }
    fig = build_pair_figure(record)
    assert len(fig.axes) == 2
    for ax, case in zip(fig.axes, record["cases"]):
        assert "Crossed x=26" in ax.get_title()
        reference = [line for line in ax.lines if line.get_label() == "Crossing plane x=26"]
        assert len(reference) == 1
        assert reference[0].get_xdata()[0] == 26
        paths = [line for line in ax.lines if line.get_label() == "Actual trajectory"]
        assert len(paths) == 1
        assert list(paths[0].get_xdata()) == [p[0] for p in case["actual_positions"]]
        predicted = [item for item in ax.collections if item.get_label() == "Predicted endpoint"]
        assert len(predicted) == 1
        assert predicted[0].get_offsets()[0].tolist() == case["predicted_final_position"]
        guides = [line for line in ax.lines if line.get_label() == "Training wall x"]
        assert [line.get_xdata()[0] for line in guides] == [24, 32, 40]
        assert all(line.get_alpha() < 0.5 for line in guides)
        assert len(ax.patches) == 2
    plt.close(fig)
    record["cases"][1]["predicted_final_position"] = [65.0, 60.0]
    record["cases"][1]["predicted_crossed"] = True
    wide_fig = build_pair_figure(record)
    assert all(ax.get_xlim()[1] > 65 and ax.get_ylim()[0] > 60 for ax in wide_fig.axes)
    plt.close(wide_fig)
    path = tmp_path / "nested" / "comparison.png"
    save_pair_figure(record, path)
    assert path.is_file() and path.stat().st_size > 10_000
