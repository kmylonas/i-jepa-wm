"""A shared real-image grid, comparable costs, and full sequential diagnostics."""

import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_curriculum_multiband import curriculum_sources, TinyRuntime, prepared
from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter


def heatmaps():
    name = "src.mylonas_ijepa_wm.curriculum_heatmaps"
    assert importlib.util.find_spec(name), "curriculum heatmap module missing"
    return importlib.import_module(name)


class GridRuntime:
    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = 0
        self.env = SimpleNamespace(env=SimpleNamespace(border_padding=7.4, img_size=65),
            reset=lambda *a, **kw: None,
            _render_at=lambda position: np.full((2, 2, 3), int(position[1]), np.uint8))

    def encode_frames(self, frames):
        self.calls += 1
        values = torch.tensor([float(np.asarray(frame).mean()) for frame in frames])
        return values[:, None, None].expand(-1, 1, 4).clone()


def test_grid_is_cached_float32_fixed_points_and_rejects_changed_bytes(tmp_path):
    mod = heatmaps()
    runtime = GridRuntime()
    signature = {"experiment": "test", "precision": "fp32", "goal_sha256": "goal"}
    grid = mod.prepare_grid(runtime, tmp_path, torch.zeros(1, 4), signature)
    assert grid["tokens"].shape == (176, 1, 4) and grid["tokens"].dtype == torch.float32
    assert grid["points"][0].tolist() == [40., 8.]
    assert grid["points"][-1].tolist() == [54., 50.]
    assert runtime.calls == 1
    second = mod.prepare_grid(runtime, tmp_path, torch.zeros(1, 4), signature)
    assert runtime.calls == 1 and torch.equal(second["raw_costs"], grid["raw_costs"])
    with pytest.raises(ValueError, match="provenance|signature"):
        mod.load_grid(tmp_path, dict(signature, precision="bf16"))
    path = tmp_path / "cache/grid.pt"
    path.write_bytes(path.read_bytes()+b"modified")
    with pytest.raises(ValueError, match="hash"):
        mod.load_grid(tmp_path, signature)


def test_heatmap_uses_shared_raw_scale_and_saves_unclipped_numeric_values(tmp_path):
    mod = heatmaps()
    signature = {"experiment": "test"}
    grid = mod.prepare_grid(GridRuntime(), tmp_path, torch.zeros(1, 4), signature)
    adapter = ResidualTokenAdapter(4, 3).eval().requires_grad_(False)
    first = mod.checkpoint_heatmap(tmp_path, "original", 0, adapter, np.array([[44., 32.]]),
                                  signature, checkpoint_sha256="identity")
    assert first["raw"] == first["adapted"]
    assert all(value == 0 for value in first["difference"])
    # Shared identity skip can be changed into a scaling map without changing raw targets.
    class DoubleTokens(torch.nn.Module):
        def forward(self, tokens):
            return 2*tokens
    second = mod.checkpoint_heatmap(tmp_path, "original", 1, DoubleTokens(), np.array([[44., 24.]]),
                                   signature, checkpoint_sha256="double")
    assert second["raw"] == first["raw"]
    assert second["display_max"] == first["display_max"] == 2500.
    assert np.allclose(second["adapted"], 4*np.array(first["raw"]))
    assert second["adapted_max"] == 10000. and second["saturated_adapted_points"] > 0
    assert second["difference_limit"] == 2500.
    assert second["grid_sha256"] == first["grid_sha256"]
    assert (tmp_path / "branches/original/heatmaps/v1.png").is_file()


def test_two_full_curricula_generate_maps_and_account_for_840_episodes(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    raw_runtime = TinyRuntime(curriculum, "raw")
    curriculum.grid(raw_runtime)
    grid_encodes = raw_runtime.encode_calls
    curriculum.grid(raw_runtime)
    assert raw_runtime.encode_calls == grid_encodes
    curriculum.baseline(raw_runtime)
    for recipe in ("original", "clean"):
        runtime = TinyRuntime(curriculum, recipe)
        curriculum.initial(recipe, runtime)
        for index in (1, 2, 3):
            assert curriculum.band(recipe, index, runtime)["status"] == "complete"
        fit = json.loads((runtime.root / "updates/v3/training.json").read_text())
        assert fit["band_episode_counts"] == {"near": 3, "band_1": 40, "band_2": 40, "band_3": 40}
        assert all(case_id.startswith("train_") for case_id in fit["training_case_ids"])
        initial = json.loads((runtime.root / "heatmaps/v0.json").read_text())
        final = json.loads((runtime.root / "heatmaps/v3.json").read_text())
        assert initial["raw"] == final["raw"] and initial["display_max"] == final["display_max"]
        assert initial["checkpoint_sha256"] != final["checkpoint_sha256"]
    report = curriculum.report(render=True)
    assert report["total_primitive_interactions"] == 3360
    assert report["diagnostic_primitive_interactions"] == 0
    assert all(row["status"] == "complete" for row in report["branches"].values())
    assert (curriculum.root / "curriculum_matrix.png").is_file()
    assert report["branches"]["clean"]["per_case"]["3"]["band_3"][0]["case"] == curriculum.manifest["evaluation"]["band_3"][0]
    assert (curriculum.root / "branches/clean/updates/v3/evaluation_paths.png").is_file()


def test_partial_report_has_unavailable_cells_and_renders(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    report = curriculum.report(render=True)
    assert all(value is None for value in report["raw"].values())
    assert report["branches"]["original"]["versions"]["3"]["band_3"] is None
    assert (curriculum.root / "curriculum_matrix.png").is_file()
