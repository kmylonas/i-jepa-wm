"""The diagnostic must distinguish a bad target copy from batch encoding effects."""

import importlib
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_curriculum_multiband import curriculum_sources, prepared, TinyRuntime
from test_curriculum_clean_pair_ablation import file_hashes


def module():
    name = 'src.mylonas_ijepa_wm.curriculum_goal_audit'
    assert importlib.util.find_spec(name), 'goal consistency diagnostic is missing'
    return importlib.import_module(name)


def test_saved_audit_locates_exact_goal_and_preserves_existing_artifacts(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    curriculum.grid(TinyRuntime(curriculum, 'raw'))
    before = [file_hashes(path) for path in (*curriculum_sources, curriculum.root)]
    saved = module().load_saved(curriculum.root, curriculum_sources[1])
    checks = saved['report']
    assert checks['goal_position'] == [50., 42.]
    assert checks['goal_grid_index'] == 141
    assert checks['goal_copy']['mse'] == 0
    assert checks['grid_target_copy']['mse'] == 0
    assert checks['goal_self']['mse'] == 0
    # Synthetic grid at y=42 yields tokens [43,44,45,46], against a zero target.
    assert checks['grid_goal_vs_cached_goal']['mse'] == 1981.5
    assert checks['stored_raw_goal_cost'] == 1981.5
    assert [file_hashes(path) for path in (*curriculum_sources, curriculum.root)] == before


def test_saved_audit_rejects_changed_source_goal(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    curriculum.grid(TinyRuntime(curriculum, 'raw'))
    path = curriculum_sources[1] / 'run_0/goal_tokens.pt'
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='source goal.*hash'):
        module().load_saved(curriculum.root, curriculum_sources[1])


class RenderEnvironment:
    def __init__(self, goal_offset=0):
        self.goal_offset = goal_offset

    def reset(self, seed, *, layout, start, goal):
        self.goal = np.asarray(goal)
        return SimpleNamespace(frame=self._render_at(start))

    def _render_at(self, position):
        return np.full((224, 224, 3), int(sum(position)), dtype=np.uint8)

    def render_goal(self):
        return self._render_at(self.goal) + self.goal_offset

    def step(self, action):
        raise AssertionError('diagnostics must never execute an action')


def single(frame):
    return torch.full((1, 4), float(np.asarray(frame).mean()))


def batch(frames):
    # Model a batch-dependent encoder: exact same pixels have a +2 latent offset.
    return torch.stack([single(frame) + 2 for frame in frames])


def test_probe_exposes_batch_offset_with_identical_pixels_without_repairing_cache():
    mod = module()
    saved = dict(goal=torch.full((1, 4), 92.), grid_goal=torch.full((1, 4), 94.),
                 grid_batch_points=[[x, 42.] for x in range(40, 55, 2)], batch_goal_offset=5)
    report, payload = mod.probe(saved, RenderEnvironment(), single, batch)
    assert all(row['exact'] for row in report['pixel_comparisons'].values())
    pairs = report['token_comparisons']
    assert pairs['fresh_single_vs_cached_goal']['mse'] == 0
    assert pairs['fresh_grid_batch_vs_saved_grid_goal']['mse'] == 0
    assert pairs['fresh_single_vs_fresh_repeat']['mse'] == 0
    assert pairs['single_vs_repeated_image_batch8']['mse'] == 4
    assert pairs['single_vs_grid_batch8']['mse'] == 4
    assert torch.equal(saved['goal'], torch.full((1, 4), 92.))
    assert torch.equal(payload['tokens']['fresh_single'], torch.full((1, 4), 92.))


def test_probe_separates_render_route_mismatch_from_batch_offset():
    saved = dict(goal=torch.full((1, 4), 92.), grid_goal=torch.full((1, 4), 92.),
                 grid_batch_points=[[x, 42.] for x in range(40, 55, 2)], batch_goal_offset=5)
    report, _ = module().probe(saved, RenderEnvironment(goal_offset=1), single,
                              lambda frames: torch.stack([single(frame) for frame in frames]))
    assert not report['pixel_comparisons']['render_goal_vs_grid_route']['exact']
    assert report['pixel_comparisons']['render_goal_vs_grid_route']['max_abs'] == 1
    assert report['token_comparisons']['single_vs_repeated_image_batch8']['mse'] == 0
    assert report['token_comparisons']['single_vs_grid_batch8']['mse'] == 1


def test_probe_detects_difference_in_middle_of_identical_image_batch():
    saved = dict(goal=torch.full((1, 4), 92.), grid_goal=torch.full((1, 4), 94.),
                 grid_batch_points=[[x, 42.] for x in range(40, 55, 2)], batch_goal_offset=5)
    def position_dependent_batch(frames):
        values = torch.stack([single(frame) for frame in frames])
        values[5] += 2
        return values
    report, _ = module().probe(saved, RenderEnvironment(), single, position_dependent_batch)
    pairs = report['token_comparisons']
    assert pairs['single_vs_repeated_image_batch8']['mse'] == 4
    assert pairs['within_identical_image_batch8']['max_abs'] == 2
    assert pairs['within_identical_image_batch8']['mse'] == .5
    assert pairs['identical_vs_mixed_batch_goal_slot']['mse'] == 0


@pytest.mark.parametrize('precision', ['highest', 'high', 'medium'])
@pytest.mark.parametrize('exception_exit', [False, True])
def test_strict_probe_restores_arithmetic_without_poisoning_backend_state(precision, exception_exit):
    # A subprocess prevents a failed restoration from affecting unrelated tests.
    result = subprocess.run([sys.executable, '-c', f'''
import torch
from src.mylonas_ijepa_wm.curriculum_goal_audit import arithmetic_settings, strict_fp32
torch.set_float32_matmul_precision({precision!r})
torch.backends.cudnn.allow_tf32 = False
before = arithmetic_settings()
try:
    with strict_fp32():
        assert arithmetic_settings() == dict(matmul_precision='highest', cuda_matmul_tf32=False, cudnn_tf32=False)
        if {exception_exit!r}:
            raise LookupError('probe failed')
except LookupError:
    pass
assert arithmetic_settings() == before
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
