import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm import plan_proprio
from src.mylonas_ijepa_wm.environment_adapters import StepResult
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
from src.mylonas_ijepa_wm.wall_layout_manifest import prepare_manifest


def test_matrix_builds_matched_modes_with_requested_action_caps(tmp_path):
    from src.mylonas_ijepa_wm.run_wall_planning_matrix import build_invocations

    manifest = prepare_manifest(train_episodes=90, val_episodes=20, test_episodes=40)
    configs = {tuple(layout): tmp_path / f"{layout[0]}_{layout[1]}.json"
               for layout in manifest["test_layouts"]}
    invocations = build_invocations(
        manifest, configs, tmp_path, Path("encoder.pt"),
        Path("multi.pt"), Path("fixed.pt"), base_seed=10000, num_episodes=20,
    )
    assert len(invocations) == 24
    for layout in manifest["test_layouts"]:
        matched = [item for item in invocations if item["layout"] == layout]
        assert {(item["model"], item["mode"]) for item in matched} == {
            (model, mode) for model in ("multi", "fixed")
            for mode in ("unbounded", "cap4", "cap_train")
        }
        for item in matched:
            args = item["argv"]
            assert args[args.index("--base-seed") + 1] == "10000"
            assert args[args.index("--num-episodes") + 1] == "20"
            assert args[args.index("--planning-horizon") + 1] == "5"
            assert args[args.index("--actions-per-plan") + 1] == "5"
            assert "--unbounded-actions" in args
            if item["mode"] == "unbounded":
                assert "--max-action-length" not in args
            else:
                expected = "4.0" if item["mode"] == "cap4" else "3.5"
                assert args[args.index("--max-action-length") + 1] == expected


def test_wilson_interval_and_matrix_summary_use_each_layout_separately():
    from src.mylonas_ijepa_wm.run_wall_planning_matrix import summarize_run, wilson_interval

    low, high = wilson_interval(0, 20)
    assert low == pytest.approx(0.0)
    assert 0.15 < high < 0.17
    payload = {"episodes": [
        {"seed": 10, "wall_layout": [26, 22], "success": True,
         "final_goal_distance": 1.0, "failed_by_step_limit": False},
        {"seed": 11, "wall_layout": [26, 22], "success": False,
         "final_goal_distance": 9.0, "failed_by_step_limit": True},
    ]}
    summary = summarize_run(payload, [26, 22], [10, 11])
    assert summary["successes"] == 1
    assert summary["episodes"] == 2
    assert summary["mean_final_distance"] == pytest.approx(5.0)
    assert summary["failure_counts"]["step_limit"] == 1
    with pytest.raises(ValueError, match="layout"):
        summarize_run(payload, [38, 38], [10, 11])


def test_planning_record_contains_actual_wall_layout(monkeypatch):
    class FakeModel(torch.nn.Module):
        num_hist = 1

        def forward(self, latents, states, actions):
            return latents, states

    class FakeAdapter:
        goal = np.array([0., 0.], dtype=np.float32)
        action_low = np.array([-1., -1.], dtype=np.float32)
        action_high = np.array([1., 1.], dtype=np.float32)
        layout = (26, 22)

        def reset(self, seed):
            self.position = np.array([1., 0.], dtype=np.float32)
            return StepResult(self.position.copy(), np.zeros((224, 224, 3), dtype=np.uint8), False)

        def step(self, action):
            return StepResult(self.position.copy(), np.zeros((224, 224, 3), dtype=np.uint8), False)

    class Plan:
        action_sequence = torch.zeros(1, 10)
        cost = 0.0

    monkeypatch.setattr(plan_proprio.visual, "encode_frame", lambda *args: torch.zeros(1, 1))
    monkeypatch.setattr(plan_proprio.planner, "plan_action", lambda **kwargs: Plan())
    args = argparse.Namespace(
        cost="position", max_macro_steps=1, planning_horizon=1,
        actions_per_plan=1, population=1, num_elites=1,
        cem_iterations=1, candidate_batch_size=1, min_std=0.05,
        unbounded_actions=True, max_action_length=3.5,
    )
    metadata = dict(
        state_mean=torch.zeros(2), state_std=torch.ones(2),
        action_mean=None, action_std=None,
    )
    record = plan_proprio.run_episode(
        FakeAdapter(), EnvironmentConfig(env="wall"), object(), FakeModel(),
        metadata, args, torch.device("cpu"), "fp32", 10, None,
    )
    assert record["wall_layout"] == [26, 22]
