"""Protect identity scoring, frozen dynamics, fixed resets and real timing."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm import plan_proprio, plan_pointmaze_proprio as planner
from src.mylonas_ijepa_wm.environment_adapters import StepResult, WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
from src.mylonas_ijepa_wm.planning import CEMResult, compute_terminal_cost


def metric_module():
    name = "src.mylonas_ijepa_wm.cost_adapter"
    assert importlib.util.find_spec(name), "cost adapter module is missing"
    return importlib.import_module(name)


def test_identity_is_exact_existing_float32_terminal_mse():
    metric = metric_module()
    torch.manual_seed(2)
    adapter = metric.ResidualTokenAdapter(dim=8, bottleneck=3)
    predicted = torch.randn(7, 3, 4, 8, dtype=torch.bfloat16)
    goal = torch.randn(4, 8, dtype=torch.bfloat16)
    expected = compute_terminal_cost(predicted, goal, "latent")
    actual = metric.adapter_goal_cost(adapter, goal)(predicted[:, -1])
    assert torch.equal(actual, expected)
    assert torch.equal(adapter(goal.float()), goal.float())


def test_training_changes_shared_metric_without_encoder_gradients():
    metric = metric_module()
    adapter = metric.ResidualTokenAdapter(dim=4, bottleneck=3)
    encoder = torch.nn.Linear(4, 4).requires_grad_(False)
    with torch.inference_mode():
        tokens = encoder(torch.randn(2, 3, 4))
        goal = encoder(torch.randn(3, 4))
    # Replay tensors must be normal tensors before backpropagation.
    tokens, goal = tokens.clone(), goal.clone()
    before = metric.token_goal_cost(adapter, tokens, goal).detach()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    metric.token_goal_cost(adapter, tokens, goal).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    optimizer.step()
    after = metric.token_goal_cost(adapter, tokens, goal)
    assert not torch.equal(before, after)
    assert torch.allclose(
        metric.token_goal_cost(adapter, tokens[:1], goal),
        metric.token_goal_cost(adapter, goal.unsqueeze(0), tokens[0]),
    )
    assert metric.token_goal_cost(adapter, goal.unsqueeze(0), goal).item() == 0


class ActionDynamics(torch.nn.Module):
    num_hist = 1

    def forward(self, latents, states, actions):
        state = states + actions[..., :2]
        return state.unsqueeze(-2), state


def test_terminal_cost_hook_changes_cem_without_adapting_model_inputs():
    common = dict(
        model=ActionDynamics(), latent_history=torch.zeros(1, 1, 2),
        state_history=torch.zeros(1, 2), action_history=torch.zeros(0, 2),
        target_latent=torch.tensor([[2., 0.]]), action_low=torch.tensor([-1., -1.]),
        action_high=torch.tensor([1., 1.]), device=torch.device("cpu"),
        planning_horizon=1, population=64, num_elites=8, num_iterations=4,
        candidate_batch_size=16, cem_selection="elite_mean",
    )
    raw = planner.plan_action(**common, generator=torch.Generator().manual_seed(7))
    changed = planner.plan_action(
        **common, generator=torch.Generator().manual_seed(7),
        latent_cost_fn=lambda terminal: terminal[:, 0, 0],
    )
    assert raw.action_sequence[0, 0] > 0.5
    assert changed.action_sequence[0, 0] < -0.5


def test_fixed_wall_reset_reproduces_case_and_rejects_invalid_states():
    source = next((parent / "data/dino_wm_reference/source"
                   for parent in Path(__file__).resolve().parents
                   if (parent / "data/dino_wm_reference/source/env/wall/envs/wall.py").is_file()), None)
    if source is None:
        pytest.skip("pinned DotWall source unavailable")
    wall = WallAdapter(EnvironmentConfig(env="wall", dino_wm_root=str(source)), 100)
    for seed in (3, 5):
        initial = wall.reset(seed, layout=(32, 30), start=(44, 34), goal=(50, 42))
        np.testing.assert_array_equal(initial.state, [44, 34])
        np.testing.assert_array_equal(wall.goal, [50, 42])
        assert not initial.success
    assert wall.reset(1, start=(50, 42), goal=(50, 42)).success
    for start in ((32, 42), (0, 20), (float("nan"), 30)):
        with pytest.raises(ValueError, match="position"):
            wall.reset(0, start=start, goal=(50, 42))


def test_partial_macro_capture_contains_only_executed_primitive_actions(monkeypatch):
    class SimpleWall:
        goal = np.array([3, 0], dtype=np.float32)
        layout = (32, 30)
        action_low = np.full(2, -1.)
        action_high = np.full(2, 1.)

        def step(self, action):
            self.position += action
            return StepResult(self.position.copy(), np.zeros((224, 224, 3), np.uint8), self.position[0] >= 3)

    wall = SimpleWall()
    wall.position = np.zeros(2, dtype=np.float32)
    initial = StepResult(wall.position.copy(), np.zeros((224, 224, 3), np.uint8), False)
    args = plan_proprio.parse_args(["--env", "wall", "--world-model-checkpoint", "unused", "--output", "unused"])
    args.cost = "position"
    macro = torch.tensor([[1., 0.] * 5])
    monkeypatch.setattr(planner, "plan_action", lambda **kwargs: CEMResult(macro, 0.))
    monkeypatch.setattr(plan_proprio.visual, "encode_frame", lambda *args: torch.zeros(1, 2))
    observed = []
    record = plan_proprio.run_episode(
        wall, EnvironmentConfig(env="wall"), object(), SimpleNamespace(num_hist=1),
        {"state_mean": torch.zeros(2), "state_std": torch.ones(2)}, args,
        torch.device("cpu"), "fp32", 3, None, initial=initial,
        observation_observer=lambda step, outcome, action: observed.append((step, outcome.state.copy(), action)),
    )
    assert record["micro_steps"] == 3
    assert record["macro_durations"] == [3]
    assert record["primitive_actions"] == [[1., 0.]] * 3
    assert record["mpc_updates"] == 1
    assert [item[0] for item in observed] == [0, 1, 2, 3]
    assert observed[0][2] is None
    np.testing.assert_array_equal(observed[-1][1], [3, 0])
