import importlib
import importlib.util

import pytest
import torch

from src.mylonas_ijepa_wm.cost_adapter import token_goal_cost


def replay_module():
    name = "src.mylonas_ijepa_wm.curriculum_replay"
    assert importlib.util.find_spec(name), "temporal replay module missing"
    return importlib.import_module(name)


def episode(success=True, split="train"):
    # Cost is deliberately inverted: chronological success should be lower.
    tokens = torch.tensor([[[1., 0., 0., 0.]], [[0., 2., 0., 0.]], [[0., 0., 3., 0.]]])
    return dict(case_id="train_0", split=split, success=success, latents=tokens,
                states=torch.tensor([[40., 30.], [44., 34.], [48., 38.]]),
                primitive_steps=torch.tensor([0, 2, 4]))


def test_loop_pairs_use_last_visit_and_actual_primitive_time():
    replay = replay_module()
    states = torch.tensor([[40., 30.], [44., 30.], [40.1, 30.], [46., 34.], [50., 40.]])
    retained, pairs = replay.eligible_pairs(states, torch.tensor([0, 1, 3, 4, 6]))
    assert retained == [2, 3, 4]
    assert pairs == [(2, 4), (3, 4)]


def test_oracle_progress_filter_keeps_inclusive_margin_and_drops_regression():
    replay = replay_module()
    states = torch.tensor([[2., 0.], [0., 1.5], [3., 0.]])
    steps = torch.tensor([0, 2, 4])
    retained, temporal = replay.eligible_pairs(states, steps)
    clean_retained, clean = replay.eligible_pairs(
        states, steps, goal=torch.tensor([0., 0.]), min_goal_progress=.5,
    )
    assert clean_retained == retained
    assert temporal == [(0, 1), (1, 2)]
    assert clean == [(0, 1)]  # Exactly 0.5 units closer is retained.


def test_oracle_progress_filter_validates_only_when_enabled():
    replay = replay_module()
    states = torch.tensor([[0., 0.], [2., 0.]])
    steps = torch.tensor([0, 2])
    expected = replay.eligible_pairs(states, steps)
    assert replay.eligible_pairs(states, steps, goal=object(), min_goal_progress=None) == expected
    for goal, margin in ((None, .5), ([0., float("nan")], .5), ([0., 0.], 0.), ([0., 0.], float("inf"))):
        with pytest.raises(ValueError, match="goal|progress"):
            replay.eligible_pairs(states, steps, goal=goal, min_goal_progress=margin)


def test_failed_or_eval_episodes_cannot_become_training_replay():
    replay = replay_module()
    for data in ([], [episode(False)], [episode(split="eval")]):
        with pytest.raises(ValueError, match="successful|training"):
            replay.fit_adapter(data, torch.zeros(1, 4), seed=0, updates=1)
    short = episode()
    short["states"][:] = torch.tensor([40., 30.])
    with pytest.raises(ValueError, match="eligible"):
        replay.fit_adapter([short], torch.zeros(1, 4), seed=0, updates=1)


def test_temporal_fitting_reduces_inversions_and_restores_frozen_checkpoint(tmp_path):
    replay = replay_module()
    data, goal = episode(), torch.zeros(1, 4)
    adapter, result = replay.fit_adapter(
        [data], goal, seed=3, updates=100, batch_size=8, lr=0.01, bottleneck=8,
        anchor_weight=0.05,
    )
    cost = token_goal_cost(adapter, data["latents"], goal).detach()
    assert cost[0] > cost[-1]
    assert result["history"][-1]["rank_loss"] < result["history"][0]["rank_loss"]
    assert result["training_case_ids"] == ["train_0"]
    assert result["updates"] == 100
    path = tmp_path / "adapter.pt"
    replay.save_adapter(path, adapter, result, {"runtime_id": "abc"})
    restored, saved = replay.load_adapter(path, torch.device("cpu"), expected_runtime_id="abc")
    assert all(not p.requires_grad for p in restored.parameters())
    assert torch.equal(token_goal_cost(restored, data["latents"], goal), cost)
    assert "optimizer" in saved["training"]
    with pytest.raises(ValueError, match="provenance"):
        replay.load_adapter(path, torch.device("cpu"), expected_runtime_id="different")


def test_fitting_is_seeded_and_rejects_invalid_timelines():
    replay = replay_module()
    data, goal = episode(), torch.zeros(1, 4)
    first, first_result = replay.fit_adapter([data], goal, seed=9, updates=3, batch_size=2, bottleneck=3)
    second, _ = replay.fit_adapter([data], goal, seed=9, updates=3, batch_size=2, bottleneck=3)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    explicit_default, default_result = replay.fit_adapter(
        [data], goal, seed=9, updates=3, batch_size=2, bottleneck=3,
        min_goal_progress=None,
    )
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), explicit_default.parameters()))
    assert "pair_filter" not in default_result
    assert default_result["trajectories"] == first_result["trajectories"]
    assert default_result["history"] == first_result["history"]
    data["primitive_steps"] = torch.tensor([0, 4, 2])
    with pytest.raises(ValueError, match="increasing"):
        replay.fit_adapter([data], goal, seed=0, updates=1)


def test_clean_fitting_records_pairs_before_and_after_oracle_filter():
    replay = replay_module()
    data = episode()
    data["case"] = {"goal": [0., 0.]}
    data["states"] = torch.tensor([[2., 0.], [0., 1.5], [3., 0.]])
    _, result = replay.fit_adapter(
        [data], torch.zeros(1, 4), seed=1, updates=1, batch_size=1,
        min_goal_progress=.5,
    )
    assert result["pair_filter"] == {"kind": "oracle_goal_progress", "min_goal_progress": .5}
    assert result["trajectories"] == [{
        "case_id": "train_0", "frames": 3, "retained": 3,
        "temporal_pairs": 2, "eligible_pairs": 1,
    }]
