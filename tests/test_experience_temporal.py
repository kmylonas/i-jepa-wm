"""Temporal ranking labels depend on rollout success and frame order."""
import torch

from src.mylonas_ijepa_wm import experience_curriculum as experience
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash


def test_temporal_labels_keep_reversals_in_success_and_drop_all_failures():
    frames = torch.tensor([[0], [1], [1], [2], [3]])
    raw = dict(states=torch.tensor([[50, 36], [50, 35], [50, 35], [50, 38], [50, 42]]),
               frames=frames, primitive_steps=torch.arange(5), success=True)
    case = dict(goal=[50, 42], goal_id=0)
    design = dict(points=[[50, 42]], goal_ids=[0], heldout_mask=[False],
                  frame_sha256=[_frame_hash(frames[-1])])

    progress, _ = experience._filter(raw, case, design, 8)
    assert progress == [2]
    temporal, stats = experience._temporal_filter(raw, case, design, 8)
    assert temporal == [0, 2]
    assert stats["retained_pairs"] == 2
    assert stats["stationary_steps"] == 1

    temporal, stats = experience._temporal_filter(dict(raw, success=False), case, design, 8)
    assert temporal == []
    assert stats["eligible_pairs"] == stats["retained_pairs"] == 0
    assert stats["skipped_failed"]
