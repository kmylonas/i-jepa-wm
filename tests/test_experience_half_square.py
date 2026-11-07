"""Half-map temporal collection expands over the full right room."""
import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_half_square as half
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash


def test_square_shells_are_tight_and_cover_reset_domain_for_every_goal():
    points = [[50, 42], [44, 40], [53, 48], [44, 20], [50, 12], [55, 14]]
    design = dict(points=points, train_ids=list(range(len(points))))
    manifest = half.make_manifest(design)
    assert manifest == half.make_manifest(design)
    assert len(manifest["bands"]) > 8 and all(manifest["stages"])
    assert manifest["bands"][0] == [0, 6]
    assert all(hi - lo <= 6 for lo, hi in manifest["bands"])
    final = manifest["bands"][-1][-1]
    assert all(max(abs(x - g[0]), abs(y - g[1])) <= final
               for g in points for x in (35.01, 56.5) for y in (7.5, 56.5))
    for (lo, hi), cases in zip(manifest["bands"], manifest["stages"]):
        for case in cases:
            start, goal = np.asarray(case["start"]), np.asarray(case["goal"])
            assert lo < np.abs(start - goal).max() <= hi
            assert 35.01 <= start[0] <= 56.5 and 7.5 <= start[1] <= 56.5
            assert np.linalg.norm(start - goal) >= 4.5


def test_full_right_room_filter_keeps_far_reversals_only_in_successes():
    frames = torch.arange(5)[:, None]
    raw = dict(states=torch.tensor([[50, 55], [50, 56], [50, 54], [25, 30], [50, 12]]),
               frames=frames, primitive_steps=torch.arange(5), success=True)
    case = dict(goal=[50, 12], goal_id=0)
    design = dict(points=[[50, 12]], goal_ids=[0], heldout_mask=[False],
                  frame_sha256=[_frame_hash(frames[-1])])
    assert half.filter_steps(raw, case, design, 8)[0] == [0, 1]
    assert half.filter_steps(dict(raw, success=False), case, design, 8)[0] == []
