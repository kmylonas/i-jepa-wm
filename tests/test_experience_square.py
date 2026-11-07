"""Full-map collection samples new square strips and ranks actual detours."""
import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_square as square
from src.mylonas_ijepa_wm import geodesic_radial as radial


def test_square_manifest_is_seeded_and_covers_both_rooms_and_directions():
    manifest = square.make_manifest()
    assert manifest == square.make_manifest()
    assert len(manifest["stages"]) == 8 and all(manifest["stages"])
    cases = [case for stage in manifest["stages"] for case in stage]
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {"left_to_right", "right_to_left"} <= {case["route"] for case in cases}
    for case in cases:
        start, goal = np.asarray(case["start"]), np.asarray(case["goal"])
        lo, hi = manifest["bands"][case["band"] - 1]
        assert lo < np.abs(start - goal).max() <= hi
        assert np.linalg.norm(start - goal) >= 4.5
        assert radial.valid(start) and radial.valid(goal)


def test_filter_keeps_legal_detour_and_drops_still_and_reverse_steps():
    points = [[23, 12], [23, 16], [23, 16], [23, 12]]
    raw = dict(states=torch.tensor(points), primitive_steps=torch.arange(len(points)),
               frames=torch.arange(len(points))[:, None])
    design = dict(points=[[48, 12]], goal_ids=[], heldout_mask=[False], frame_sha256=["goal"])
    assert np.linalg.norm(np.subtract(points[1], [48, 12])) > 25
    selected, stats = square.filter_steps(raw, dict(goal=[48, 12], goal_id=0, route="left_to_right"), design, 8)
    assert selected == [0]
    assert stats["stationary_steps"] == 1 and stats["away_steps"] == 1
    assert type(stats["retained_doorway_pairs"]) is int


def test_crossing_requires_reaching_the_other_room_through_the_opening():
    assert square.crossings([[24, 30], [31, 30], [33, 30], [39, 30]]) == ["left_to_right"]
    assert square.crossings([[40, 30], [33, 30], [31, 30], [23, 30]]) == ["right_to_left"]
    assert square.crossings([[24, 30], [33, 30], [28, 30]]) == []
    assert square.crossings([[24, 15], [40, 15]]) == []
