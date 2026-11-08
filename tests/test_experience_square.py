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


def test_temporal_rounds_keep_full_map_goals_and_first_batch():
    old = square.make_manifest()
    current = square.make_repeated_manifest(6, rounds=3)
    assert current == square.make_repeated_manifest(6, rounds=3)
    assert current["goals"] == old["goals"] and current["bands"] == old["bands"]
    assert len(current["stages"]) == 24
    assert sum(map(len, current["stages"])) == 810
    for i, batch in enumerate(old["stages"]):
        first, second, third = current["stages"][i * 3:i * 3 + 3]
        assert [(c["start"], c["goal"], c["seed"]) for c in first] == [
            (c["start"], c["goal"], c["seed"]) for c in batch
        ]
        all_starts = [tuple(c["start"]) for c in first + second + third]
        assert len(all_starts) == len(set(all_starts))
    assert current["planning"]["planning_horizon"] == current["planning"]["actions_per_plan"] == 1


def test_temporal_labels_keep_both_directions_and_exclude_failed_rollouts():
    xy = [[24, 30], [31, 30], [33, 30], [39, 30], [39, 25], [37, 25]]
    raw = dict(states=torch.tensor(xy), primitive_steps=torch.arange(len(xy)),
               frames=torch.arange(len(xy))[:, None], success=True)
    design = dict(points=[[48, 30]], goal_ids=[], heldout_mask=[False], frame_sha256=["goal"])
    case = dict(goal=[48, 30], goal_id=0, route="left_to_right")
    assert square.temporal_filter(raw, case, design, 8)[0] == [0, 1, 2, 3, 4]
    assert square.temporal_filter(dict(raw, success=False), case, design, 8)[0] == []
    summary = square.summarize_round_episode(dict(case=case, primitive_positions=xy), {}, 16, 5)
    assert summary["reached_other_room"]
    assert summary["crossings"] == ["left_to_right"]
    assert summary["boundary"]["first_entry_step"] == 2
