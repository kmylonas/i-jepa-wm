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


def test_expanded_goal_manifest_uses_all_six_right_room_goals():
    right = [[50, 42], [44.2, 40.45], [53, 47.2], [44.3, 20.75], [50, 12], [50, 16]]
    archive = dict(goal_ids=list(range(6)), stages=[[dict(goal_index=i, goal=g)
                   for i, g in enumerate(right)]])
    goals = square.expanded_goals(archive)
    assert goals[:3] == square.make_manifest()["goals"][:3]
    assert goals[3:] == right
    manifest = square.make_repeated_manifest(6, rounds=3, goals=goals)
    assert manifest["goals"] == goals
    assert all(case["goal"] == goals[case["goal_index"]]
               for batch in manifest["stages"] for case in batch)
    assert len(manifest["stages"][0]) == 54
    assert {c["route"] for batch in manifest["stages"] for c in batch} >= {"left_to_right", "right_to_left"}


def test_fixed_benchmark_uses_archived_starts_seeds_and_goal_tokens(tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm.curriculum_protocol import write_json
    from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, save_tensors

    old, output = tmp_path / "old", tmp_path / "output"
    manifest = square.make_repeated_manifest(1, rounds=2)
    cases = manifest["stages"][6]
    write_json(old / "experiment.json", dict(manifest=manifest))
    target = old / "collection_goals.pt"
    save_tensors(target, dict(runtime_id="archive", tokens=torch.arange(6.).reshape(6, 1, 1)))
    write_json(old / "band_4_round_1" / "complete.json", dict(runtime_id="archive",
        goal_sha256=file_sha256(target), collection=[
            dict(case=c, success=False, reached_other_room=False) for c in cases]))
    monkeypatch.setattr(square.experience, "_checkpoint", lambda _, index, rounds:
                        (None, {"checkpoint_sha256": None}) if (index, rounds) == (5, 2) else None)
    seen = []

    class Runtime:
        root = output
        runtime_id = "new"
        device = "cpu"

        def run_case(self, case, *, latent_cost_fn, journal_stage, record_positions):
            assert record_positions and journal_stage == "benchmark_band_4" and case["split"] == "eval"
            assert latent_cost_fn(torch.zeros(1, 1, 1)).item() == case["goal_index"] ** 2
            seen.append(case)
            return dict(success=True, primitive_positions=[case["start"], case["goal"]], micro_steps=1)

    rows = square.benchmark_band4(Runtime(), manifest, old)
    assert seen == [dict(c, split="eval") for c in cases]
    assert all(r["success"] and not r["previous_success"] for r in rows)
    assert square.benchmark_band4(Runtime(), manifest, old) == rows
    assert len(seen) == len(cases)


def test_extra_left_goals_append_to_the_nine_goal_run():
    right = [[50, 42], [44.2, 40.45], [53, 47.2], [44.3, 20.75], [50, 12], [50, 16]]
    archive = dict(stages=[[dict(goal_index=i, goal=g) for i, g in enumerate(right)]])
    extra = [[14, 12], [23, 30], [16, 51]]
    nine = square.expanded_goals(archive)
    goals = square.expanded_goals(archive, extra_left=extra)
    assert goals == nine + extra
    original = square.make_repeated_manifest(6, rounds=3, goals=nine)
    manifest = square.make_repeated_manifest(6, rounds=3, goals=goals, extra_goal_count=3)
    assert [[c for c in batch if c["goal_index"] < 9] for batch in manifest["stages"]] == original["stages"]
    assert sum(map(len, manifest["stages"][:12])) == 864
    assert all(sum(c["goal_index"] == 10 for c in batch) == 6 for batch in manifest["stages"][:12])


def test_failure_feedback_samples_distinct_frames_and_prefers_the_start(tmp_path):
    from types import SimpleNamespace
    from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, save_tensors
    from src.mylonas_ijepa_wm.ranking_planning import _frame_hash

    xy = np.asarray([[20, 20], [20, 20], [20, 21], [20, 22], [20, 23], [20, 23], [20, 24]])
    frames = np.asarray([[0], [0], [1], [2], [3], [3], [4]])
    case = dict(case_id="failure", goal=[20, 30], goal_index=0)
    design = dict(points=[case["goal"], [20, 22]], goal_ids=[0], heldout_mask=[False, True],
                  frame_sha256=["reference", _frame_hash(frames[3])])
    indices = square.sample_failed_frames(xy, frames, case, design, 2)
    assert indices == [0, 2, 6]
    assert square.sample_failed_frames(xy, frames, case, design, 1) == [0, 6]

    path = tmp_path / "failed.pt"
    save_tensors(path, dict(runtime_id="fixture", case=case, success=False, sampled_failure=True,
                           frames=torch.as_tensor(frames[indices]), states=torch.as_tensor(xy[indices]),
                           primitive_steps=torch.as_tensor(indices)))
    runtime = SimpleNamespace(root=tmp_path, runtime_id="fixture",
                              encode_frames=lambda batch: batch.float().unsqueeze(-1))
    record = dict(case=case, raw_path=path.name, raw_sha256=file_sha256(path), success=False,
                  micro_steps=6, primitive_positions=xy.tolist())
    manifest = dict(goal_ids=[0], pair_cap=8, failure_pair_cap=2)
    saved = square.experience._replay(runtime, record, design, manifest, filter_steps=square.temporal_filter)
    assert saved["pairs"] == [[1, 0], [2, 0]]
    assert saved["positions"].tolist() == xy[indices].tolist()
    assert saved["stats"]["steps"] == 6 and saved["stats"]["retained_pairs"] == 2


def test_failure_capture_saves_sparse_frames_and_keeps_the_full_position_trace(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.mylonas_ijepa_wm import curriculum_runtime as runtime

    def rollout(*args, observation_observer, **kwargs):
        for i in range(6):
            observation_observer(i, SimpleNamespace(state=np.array([20, 20 + i]), frame=np.array([i])),
                                 None if i == 0 else np.array([0, 1]))
        return dict(success=False, primitive_actions=[[0, 1]] * 5, macro_durations=[5], micro_steps=5)

    monkeypatch.setattr(runtime.plan_proprio, "run_episode", rollout)
    rt = SimpleNamespace(root=tmp_path, runtime_id="fixture", manifest={"planning": {}},
                         load_world_model=lambda: None, load_encoder=lambda: None,
                         env=SimpleNamespace(reset=lambda *a, **kw: None), config=None,
                         encoder=None, model=None, metadata=None, args=SimpleNamespace(run_seed=0, videos=False),
                         device="cpu", precision="fp32")
    case = dict(case_id="failure", run_seed=0, seed=1, layout=[32, 30], start=[20, 20], goal=[20, 30])
    record = runtime.PilotRuntime.run_case(rt, case, capture=True, capture_failures=False,
                                          failure_frame_selector=lambda *_: [0, 2, 5])
    saved = torch.load(tmp_path / record["raw_path"], weights_only=True)
    assert saved["sampled_failure"] and saved["primitive_steps"].tolist() == [0, 2, 5]
    assert saved["frames"].flatten().tolist() == [0, 2, 5]
    assert len(record["primitive_positions"]) == 6 and len(saved["primitive_actions"]) == 5


def test_loop_filter_drops_loop_touching_pairs_before_the_episode_cap():
    xy = [[40, 12], [42, 12], [42, 15], [42.2, 12], [44, 12], [46, 12], [48, 12]]
    raw = dict(states=torch.tensor(xy), primitive_steps=torch.arange(len(xy)),
               frames=torch.arange(len(xy))[:, None], success=True)
    design = dict(points=[[54, 12]], goal_ids=[], heldout_mask=[False], frame_sha256=["goal"])
    case = dict(goal=[54, 12], goal_id=0)
    selected, stats = square.loop_filtered_steps(raw, case, design, 2, radius=.5)
    assert selected == [3, 5]
    assert stats["erased_steps"] == [1, 2]
    assert stats["loop_removed_pairs"] == 3


def test_loop_filter_preserves_progress_and_contiguous_slow_moves():
    xy = [[24, 20], [24, 20.1], [24, 20.3], [24, 30], [32, 30], [38, 30]]
    raw = dict(states=torch.tensor(xy), primitive_steps=torch.arange(len(xy)),
               frames=torch.arange(len(xy))[:, None], success=True)
    design = dict(points=[[48, 30]], goal_ids=[], heldout_mask=[False], frame_sha256=["goal"])
    selected, stats = square.loop_filtered_steps(raw, dict(goal_id=0), design, 8, radius=.5)
    assert selected == list(range(5))
    assert stats["erased_steps"] == []
