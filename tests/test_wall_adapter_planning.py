import pytest

from src.mylonas_ijepa_wm.wall_adapter_planning import ensure_identity, summarize


def _record(condition, layout, seed, success, final_distance):
    return dict(condition=condition, layout=list(layout), seed=seed,
                success=success, final_goal_distance=final_distance,
                goal_position=[50.0, 30.0], achieved_positions=[[10.0, 30.0]])


def test_summary_pairs_seeds_across_layouts():
    layouts = [(26, 22), (38, 38)]
    records = [
        _record("fixed", layouts[0], 10, False, 12),
        _record("multi", layouts[0], 10, True, 3),
        _record("fixed", layouts[1], 10, True, 2),
        _record("multi", layouts[1], 10, True, 2),
        _record("fixed", layouts[0], 11, False, 14),
        _record("multi", layouts[0], 11, False, 10),
        _record("fixed", layouts[1], 11, True, 2),
        _record("multi", layouts[1], 11, False, 8),
    ]
    result = summarize(records, layouts, [10, 11], bootstrap_samples=100)
    assert result["overall"]["fixed"]["successes"] == 2
    assert result["overall"]["multi"]["successes"] == 2
    assert result["paired"]["difference"] == 0
    assert result["paired"]["multi_only"] == 1
    assert result["paired"]["fixed_only"] == 1
    assert result["by_layout"]["26,22"]["multi"]["successes"] == 1


def test_summary_rejects_unpaired_start():
    layouts = [(26, 22)]
    fixed = _record("fixed", layouts[0], 10, False, 12)
    multi = _record("multi", layouts[0], 10, True, 3)
    multi["achieved_positions"][0] = [11.0, 30.0]
    with pytest.raises(ValueError, match="start or goal"):
        summarize([fixed, multi], layouts, [10], bootstrap_samples=10)


def test_identity_rejects_changed_artifact(tmp_path):
    path = tmp_path / "identity.json"
    ensure_identity(path, {"model_sha256": "first"})
    ensure_identity(path, {"model_sha256": "first"})
    with pytest.raises(ValueError, match="identity"):
        ensure_identity(path, {"model_sha256": "second"})
