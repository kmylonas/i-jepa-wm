"""Hand-checkable trajectory and supervision audit contracts."""

import csv
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.curriculum_protocol import digest, make_manifest, run_stage


def audit_module():
    name = "src.mylonas_ijepa_wm.audit_curriculum_experience"
    assert importlib.util.find_spec(name), "experience audit module missing"
    return importlib.import_module(name)


def raw_episode(xs, *, goal=(10., 0.), success=True):
    states = np.column_stack((xs, np.zeros(len(xs)))).astype(np.float32)
    return dict(case=dict(case_id="train_0", split="train", goal=list(goal), start=states[0].tolist()),
                success=success, states=torch.from_numpy(states), primitive_steps=torch.arange(len(xs)),
                primitive_actions=torch.from_numpy(np.diff(states, axis=0) / 2))


def test_progress_path_efficiency_and_reversals_distinguish_wandering():
    audit = audit_module()
    straight = audit.analyze_episode(raw_episode([0, 2, 4, 6, 8, 10]), success_radius=1)
    assert straight["path_length"] == 10
    assert straight["success_path_efficiency"] == pytest.approx(.9)
    assert straight["away_fraction"] == 0
    assert straight["strict_closer_fraction"] == 1
    assert straight["strict_away_fraction"] == 0
    assert straight["action_reversal_fraction"] == 0
    wandering = audit.analyze_episode(raw_episode([0, 2, 1, 4, 10]), success_radius=1)
    assert wandering["path_length"] == 12
    assert wandering["success_path_efficiency"] == pytest.approx(.75)
    assert wandering["away_fraction"] == .25
    assert wandering["strict_closer_actions"] == 3
    assert wandering["strict_away_actions"] == 1
    assert wandering["cumulative_distance_increase"] == 1
    assert wandering["action_reversal_fraction"] == pytest.approx(2 / 3)
    assert wandering["mean_action_change"] == pytest.approx(5 / 3)


def test_pair_filter_can_retain_geometrically_wrong_temporal_labels():
    audit = audit_module()
    row = audit.analyze_episode(raw_episode([0, 3, 2, 1, 4, 10]), success_radius=1)
    assert row["pairs_before"]["count"] == 8
    assert row["pairs_after"]["count"] == 8
    assert row["pairs_after"]["contradictory"] == 1
    assert row["pairs_after"]["contradictory_fraction"] == .125
    assert row["pairs_after"]["strict_progress"] == 7
    assert row["pairs_after"]["strict_contradictory"] == 1
    assert row["contradictory_examples"][0]["indices"] == [1, 3]
    assert row["contradictory_examples"][0]["distance_increase"] == 2


def test_pair_filter_erases_revisits_without_renumbering_time():
    audit = audit_module()
    row = audit.analyze_episode(raw_episode([0, 3, 6, 3.1, 8, 10]), success_radius=1)
    assert row["trace"]["retained_indices"] == [0, 3, 4, 5]
    assert row["pairs_after"]["count"] == 4
    assert row["trace"]["steps"] == [0, 1, 2, 3, 4, 5]


def test_pair_summary_matches_episode_uniform_training_sampler():
    audit = audit_module()
    rows = [dict(success=True, pairs_after=dict(count=2, progress=1, ambiguous=0, contradictory=1,
                                                strict_progress=1, strict_contradictory=1)),
            dict(success=True, pairs_after=dict(count=100, progress=100, ambiguous=0, contradictory=0,
                                                strict_progress=100, strict_contradictory=0)),
            dict(success=False, pairs_after=dict(count=2, progress=0, ambiguous=0, contradictory=2,
                                                 strict_progress=0, strict_contradictory=2))]
    summary = audit.summarize_pair_quality(rows, "pairs_after")
    assert summary["usable_successful_episodes"] == 2
    assert summary["episode_uniform"]["contradictory_fraction"] == .25
    assert summary["pooled_pairs"]["contradictory_fraction"] == pytest.approx(1 / 102)


def test_short_or_stationary_episodes_have_defined_values_and_invalid_times_fail():
    audit = audit_module()
    row = audit.analyze_episode(raw_episode([0, 0, 0], success=False))
    assert row["path_length"] == 0
    assert row["success_path_efficiency"] is None
    assert row["stationary_fraction"] == 1
    assert row["pairs_after"]["count"] == 0
    json.dumps(row, allow_nan=False)
    raw = raw_episode([0, 2, 4])
    raw["primitive_steps"] = torch.tensor([0, 1, 3])
    with pytest.raises(ValueError, match="primitive"):
        audit.analyze_episode(raw)


@pytest.fixture
def saved_pilot(tmp_path):
    manifest = make_manifest(eval_per_band=2, train_per_run=3, run_seeds=(0,))
    cases = manifest["training"]["0"]
    for i, case in enumerate(cases):
        case["start"] = [44. + i, 30.]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "runtime.json").write_text(json.dumps(dict(runtime_id="fixture", source_commit="original-pilot",
                                                          signature={"manifest": digest(manifest)})))

    def collect(case):
        start, goal = np.array(case["start"]), np.array(case["goal"])
        direction = (goal - start) / np.linalg.norm(goal - start)
        states = np.array([start + 2 * i * direction for i in range(6)], dtype=np.float32)
        path = f"raw/run_0/{case['case_id']}.pt"
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(runtime_id="fixture", case=case, success=True, states=torch.tensor(states),
                        primitive_steps=torch.arange(6), primitive_actions=torch.tensor(np.diff(states, axis=0) / 2),
                        frames=torch.zeros(6, 224, 224, 3, dtype=torch.uint8), macro_durations=[5]), tmp_path / path)
        return dict(success=True, micro_steps=5, raw_path=path, macro_durations=[5],
                    primitive_actions=(np.diff(states, axis=0) / 2).tolist(), macro_steps=1,
                    mpc_updates=1, achieved_positions=states[[0, -1]].tolist())

    run_stage(tmp_path, "run_0/collect", cases, "fixture", collect)
    (tmp_path / "run_0").mkdir()
    (tmp_path / "run_0/training.json").write_text(json.dumps(dict(
        training_case_ids=[case["case_id"] for case in cases],
        trajectories=[dict(case_id=case["case_id"], frames=6, retained=6, eligible_pairs=10) for case in cases],
        history=[dict(update=1, rank_loss=.7, anchor_loss=0., loss=.7), dict(update=500, rank_loss=.6, anchor_loss=.02, loss=.62)],
    )))
    return tmp_path


def test_saved_audit_outputs_do_not_modify_original_inputs(saved_pilot):
    audit = audit_module()
    files = [path for path in saved_pilot.rglob("*") if path.is_file()]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result = audit.run_audit(saved_pilot, render=False)
    assert result["episodes"] == 3
    assert result["primitive_actions"] == 15
    assert result["frames"] == 18
    assert result["saved_filter_verified"]
    assert result["pair_quality"]["after"]["episode_uniform"]["contradictory_fraction"] == 0
    output = saved_pilot / "experience_audit/run_0"
    assert (output / "summary.json").is_file()
    with (output / "episodes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    for path in files:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before[path]


def test_mixed_raw_provenance_and_changed_filter_counts_are_rejected(saved_pilot):
    audit = audit_module()
    path = next((saved_pilot / "raw/run_0").glob("*.pt"))
    raw = torch.load(path, weights_only=True)
    raw["runtime_id"] = "other-run"
    torch.save(raw, path)
    with pytest.raises(ValueError, match="provenance"):
        audit.run_audit(saved_pilot, render=False)
    raw["runtime_id"] = "fixture"
    torch.save(raw, path)
    raw["macro_durations"] = [2, 3]
    torch.save(raw, path)
    with pytest.raises(ValueError, match="duration"):
        audit.run_audit(saved_pilot, render=False)
    raw["macro_durations"] = [5]
    raw["primitive_actions"][0, 0] += 1
    torch.save(raw, path)
    with pytest.raises(ValueError, match="action"):
        audit.run_audit(saved_pilot, render=False)
    raw["primitive_actions"][0, 0] -= 1
    torch.save(raw, path)
    training_path = saved_pilot / "run_0/training.json"
    training = json.loads(training_path.read_text())
    training["trajectories"][0]["eligible_pairs"] = 999
    training_path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="filter"):
        audit.run_audit(saved_pilot, render=False)


def test_cpu_audit_renders_report_and_lost_cases_without_loading_models(saved_pilot):
    from PIL import Image
    audit = audit_module()
    manifest = json.loads((saved_pilot / "manifest.json").read_text())
    cases = manifest["evaluation"]

    def evaluate(case, learned=False):
        success = case["band"] == "near" or (case["case_id"].endswith("000") and not learned)
        start = np.array(case["start"])
        end = case["goal"] if success else (start + np.array([0, -3])).tolist()
        return dict(success=success, micro_steps=3 if success else 5,
                    achieved_positions=[case["start"], end], macro_durations=[3 if success else 5], mpc_updates=1)

    run_stage(saved_pilot, "baseline", cases, "fixture", evaluate)
    run_stage(saved_pilot, "run_0/evaluate", cases, "fixture", lambda case: evaluate(case, True))
    result = audit.run_audit(saved_pilot)
    assert [case["case_id"] for case in result["evaluation"]["lost_cases"]] == ["eval_far_000"]
    lost = result["evaluation"]["lost_cases"][0]
    assert lost["traces"]["raw"]["steps"] == [0, 3]
    assert lost["traces"]["adapted"]["steps"] == [0, 5]
    output = saved_pilot / "experience_audit/run_0"
    for name in ("progress", "pair_quality", "coverage", "episode_examples", "frames", "training_pair_examples", "lost_cases"):
        with Image.open(output / f"{name}.png") as image:
            assert image.width > 400 and image.height > 200
