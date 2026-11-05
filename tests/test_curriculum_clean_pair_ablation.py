"""Immutable-parent workflow for the clean oracle-pair ablation."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.mylonas_ijepa_wm.curriculum_protocol import digest, make_manifest, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256


def ablation_module():
    name = "src.mylonas_ijepa_wm.curriculum_clean_pair_ablation"
    assert importlib.util.find_spec(name), "clean-pair ablation runner missing"
    return importlib.import_module(name)


def episode_result(case, success=True, runtime_id=None):
    end = case["goal"] if success else [case["start"][0], case["start"][1] - 2]
    return dict(success=success, micro_steps=4, macro_steps=1, mpc_updates=1,
                macro_durations=[4], primitive_actions=[[1., 0.]] * 4,
                achieved_positions=[case["start"], end], final_goal_distance=3. if success else 12.)


@pytest.fixture
def parent_pilot(tmp_path):
    root = tmp_path / "pilot_v1"
    root.mkdir()
    manifest = make_manifest(eval_per_band=2, train_per_run=3, run_seeds=(0,))
    manifest["fitting"].update(updates=2, batch_size=1, bottleneck=3)
    goal = np.array([50., 42.])
    offsets = np.array([[7., 0.], [6., 2.], [0., 6.5], [8., 0.], [3., 0.]], np.float32)
    for case in manifest["training"]["0"]:
        case["start"] = (goal + offsets[0]).tolist()
    write_json(root / "manifest.json", manifest)
    runtime_id = "parent-runtime"
    write_json(root / "runtime.json", dict(runtime_id=runtime_id, source_commit="parent-source",
                                             signature={"manifest": digest(manifest),
                                                        "models": {"world_model": "wm", "encoder": "enc"}}))

    evaluation = manifest["evaluation"]
    raw_success = {case["case_id"]: case["band"] == "near" or case["case_id"].endswith("000")
                   for case in evaluation}
    run_stage(root, "baseline", evaluation, runtime_id,
              lambda case: episode_result(case, raw_success[case["case_id"]]),
              condition={"cost": "latent", "adapter_sha256": None})
    cases = manifest["training"]["0"]
    run_stage(root, "run_0/collect", cases, runtime_id, episode_result,
              condition={"cost": "latent", "adapter_sha256": None})

    replay_dir = root / "replay/run_0"
    replay_dir.mkdir(parents=True)
    states = torch.tensor(goal + offsets)
    for index, case in enumerate(cases):
        torch.save(dict(runtime_id=runtime_id, case=case, case_id=case["case_id"], split="train",
                        success=True, states=states, primitive_steps=torch.arange(5),
                        latents=(torch.arange(20).reshape(5, 1, 4) + index).half(),
                        latent_storage_dtype="float16"), replay_dir / f"{case['case_id']}.pt")
    (root / "run_0").mkdir()
    (root / "run_0/adapter.pt").write_bytes(b"original-adapter")
    write_json(root / "run_0/training.json", dict(training_case_ids=[case["case_id"] for case in cases]))
    write_json(root / "stages/run_0/train.json", dict(status="complete", runtime_id=runtime_id))
    original_sha = file_sha256(root / "run_0/adapter.pt")
    run_stage(root, "run_0/evaluate", evaluation, runtime_id,
              lambda case: episode_result(case, case["band"] == "near"),
              condition={"cost": "latent", "adapter_sha256": original_sha})
    write_json(root / "report.json", {"runtime_id": runtime_id, "runs": {"0": {}}})
    return root


def file_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def write_child_runtime(root, runtime_id="child-runtime"):
    manifest = json.loads((root / "manifest.json").read_text())
    write_json(root / "runtime.json", dict(runtime_id=runtime_id,
                                             signature={"manifest": digest(manifest),
                                                        "models": {"world_model": "wm", "encoder": "enc"}}))
    return runtime_id


def test_prepare_validates_complete_parent_and_never_mutates_it(parent_pilot, tmp_path):
    clean = ablation_module()
    child = tmp_path / "clean_pair_v1"
    before = file_hashes(parent_pilot)
    config = clean.prepare_ablation(parent_pilot, child)
    assert config["parent_runtime_id"] == "parent-runtime"
    assert config["parent_models"] == {"world_model": "wm", "encoder": "enc"}
    assert config["run_seed"] == 0
    assert config["pair_filter"] == {"kind": "oracle_goal_progress", "min_goal_progress": .5}
    assert config["replay_episodes"] == 3
    assert len(config["replay_files"]) == 3
    assert json.loads((child / "manifest.json").read_text()) == json.loads((parent_pilot / "manifest.json").read_text())
    assert file_hashes(parent_pilot) == before
    assert clean.prepare_ablation(parent_pilot, child) == config

    (parent_pilot / "report.json").unlink()
    with pytest.raises(ValueError, match="complete parent"):
        clean.prepare_ablation(parent_pilot, tmp_path / "other")


def test_changed_parent_replay_is_rejected_after_prepare(parent_pilot, tmp_path):
    clean = ablation_module()
    child = tmp_path / "clean_pair_v1"
    clean.prepare_ablation(parent_pilot, child)
    replay = next((parent_pilot / "replay/run_0").glob("*.pt"))
    saved = torch.load(replay, weights_only=True)
    saved["states"][0, 0] += 1
    torch.save(saved, replay)
    with pytest.raises(ValueError, match="replay.*changed"):
        clean.load_parent_replay(parent_pilot, child)


def test_clean_training_saves_goal_rule_and_parent_child_provenance(parent_pilot, tmp_path):
    clean = ablation_module()
    child = tmp_path / "clean_pair_v1"
    clean.prepare_ablation(parent_pilot, child)
    parent_before = file_hashes(parent_pilot)
    runtime_id = write_child_runtime(child)
    calls = []
    checkpoint = clean.train_clean_stage(
        parent_pilot, child, runtime_id, goal=lambda: calls.append(True) or torch.zeros(1, 4), device="cpu",
    )
    assert calls == [True]
    training = json.loads((child / "run_0/training.json").read_text())
    assert training["pair_filter"]["min_goal_progress"] == .5
    assert all(row["temporal_pairs"] > row["eligible_pairs"] for row in training["trajectories"])
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["provenance"]["runtime_id"] == runtime_id
    assert saved["provenance"]["parent_runtime_id"] == "parent-runtime"
    goal = torch.load(child / "run_0/goal_tokens.pt", weights_only=True)
    assert goal["runtime_id"] == runtime_id
    assert torch.equal(goal["tokens"], torch.zeros(1, 4))
    first_mtime = checkpoint.stat().st_mtime_ns
    clean.train_clean_stage(parent_pilot, child, runtime_id,
                            goal=lambda: pytest.fail("completed training must resume"), device="cpu")
    assert checkpoint.stat().st_mtime_ns == first_mtime
    assert file_hashes(parent_pilot) == parent_before


def test_report_pairs_parent_and_child_runtimes_and_applies_clean_screen(parent_pilot, tmp_path):
    clean = ablation_module()
    child = tmp_path / "clean_pair_v1"
    config = clean.prepare_ablation(parent_pilot, child)
    runtime_id = write_child_runtime(child)
    checkpoint = clean.train_clean_stage(parent_pilot, child, runtime_id, goal=torch.zeros(1, 4), device="cpu")
    manifest = json.loads((child / "manifest.json").read_text())
    evaluation = manifest["evaluation"]
    condition = clean.clean_condition(config, file_sha256(checkpoint))
    run_stage(child, "run_0/evaluate", evaluation, runtime_id,
              lambda case: episode_result(case, case["band"] == "near" or case["case_id"].endswith("000")),
              condition=condition)
    report = clean.make_clean_report(parent_pilot, child, render=True)
    assert report["parent_runtime_id"] == "parent-runtime"
    assert report["runtime_id"] == runtime_id
    assert report["clean_vs_raw"]["far"]["adapted_successes"] == 1
    assert report["original_vs_raw"]["far"]["adapted_successes"] == 0
    assert report["clean_screen"]["passed"]
    assert report["outcome"] == "avoided_degradation_without_far_gain"
    assert report["interaction_budgets"]["clean_evaluation_completed"] == 16
    for name in ("success_by_band.png", "trajectories.png", "training_loss.png"):
        assert (child / name).is_file()

    record = child / "episodes/run_0/evaluate" / f"{evaluation[0]['case_id']}.json"
    changed = json.loads(record.read_text())
    changed["case"]["start"] = [1., 2.]
    record.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="provenance|paired"):
        clean.make_clean_report(parent_pilot, child, render=False)
