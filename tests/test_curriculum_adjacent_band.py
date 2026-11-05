"""Three frozen costs evaluated on the same adjacent-band cases."""

import importlib
import importlib.util
import json

import pytest
import torch

from test_curriculum_clean_pair_ablation import (
    parent_pilot, file_hashes, write_child_runtime, episode_result,
)
from src.mylonas_ijepa_wm import curriculum_clean_pair_ablation as clean_runner
from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import InteractionJournal, file_sha256


def runner():
    name = "src.mylonas_ijepa_wm.curriculum_adjacent_band"
    assert importlib.util.find_spec(name), "adjacent evaluation runner missing"
    return importlib.import_module(name)


@pytest.fixture
def sources(parent_pilot, tmp_path):
    # A real loadable source adapter, with the parent evaluation bound to its hash.
    training = json.loads((parent_pilot / "run_0/training.json").read_text())
    save_adapter(parent_pilot / "run_0/adapter.pt", ResidualTokenAdapter(4, 3), training,
                 {"runtime_id": "parent-runtime"})
    sha = file_sha256(parent_pilot / "run_0/adapter.pt")
    for path in [parent_pilot / "stages/run_0/evaluate.json",
                 *(parent_pilot / "episodes/run_0/evaluate").glob("*.json")]:
        value = json.loads(path.read_text())
        value["condition"]["adapter_sha256"] = sha
        write_json(path, value)
    clean = tmp_path / "clean_pair_v1"
    clean_runner.prepare_ablation(parent_pilot, clean)
    rid = write_child_runtime(clean)
    clean_runner.train_clean_stage(parent_pilot, clean, rid, goal=torch.zeros(1, 4), device="cpu")
    return parent_pilot, clean


class SyntheticRuntime:
    """Replace only the GPU/simulator boundary, retain real stages and journals."""

    def __init__(self, root, manifest, config, condition, *, interrupt=None):
        self.root = root / "conditions" / condition
        self.device = torch.device("cpu")
        self.manifest = manifest
        self.condition = condition
        self.interrupt = interrupt
        self.visited = []
        signature = dict(manifest=digest(manifest), models=config["sources"]["models"],
                         runtime_variant=dict(kind="adjacent_band_evaluation",
                                              experiment_digest=digest(config), condition=condition))
        self.runtime_id = digest(signature)
        write_json(self.root / "runtime.json", dict(runtime_id=self.runtime_id, signature=signature))
        self.goal = None

    def run_case(self, case, *, metric=None):
        assert (metric is None) == (self.condition == "raw")
        assert torch.equal(self.goal.cpu(), torch.zeros(1, 4))
        self.visited.append(case["case_id"])
        stage = "baseline" if metric is None else "run_0/evaluate"
        with InteractionJournal(self.root, self.runtime_id, stage, case["case_id"]) as journal:
            for i in range(4):
                journal.record(i + 1, [1., 0.])
                if case["case_id"] == self.interrupt and i == 1:
                    raise RuntimeError("interrupted")
        index = int(case["case_id"].split("_")[-1])
        failed = {"raw": {18, 19}, "original": {0, 19}, "clean": {0, 1, 2, 19}}
        return episode_result(case, index not in failed[self.condition])


def prepared(sources, tmp_path):
    mod = runner()
    output = tmp_path / "adjacent_band_v1"
    mod.prepare_evaluation(*sources, output)
    return mod, output, json.loads((output / "manifest.json").read_text()), json.loads((output / "experiment.json").read_text())


def test_prepare_makes_twenty_fixed_adjacent_cases_without_source_writes(sources, tmp_path):
    mod = runner()
    before = [file_hashes(path) for path in sources]
    _, output, manifest, config = prepared(sources, tmp_path)
    mod.prepare_evaluation(*sources, output)
    assert [file_hashes(path) for path in sources] == before
    assert len(manifest["evaluation"]) == 20
    assert len({case["case_id"] for case in manifest["evaluation"]}) == 20
    for case in manifest["evaluation"]:
        assert case["split"] == "eval" and case["band"] == "adjacent"
        assert 40 <= case["start"][0] <= 54 and 22 <= case["start"][1] <= 30
        assert case["goal"] == [50., 42.] and case["layout"] == [32, 30]
    other = tmp_path / "repeat"
    mod.prepare_evaluation(*sources, other)
    assert json.loads((other / "manifest.json").read_text()) == manifest
    assert config["sources"]["original"]["runtime_id"] == "parent-runtime"
    assert config["sources"]["clean"]["runtime_id"] == "child-runtime"


def test_output_cannot_alias_or_descend_into_either_source(sources, tmp_path):
    mod = runner()
    for index, source in enumerate(sources):
        for output in (source, source / "nested"):
            with pytest.raises(ValueError, match="outside"):
                mod.prepare_evaluation(*sources, output)
        alias = tmp_path / f"alias-{index}"
        alias.symlink_to(source, target_is_directory=True)
        with pytest.raises(ValueError, match="outside"):
            mod.prepare_evaluation(*sources, alias / "nested")


@pytest.mark.parametrize("relative", ["run_0/adapter.pt", "run_0/goal_tokens.pt", "runtime.json"])
def test_changed_sources_rejected_before_case_execution(sources, tmp_path, relative):
    mod, output, manifest, config = prepared(sources, tmp_path)
    runtime = SyntheticRuntime(output, manifest, config, "raw")
    path = sources[1] / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        mod.evaluate_condition(*sources, output, "raw", runtime)
    assert not runtime.visited


def test_three_conditions_and_paired_reports_use_same_cases(sources, tmp_path):
    mod, output, manifest, config = prepared(sources, tmp_path)
    before = [file_hashes(path) for path in sources]
    for condition in ("raw", "original", "clean"):
        runtime = SyntheticRuntime(output, manifest, config, condition)
        records = mod.evaluate_condition(*sources, output, condition, runtime)
        assert [row["case"] for row in records] == manifest["evaluation"]
    report = mod.make_report(*sources, output, render=True)
    assert report["raw_competence"]["passed"]
    assert report["conditions"]["raw"]["successes"] == 18
    assert report["conditions"]["original"]["successes"] == 18
    assert report["conditions"]["clean"]["successes"] == 16
    assert report["comparisons"]["original"]["wins"] == 1
    assert report["comparisons"]["original"]["losses"] == 1
    assert report["comparisons"]["original"]["retention_passed"]
    assert report["comparisons"]["clean"]["wins"] == 1
    assert report["comparisons"]["clean"]["losses"] == 3
    assert not report["comparisons"]["clean"]["retention_passed"]
    assert report["comparisons"]["clean"]["lost_case_ids"] == ["eval_adjacent_000", "eval_adjacent_001", "eval_adjacent_002"]
    for condition in ("raw", "original", "clean"):
        assert report["conditions"][condition]["primitive_steps"] == 80
        assert report["conditions"][condition]["journal"]["primitive_interactions"] == 80
    assert [file_hashes(path) for path in sources] == before
    for name in ("success_comparison.png", "trajectories.png"):
        assert (output / name).stat().st_size > 1000
    record_path = output / "conditions/clean/episodes/run_0/evaluate/eval_adjacent_000.json"
    record = json.loads(record_path.read_text())
    record["case"]["seed"] += 1
    write_json(record_path, record)
    with pytest.raises(ValueError, match="provenance|paired"):
        mod.make_report(*sources, output, render=False)


def test_interrupted_evaluation_resumes_complete_cases_and_counts_all_actions(sources, tmp_path):
    mod, output, manifest, config = prepared(sources, tmp_path)
    runtime = SyntheticRuntime(output, manifest, config, "original", interrupt="eval_adjacent_001")
    with pytest.raises(RuntimeError, match="interrupted"):
        mod.evaluate_condition(*sources, output, "original", runtime)
    runtime.interrupt = None
    records = mod.evaluate_condition(*sources, output, "original", runtime)
    assert len(records) == 20
    assert runtime.visited.count("eval_adjacent_000") == 1
    assert runtime.visited.count("eval_adjacent_001") == 2
    from src.mylonas_ijepa_wm.curriculum_runtime import interaction_budget
    budget = interaction_budget(runtime.root, runtime.runtime_id)
    assert budget["primitive_interactions"] == 82 and budget["interrupted_attempts"] == 1


def test_changed_evaluation_runtime_or_model_is_rejected(sources, tmp_path):
    mod, output, manifest, config = prepared(sources, tmp_path)
    runtime = SyntheticRuntime(output, manifest, config, "original")
    saved = json.loads((runtime.root / "runtime.json").read_text())
    saved["signature"]["models"]["world_model"] = "wrong"
    write_json(runtime.root / "runtime.json", saved)
    with pytest.raises(ValueError, match="runtime"):
        mod.evaluate_condition(*sources, output, "original", runtime)
    assert runtime.visited == []


def test_changed_output_manifest_cannot_resume(sources, tmp_path):
    mod, output, manifest, config = prepared(sources, tmp_path)
    runtime = SyntheticRuntime(output, manifest, config, "raw")
    altered = json.loads((output / "manifest.json").read_text())
    altered["evaluation"][0]["seed"] += 1
    write_json(output / "manifest.json", altered)
    with pytest.raises(ValueError, match="manifest"):
        mod.evaluate_condition(*sources, output, "raw", runtime)
    assert runtime.visited == []


def test_retention_uninterpretable_when_raw_incompetent(sources, tmp_path):
    mod, output, manifest, config = prepared(sources, tmp_path)
    for condition in ("raw", "original", "clean"):
        runtime = SyntheticRuntime(output, manifest, config, condition)
        mod.evaluate_condition(*sources, output, condition, runtime)
    path = output / "conditions/raw/episodes/baseline/eval_adjacent_000.json"
    record = json.loads(path.read_text())
    record["success"] = False
    write_json(path, record)
    report = mod.make_report(*sources, output, render=False)
    assert not report["raw_competence"]["passed"]
    assert report["comparisons"]["original"]["retention_passed"] is None
    assert report["comparisons"]["clean"]["retention_passed"] is None
