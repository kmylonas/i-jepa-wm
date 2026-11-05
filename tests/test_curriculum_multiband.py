"""Sequential curricula with real fit/checkpoint/stage logic and a synthetic simulator."""

import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_curriculum_clean_pair_ablation import file_hashes, write_child_runtime, episode_result
from test_curriculum_adjacent_band import SyntheticRuntime as AdjacentRuntime
from src.mylonas_ijepa_wm import curriculum_clean_pair_ablation as clean_runner
from src.mylonas_ijepa_wm import curriculum_adjacent_band as adjacent_runner
from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter
from src.mylonas_ijepa_wm.curriculum_protocol import digest, make_manifest, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_replay import save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import InteractionJournal, PilotRuntime, file_sha256, save_tensors


def module():
    name = "src.mylonas_ijepa_wm.curriculum_multiband"
    assert importlib.util.find_spec(name), "multi-band runner missing"
    return importlib.import_module(name)


@pytest.fixture
def curriculum_sources(tmp_path):
    parent = tmp_path / "pilot_v1"
    manifest = make_manifest(eval_per_band=20, train_per_run=3, run_seeds=(0,))
    manifest["fitting"].update(updates=2, batch_size=2, bottleneck=3)
    write_json(parent / "manifest.json", manifest)
    rid = "parent-runtime"
    write_json(parent / "runtime.json", dict(runtime_id=rid,
        signature=dict(manifest=digest(manifest), models={"world_model": "wm", "encoder": "enc"})))
    training = {"training_case_ids": [case["case_id"] for case in manifest["training"]["0"]]}
    save_adapter(parent / "run_0/adapter.pt", ResidualTokenAdapter(4, 3), training, {"runtime_id": rid})
    write_json(parent / "run_0/training.json", training)
    write_json(parent / "stages/run_0/train.json", dict(status="complete", runtime_id=rid,
        checkpoint_sha256=file_sha256(parent / "run_0/adapter.pt")))
    for stage, cases, sha in (("baseline", manifest["evaluation"], None),
                             ("run_0/collect", manifest["training"]["0"], None),
                             ("run_0/evaluate", manifest["evaluation"], file_sha256(parent / "run_0/adapter.pt"))):
        run_stage(parent, stage, cases, rid, episode_result, condition={"cost": "latent", "adapter_sha256": sha})
    for case in manifest["training"]["0"]:
        states = torch.stack([torch.tensor(case["start"]) * (1-t) + torch.tensor(case["goal"]) * t
                              for t in (0., .25, .5, .75, 1.)])
        save_tensors(parent / "replay/run_0" / f"{case['case_id']}.pt",
            dict(runtime_id=rid, case=case, case_id=case["case_id"], split="train", success=True,
                 states=states, primitive_steps=torch.arange(5), latents=torch.arange(20).reshape(5, 1, 4).half()))
    write_json(parent / "report.json", {"runtime_id": rid})
    clean = tmp_path / "clean_pair_v1"
    clean_runner.prepare_ablation(parent, clean)
    cid = write_child_runtime(clean)
    clean_runner.train_clean_stage(parent, clean, cid, goal=torch.zeros(1, 4), device="cpu")
    adjacent = tmp_path / "adjacent_band_v1"
    config = adjacent_runner.prepare_evaluation(parent, clean, adjacent)
    am = json.loads((adjacent / "manifest.json").read_text())
    for condition in ("raw", "original", "clean"):
        runtime = AdjacentRuntime(adjacent, am, config, condition)
        adjacent_runner.evaluate_condition(parent, clean, adjacent, condition, runtime)
    adjacent_runner.make_report(parent, clean, adjacent, render=False)
    return parent, clean, adjacent


class TinyRuntime:
    def __init__(self, curriculum, recipe, *, success=True, interrupt=None):
        self.root = curriculum.root / ("raw" if recipe == "raw" else f"branches/{recipe}")
        self.manifest = curriculum.manifest
        self.recipe, self.success, self.interrupt = recipe, success, interrupt
        self.device, self.precision = torch.device("cpu"), "fp32"
        self.args = SimpleNamespace(run_seed=0, videos=False)
        self.goal = None
        self.visited = []
        self.encode_calls = 0
        signature = dict(manifest=digest(self.manifest), models=curriculum.config["sources"]["models"],
            source="synthetic", precision="fp32", device_type="cpu", encoder_batch_size=8,
            runtime_variant=dict(kind="multiband_curriculum", experiment_digest=digest(curriculum.config), recipe=recipe))
        self.runtime_id = digest(signature)
        write_json(self.root / "runtime.json", dict(runtime_id=self.runtime_id, signature=signature))
        self.env = SimpleNamespace(
            env=SimpleNamespace(border_padding=7.4, img_size=65),
            reset=lambda *a, **k: None,
            _render_at=lambda point: np.full((2, 2, 3), int(point[1]), np.uint8),
        )

    def encode_frames(self, frames):
        self.encode_calls += 1
        values = torch.as_tensor(np.asarray(frames)).float().mean(dim=(1, 2, 3))
        return values[:, None, None].expand(-1, 1, 4).clone() + torch.tensor([1., 2., 3., 4.])

    def run_case(self, case, *, metric=None, capture=False, latent_cost_fn=None, journal_stage=None):
        assert (metric is None) == (self.recipe == "raw")
        assert journal_stage is not None
        assert torch.equal(self.goal, torch.zeros(1, 4))
        if self.recipe == "raw":
            assert torch.equal(latent_cost_fn(torch.full((1, 1, 4), 2.)), torch.tensor([4.]))
        self.visited.append((journal_stage, case["case_id"], capture))
        with InteractionJournal(self.root, self.runtime_id, journal_stage, case["case_id"]) as journal:
            for step in range(4):
                journal.record(step + 1, [1., 0.])
                if self.interrupt == case["case_id"] and step == 1:
                    raise RuntimeError("interrupted")
        result = episode_result(case, self.success if capture else True)
        result["attempt_id"] = journal.identity["attempt_id"]
        result["raw_path"] = None
        if capture:
            path = Path("raw/run_0") / f"{case['case_id']}.pt"
            states = torch.stack([torch.tensor(case["start"]) * (1-t) + torch.tensor(case["goal"]) * t
                                  for t in (0., .25, .5, .75, 1.)])
            save_tensors(self.root / path, dict(runtime_id=self.runtime_id, case=case,
                success=result["success"], frames=torch.arange(5, dtype=torch.uint8)[:, None, None, None].expand(-1, 2, 2, 3),
                states=states, primitive_steps=torch.arange(5), primitive_actions=torch.ones(4, 2),
                macro_durations=[4]))
            result["raw_path"] = path.as_posix()
        return result

    def encode_episode(self, record):
        return PilotRuntime.encode_episode(self, record)


def prepared(curriculum_sources, tmp_path):
    mod = module()
    output = tmp_path / "multiband_curriculum_v1"
    mod.prepare(*curriculum_sources, output)
    return mod.Curriculum(*curriculum_sources, output)


def baseline(curriculum):
    runtime = TinyRuntime(curriculum, "raw")
    curriculum.grid(runtime)
    curriculum.baseline(runtime)
    return runtime


def test_manifest_reuses_cases_tiles_downward_and_disjoins_collection(curriculum_sources, tmp_path):
    before = [file_hashes(source) for source in curriculum_sources]
    curriculum = prepared(curriculum_sources, tmp_path)
    manifest = curriculum.manifest
    parent = json.loads((curriculum_sources[0] / "manifest.json").read_text())
    adjacent = json.loads((curriculum_sources[2] / "manifest.json").read_text())
    assert manifest["evaluation"]["near"] == [case for case in parent["evaluation"] if case["band"] == "near"]
    assert manifest["evaluation"]["band_1"] == adjacent["evaluation"]
    eval_starts = np.array([case["start"] for rows in manifest["evaluation"].values() for case in rows])
    for key, lo, hi in (("band_1", 22, 30), ("band_2", 14, 22), ("band_3", 8, 14)):
        assert len(manifest["evaluation"][key]) == 20
        assert len(manifest["collection"][key]) == 40
        for case in manifest["evaluation"][key] + manifest["collection"][key]:
            assert 40 <= case["start"][0] <= 54 and lo <= case["start"][1] <= hi
        for case in manifest["collection"][key]:
            assert np.linalg.norm(eval_starts-np.array(case["start"]), axis=1).min() >= .25
    assert [file_hashes(source) for source in curriculum_sources] == before


def test_curriculum_updates_chain_cumulative_replay_and_retention(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    runtime = TinyRuntime(curriculum, "original")
    curriculum.initial("original", runtime)
    source_hashes = [file_hashes(source) for source in curriculum_sources]
    first = curriculum.band("original", 1, runtime)
    assert first["status"] == "complete"
    training = json.loads((runtime.root / "updates/v1/training.json").read_text())
    assert training["warm_start"] and training["optimizer_reset"]
    assert training["band_episode_counts"] == {"near": 3, "band_1": 40}
    second = curriculum.band("original", 2, runtime)
    assert second["status"] == "complete"
    training2 = json.loads((runtime.root / "updates/v2/training.json").read_text())
    assert training2["band_episode_counts"] == {"near": 3, "band_1": 40, "band_2": 40}
    assert second["inputs"]["incoming_sha256"] == first["checkpoint_sha256"]
    assert second["inputs"]["recipe"] == "original"
    before_visits, before_encodes = len(runtime.visited), runtime.encode_calls
    curriculum.band("original", 2, runtime)
    assert len(runtime.visited) == before_visits and runtime.encode_calls == before_encodes
    report = curriculum.report()
    assert report["branches"]["original"]["versions"]["2"]["near"]["successes"] == 20
    assert report["branches"]["original"]["versions"]["2"]["band_3"] is None
    assert report["branches"]["clean"]["status"] == "not_started"
    assert [file_hashes(source) for source in curriculum_sources] == source_hashes


def test_stopped_branch_does_not_feed_other_branch_or_block_report(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    original = TinyRuntime(curriculum, "original", success=False)
    curriculum.initial("original", original)
    stopped = curriculum.band("original", 1, original)
    assert stopped["status"] == "stopped" and "usable" in stopped["reason"]
    assert not (original.root / "checkpoints/v1.pt").exists()
    clean = TinyRuntime(curriculum, "clean")
    curriculum.initial("clean", clean)
    assert curriculum.band("clean", 1, clean)["status"] == "complete"
    report = curriculum.report()
    assert report["branches"]["original"]["status"] == "stopped"
    assert report["branches"]["clean"]["status"] == "active"
    status = json.loads((clean.root / "updates/v1/status.json").read_text())
    assert all("branches/original" not in row["path"] for row in status["inputs"]["replay"])
    training = json.loads((clean.root / "updates/v1/training.json").read_text())
    assert training["pair_filter"]["min_goal_progress"] == .5


@pytest.mark.parametrize("artifact", ["checkpoints/v1.pt", "replay"])
def test_changed_checkpoint_or_replay_cannot_resume(curriculum_sources, tmp_path, artifact):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    runtime = TinyRuntime(curriculum, "clean")
    curriculum.initial("clean", runtime)
    curriculum.band("clean", 1, runtime)
    path = next((runtime.root / "replay/run_0").glob("*.pt")) if artifact == "replay" else runtime.root / artifact
    path.write_bytes(path.read_bytes()+b"changed")
    before = len(runtime.visited)
    with pytest.raises(ValueError, match="checkpoint|hash"):
        curriculum.band("clean", 2, runtime)
    assert len(runtime.visited) == before


def test_interrupted_collection_resumes_only_missing_cases(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    case = curriculum.manifest["collection"]["band_1"][1]
    runtime = TinyRuntime(curriculum, "original", interrupt=case["case_id"])
    curriculum.initial("original", runtime)
    with pytest.raises(RuntimeError, match="interrupted"):
        curriculum.band("original", 1, runtime)
    runtime.interrupt = None
    curriculum.band("original", 1, runtime)
    first_id = curriculum.manifest["collection"]["band_1"][0]["case_id"]
    captures = [case_id for _, case_id, capture in runtime.visited if capture]
    assert captures.count(first_id) == 1 and captures.count(case["case_id"]) == 2
    budget = curriculum.report()["branches"]["original"]["budget"]
    assert budget["interrupted_attempts"] == 1
    assert budget["primitive_interactions"] == 482  # 20 near + 20 pre + 40 collect + 40 post, 4 actions each; plus 2 interrupted.


def test_source_and_nested_output_aliases_are_rejected(curriculum_sources, tmp_path):
    mod = module()
    for source in curriculum_sources:
        with pytest.raises(ValueError, match="outside"):
            mod.prepare(*curriculum_sources, source / "nested")
    curriculum = prepared(curriculum_sources, tmp_path)
    (curriculum.root / "bad-link").symlink_to(curriculum_sources[0] / "unexpected")
    with pytest.raises(ValueError, match="symlink"):
        curriculum.baseline(TinyRuntime(curriculum, "raw"))


def test_later_mixed_collection_record_rejected_before_missing_case_actions(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    runtime = TinyRuntime(curriculum, "original")
    curriculum.initial("original", runtime)
    curriculum.band("original", 1, runtime)
    cases = curriculum.manifest["collection"]["band_1"]
    folder = runtime.root / "episodes/collect/band_1"
    (folder / f"{cases[0]['case_id']}.json").unlink()
    path = folder / f"{cases[-1]['case_id']}.json"
    changed = json.loads(path.read_text())
    changed["case"]["seed"] += 1
    write_json(path, changed)
    before = len(runtime.visited)
    with pytest.raises(ValueError, match="provenance"):
        curriculum.band("original", 1, runtime)
    assert len(runtime.visited) == before


def test_runtime_journals_explicit_curriculum_stage(tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm import curriculum_runtime as runtime_module
    runtime = SimpleNamespace(root=tmp_path, runtime_id="stage-test", args=SimpleNamespace(videos=False, run_seed=0),
        manifest={"planning": {}}, config=None, encoder=None, model=None, metadata=None,
        device=torch.device("cpu"), precision="fp32", load_encoder=lambda: None,
        load_world_model=lambda: None, env=SimpleNamespace(reset=lambda *a, **kw: None))
    def episode(*args, observation_observer=None, **kwargs):
        observation_observer(1, None, [1., 0.])
        return {"success": True}
    monkeypatch.setattr(runtime_module.plan_proprio, "run_episode", episode)
    PilotRuntime.run_case(runtime, {"case_id": "eval_band_2_001", "seed": 1, "layout": [32, 30],
        "start": [44., 18.], "goal": [50., 42.]}, journal_stage="eval/v1/band_2")
    from src.mylonas_ijepa_wm.curriculum_runtime import interaction_budget
    assert interaction_budget(tmp_path, "stage-test")["by_stage"] == {"eval/v1/band_2": 1}


def test_invalid_branch_preserves_other_report_and_valid_budget(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    for recipe in ("original", "clean"):
        runtime = TinyRuntime(curriculum, recipe)
        curriculum.initial(recipe, runtime)
        curriculum.band(recipe, 1, runtime)
    path = curriculum.root / "branches/original/checkpoints/v1.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    report = curriculum.report(render=True)
    assert report["branches"]["original"]["status"] == "invalid"
    assert "checkpoint" in report["branches"]["original"]["error"]
    assert report["branches"]["original"]["versions"]["1"]["near"] is None
    assert report["branches"]["clean"]["versions"]["1"]["near"]["successes"] == 20
    assert report["total_primitive_interactions"] == 1280
    assert not report["total_count_is_lower_bound"]


@pytest.mark.parametrize("damage", ["missing", "attempt", "stage", "case", "count"])
def test_damaged_journal_blocks_actions_and_marks_budget_unavailable(curriculum_sources, tmp_path, damage):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    runtime = TinyRuntime(curriculum, "original")
    curriculum.initial("original", runtime)
    curriculum.band("original", 1, runtime)
    path = runtime.root / "interactions.jsonl"
    if damage == "missing":
        path.unlink()
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        attempt = rows[0]["attempt_id"]
        if damage == "count":
            rows = [row for row in rows if not (row["attempt_id"] == attempt and row.get("step") == 4)]
        else:
            key = {"attempt": "attempt_id", "stage": "stage", "case": "case_id"}[damage]
            for row in rows:
                if row["attempt_id"] == attempt:
                    row[key] = "changed"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = len(runtime.visited)
    with pytest.raises(ValueError, match="journal"):
        curriculum.band("original", 2, runtime)
    assert len(runtime.visited) == before
    report = curriculum.report()
    assert report["branches"]["original"]["budget"]["status"] == "unavailable"
    assert report["branches"]["original"]["budget"]["primitive_interactions"] is None
    assert report["total_primitive_interactions"] is None
    assert report["total_count_is_lower_bound"]
    assert report["known_primitive_interactions"] == 320


def test_truncated_interrupted_journal_remains_resumable(curriculum_sources, tmp_path):
    curriculum = prepared(curriculum_sources, tmp_path)
    baseline(curriculum)
    runtime = TinyRuntime(curriculum, "original")
    curriculum.initial("original", runtime)
    with (runtime.root / "interactions.jsonl").open("a") as stream:
        stream.write('{"event":')
    curriculum.band("original", 1, runtime)
    report = curriculum.report()
    assert report["branches"]["original"]["budget"]["count_is_lower_bound"]
    assert report["total_count_is_lower_bound"]


@pytest.mark.parametrize("operation", ["report", "baseline"])
def test_corrupt_existing_grid_rejected_before_report_or_actions(curriculum_sources, tmp_path, operation):
    curriculum = prepared(curriculum_sources, tmp_path)
    runtime = baseline(curriculum)
    path = curriculum.root / "cache/grid.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    before = len(runtime.visited)
    with pytest.raises(ValueError, match="grid"):
        curriculum.report() if operation == "report" else curriculum.baseline(runtime)
    assert len(runtime.visited) == before


def no_anchor_prepared(curriculum_sources, tmp_path):
    mod = module()
    output = tmp_path / 'multiband_no_anchor_v1'
    mod.prepare(*curriculum_sources, output, without_anchor=True)
    return mod.Curriculum(*curriculum_sources, output)


def test_no_anchor_variant_keeps_cases_and_rejects_configuration_reuse(curriculum_sources, tmp_path):
    before = [file_hashes(path) for path in curriculum_sources]
    anchored = prepared(curriculum_sources, tmp_path)
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    assert free.manifest['fitting']['anchor_weight'] == 0
    assert anchored.manifest['fitting']['anchor_weight'] == 1
    assert free.manifest['evaluation'] == anchored.manifest['evaluation']
    assert free.manifest['collection'] == anchored.manifest['collection']
    assert free.config['ablation'] == 'ranking_only_from_identity'
    with pytest.raises(ValueError, match='configuration'):
        module().prepare(*curriculum_sources, free.root)
    assert [file_hashes(path) for path in curriculum_sources] == before


def test_no_anchor_initial_fit_matches_identity_fit_and_resumes_without_refitting(curriculum_sources, tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm.curriculum_replay import fit_adapter, load_adapter
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    baseline(free)
    runtime = TinyRuntime(free, 'original')
    free.initial('original', runtime)
    path, sha, rid = free.reference('original', 0)
    assert path == runtime.root / 'checkpoints/v0.pt'
    assert rid == runtime.runtime_id
    actual, saved = load_adapter(path, 'cpu')
    replay = free.load_replay('original', 0, free.replay_entries('original', 0))
    expected, _ = fit_adapter(replay, free.goal, seed=0, **free.manifest['fitting'])
    assert all(torch.equal(actual.state_dict()[key], value) for key, value in expected.state_dict().items())
    assert saved['training']['initialization'] == 'identity'
    assert saved['training']['anchor_weight'] == 0
    assert saved['training']['anchoring_enabled'] is False
    helper = importlib.import_module('src.mylonas_ijepa_wm.curriculum_initial_fit')
    def cannot_refit(*args, **kwargs):
        raise AssertionError('completed initial fit must not repeat')
    monkeypatch.setattr(helper, 'fit_adapter', cannot_refit)
    visits = len(runtime.visited)
    free.initial('original', runtime)
    assert len(runtime.visited) == visits
    assert file_sha256(path) == sha


def test_both_no_anchor_curricula_retrain_initial_and_chain_all_updates(curriculum_sources, tmp_path):
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    before = [file_hashes(path) for path in curriculum_sources]
    baseline(free)
    for recipe in ('original', 'clean'):
        runtime = TinyRuntime(free, recipe)
        free.initial(recipe, runtime)
        previous = free.reference(recipe, 0)[1]
        for index in (1, 2, 3):
            status = free.band(recipe, index, runtime)
            assert status['inputs']['incoming_sha256'] == previous
            training = json.loads((runtime.root / f'updates/v{index}/training.json').read_text())
            assert training['warm_start'] and training['optimizer_reset']
            assert training['anchor_weight'] == 0 and training['anchoring_enabled'] is False
            assert all(row['loss'] == row['rank_loss'] and row['anchor_loss'] == 0 for row in training['history'])
            if recipe == 'clean':
                assert training['pair_filter']['min_goal_progress'] == .5
            previous = status['checkpoint_sha256']
    report = free.report()
    assert report['branches']['original']['status'] == report['branches']['clean']['status'] == 'complete'
    assert report['total_primitive_interactions'] == 3360
    assert [file_hashes(path) for path in curriculum_sources] == before


@pytest.mark.parametrize('artifact', ['checkpoints/v0.pt', 'updates/v0/training.json', 'inputs'])
def test_no_anchor_initial_corruption_blocks_new_actions(curriculum_sources, tmp_path, artifact):
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    baseline(free)
    runtime = TinyRuntime(free, 'clean')
    free.initial('clean', runtime)
    if artifact == 'inputs':
        path = runtime.root / 'updates/v0/status.json'
        status = json.loads(path.read_text()); status['inputs']['seed'] += 1
        write_json(path, status)
    else:
        path = runtime.root / artifact
        path.write_bytes(path.read_bytes() + b'changed')
    before = len(runtime.visited)
    with pytest.raises(ValueError, match='initial|hash|provenance'):
        free.band('clean', 1, runtime)
    assert len(runtime.visited) == before


def test_partial_no_anchor_initial_fit_is_not_reported_as_evaluated(curriculum_sources, tmp_path):
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    baseline(free)
    runtime = TinyRuntime(free, 'original')
    free.attach(runtime, 'original')
    write_json(runtime.root / 'updates/v0/status.json', {'status': 'training', 'inputs': {}})
    report = free.report()
    assert report['branches']['original']['versions']['0']['near'] is None
    with pytest.raises(ValueError, match='initial|inputs'):
        free.initial('original', runtime)
    assert not runtime.visited


def test_historical_comparison_is_read_only_and_marks_missing_versions(curriculum_sources, tmp_path):
    anchored = prepared(curriculum_sources, tmp_path)
    baseline(anchored)
    anchored.initial('original', TinyRuntime(anchored, 'original'))
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    baseline(free)
    free.initial('original', TinyRuntime(free, 'original'))
    before = file_hashes(anchored.root)
    name = 'src.mylonas_ijepa_wm.curriculum_ablation_report'
    assert importlib.util.find_spec(name), 'historical comparison helper missing'
    helper = importlib.import_module(name)
    report = helper.compare_historical(free, anchored.root)
    cell = report['branches']['original']['versions']['0']['near']
    assert cell['anchored']['successes'] == cell['without_anchor']['successes'] == 20
    assert cell['paired']['wins'] == cell['paired']['losses'] == 0
    assert report['branches']['original']['versions']['1']['band_1'] is None
    assert report['branches']['clean']['versions']['0']['near'] is None
    assert file_hashes(anchored.root) == before
    with pytest.raises(ValueError, match='cases|settings'):
        free.manifest['evaluation']['near'][0]['seed'] += 1
        helper.compare_historical(free, anchored.root)


def test_historical_comparison_rejects_output_aliases(curriculum_sources, tmp_path):
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    name = 'src.mylonas_ijepa_wm.curriculum_ablation_report'
    assert importlib.util.find_spec(name), 'historical comparison helper missing'
    helper = importlib.import_module(name)
    alias = tmp_path / 'baseline-alias'; alias.symlink_to(free.root, target_is_directory=True)
    for path in (free.root, alias, free.root / 'nested'):
        with pytest.raises(ValueError, match='separate'):
            helper.compare_historical(free, path)


@pytest.mark.parametrize('damage', ['checkpoint', 'incomplete_status', 'map_binding'])
def test_diagnostic_maps_omit_rejected_initial_versions_and_keep_valid_branch(curriculum_sources, tmp_path, damage):
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    baseline(free)
    for recipe in ('original', 'clean'):
        free.initial(recipe, TinyRuntime(free, recipe))
    root = free.root / 'branches/original'
    if damage == 'checkpoint':
        path = root / 'checkpoints/v0.pt'
        path.write_bytes(path.read_bytes() + b'changed')
    elif damage == 'incomplete_status':
        path = root / 'updates/v0/status.json'
        status = json.loads(path.read_text()); status['status'] = 'training'
        write_json(path, status)
    else:
        path = root / 'heatmaps/v0.json'
        data = json.loads(path.read_text()); data['checkpoint_sha256'] = 'wrong'
        write_json(path, data)
        path = root / 'heatmaps/v0.commit.json'
        commit = json.loads(path.read_text())
        commit['inputs']['checkpoint_sha256'] = 'wrong'
        commit['data_sha256'] = file_sha256(root / 'heatmaps/v0.json')
        write_json(path, commit)
    helper = importlib.import_module('src.mylonas_ijepa_wm.curriculum_ablation_report')
    selection = helper.validated_heatmaps(free)
    assert selection['maps']['original'] == {}
    assert list(selection['maps']['clean']) == [0]
    assert selection['maps']['clean'][0]['recipe'] == 'clean'


def test_historical_comparison_rejects_cross_run_precision_difference(curriculum_sources, tmp_path):
    class BfloatRuntime(TinyRuntime):
        def __init__(self, curriculum, recipe):
            super().__init__(curriculum, recipe)
            path = self.root / 'runtime.json'
            saved = json.loads(path.read_text())
            saved['signature']['precision'] = self.precision = 'bf16'
            saved['runtime_id'] = self.runtime_id = digest(saved['signature'])
            write_json(path, saved)
    anchored = prepared(curriculum_sources, tmp_path)
    raw = BfloatRuntime(anchored, 'raw')
    anchored.grid(raw); anchored.baseline(raw)
    anchored.initial('original', BfloatRuntime(anchored, 'original'))
    free = no_anchor_prepared(curriculum_sources, tmp_path)
    baseline(free)
    free.initial('original', TinyRuntime(free, 'original'))
    helper = importlib.import_module('src.mylonas_ijepa_wm.curriculum_ablation_report')
    with pytest.raises(ValueError, match='execution|precision'):
        helper.compare_historical(free, anchored.root)
