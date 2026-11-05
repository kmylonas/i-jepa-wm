import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def pilot_module():
    name = "src.mylonas_ijepa_wm.curriculum_cost_pilot"
    assert importlib.util.find_spec(name), "pilot runner missing"
    return importlib.import_module(name)


def result(case, success=True):
    return dict(case=case, case_id=case["case_id"], success=success,
                final_goal_distance=3. if success else 12., micro_steps=5,
                macro_steps=1, mpc_updates=1, runtime_id="test")


def test_manifest_is_disjoint_seeded_and_contains_paired_cases():
    pilot = pilot_module()
    manifest = pilot.make_manifest()
    assert manifest == pilot.make_manifest()
    assert len(manifest["evaluation"]) == 40
    assert all(len(cases) == 40 for cases in manifest["training"].values())
    evaluation_starts = np.array([case["start"] for case in manifest["evaluation"]])
    ids = set()
    for cases in [manifest["evaluation"], *manifest["training"].values()]:
        for case in cases:
            assert case["case_id"] not in ids
            ids.add(case["case_id"])
            assert case["goal"] == [50., 42.]
            assert 40 <= case["start"][0] <= 54
            low, high = (30, 36) if case["band"] == "near" else (14, 24)
            assert low <= case["start"][1] <= high
            if case["split"] == "train":
                assert np.linalg.norm(evaluation_starts - case["start"], axis=1).min() >= 0.25


def test_case_resume_skips_completed_cases_and_rejects_changed_identity(tmp_path):
    pilot = pilot_module()
    cases = pilot.make_manifest(eval_per_band=1)["evaluation"]
    visited = []

    def execute(case):
        visited.append(case["case_id"])
        if len(visited) == 2:
            raise RuntimeError("interrupted")
        return result(case)

    with pytest.raises(RuntimeError, match="interrupted"):
        pilot.run_stage(tmp_path, "baseline", cases, "test", execute)
    resumed = pilot.run_stage(tmp_path, "baseline", cases, "test", lambda case: result(case, False))
    assert [item["success"] for item in resumed] == [True, False]
    with pytest.raises(ValueError, match="provenance"):
        pilot.run_stage(tmp_path, "baseline", cases, "changed", lambda case: result(case))
    with pytest.raises(ValueError, match="provenance"):
        pilot.run_stage(tmp_path, "baseline", cases, "test", lambda case: result(case), condition={"new": True})


def test_gate_and_report_use_paired_ids_and_separate_bands():
    pilot = pilot_module()
    cases = pilot.make_manifest(eval_per_band=5)["evaluation"]
    raw = [result(case, index in (0, 1, 2, 5)) for index, case in enumerate(cases)]
    assert pilot.competence_gate(raw)["passed"]
    adapted = [result(case, index in (0, 1, 2, 5, 6)) for index, case in enumerate(cases)]
    report = pilot.paired_report(raw, adapted)
    assert report["near"]["raw_successes"] == 3
    assert report["far"]["adapted_successes"] == 2
    assert report["far"]["wins"] == 1
    assert report["far"]["losses"] == 0
    assert report["screen_passed"]
    assert not pilot.competence_gate([result(case, False) for case in cases])["passed"]
    adapted[0]["case"] = dict(adapted[0]["case"], start=[1., 2.])
    with pytest.raises(ValueError, match="paired"):
        pilot.paired_report(raw, adapted)


def test_prepare_cli_is_repeatable_but_refuses_changed_protocol(tmp_path):
    pilot = pilot_module()
    argv = ["prepare", "--output-dir", str(tmp_path), "--eval-per-band", "2", "--train-per-run", "3"]
    pilot.main(argv)
    first = json.loads((tmp_path / "manifest.json").read_text())
    pilot.main(argv)
    assert json.loads((tmp_path / "manifest.json").read_text()) == first
    with pytest.raises(ValueError, match="manifest"):
        pilot.main(["prepare", "--output-dir", str(tmp_path)])


def test_training_stops_with_status_instead_of_checkpoint_on_no_success(tmp_path):
    pilot = pilot_module()
    manifest = pilot.make_manifest(eval_per_band=1, train_per_run=2, run_seeds=(0,))
    raw = [result(case, case["band"] == "near") for case in manifest["evaluation"]]
    pilot.run_stage(tmp_path, "baseline", manifest["evaluation"], "test", lambda case: raw[manifest["evaluation"].index(case)])
    cases = manifest["training"]["0"]
    pilot.run_stage(tmp_path, "run_0/collect", cases, "test", lambda case: result(case, False))
    with pytest.raises(ValueError, match="successful"):
        pilot.train_stage(tmp_path, manifest, 0, "test", goal=None, encode_episode=None, device="cpu")
    status = json.loads((tmp_path / "stages/run_0/train.json").read_text())
    assert status["status"] == "stopped"
    assert not (tmp_path / "run_0/adapter.pt").exists()


def test_candidate_audit_replays_complete_horizon_and_counts_extra_interactions(tmp_path):
    name = "src.mylonas_ijepa_wm.curriculum_diagnostics"
    assert importlib.util.find_spec(name), "pilot diagnostics missing"
    diagnostics = importlib.import_module(name)

    class Environment:
        def reset(self, seed, **kwargs):
            self.state = np.array([0., 0.], dtype=np.float32)

        def step(self, action):
            self.state += action
            return type("Outcome", (), {"success": self.state[0] >= 2, "state": self.state.copy()})()

        def render(self):
            return self.state.reshape(1, 1, 2).copy()

    class Runtime:
        env = Environment()
        device = torch.device("cpu")
        root = tmp_path
        runtime_id = "test"

        def goal_tokens(self):
            return torch.tensor([[2., 0.]])

        def encode_frames(self, frames):
            return torch.as_tensor(np.stack(frames)).reshape(-1, 1, 2)

    candidates = [dict(source="final_elite", predicted_cost=0., actions=np.array([[1., 0.] * 5]))]
    rows, interactions = diagnostics.replay_audit(Runtime(), dict(seed=3, case_id="eval_0", layout=[32, 30], start=[0., 0.], goal=[2., 0.]), candidates)
    assert interactions == 5
    assert rows[0]["reached_goal"]
    assert rows[0]["actual_distance"] == 3
    assert rows[0]["true_cost"] == 4.5


def test_synthetic_stages_fit_restore_and_report_without_using_eval_replay(tmp_path):
    pilot = pilot_module()
    manifest = pilot.make_manifest(eval_per_band=2, train_per_run=2, run_seeds=(0,))
    manifest["fitting"].update(updates=3, batch_size=2, bottleneck=3)
    (tmp_path / "runtime.json").write_text(json.dumps({"runtime_id": "test", "signature": {"manifest": pilot.digest(manifest)}}))
    cases = manifest["evaluation"]
    pilot.run_stage(tmp_path, "baseline", cases, "test", lambda case: result(case, case["band"] == "near"))
    pilot.run_stage(tmp_path, "run_0/collect", manifest["training"]["0"], "test", result)
    encoded_ids = []

    def encode(record):
        encoded_ids.append(record["case_id"])
        return dict(case_id=record["case_id"], split="train", success=True, runtime_id="test",
                    latents=torch.tensor([[[1., 0., 0., 0.]], [[0., 2., 0., 0.]], [[0., 0., 3., 0.]]]),
                    states=torch.tensor([[40., 30.], [44., 34.], [48., 38.]]), primitive_steps=torch.tensor([0, 2, 4]))

    path = pilot.train_stage(tmp_path, manifest, 0, "test", goal=torch.zeros(1, 4), encode_episode=encode, device="cpu")
    assert encoded_ids == [case["case_id"] for case in manifest["training"]["0"]]
    first_mtime = path.stat().st_mtime_ns
    (tmp_path / "stages/run_0/train.json").unlink()
    (tmp_path / "run_0/training.json").unlink()
    pilot.train_stage(tmp_path, manifest, 0, "test", goal=None, encode_episode=None, device="cpu")
    assert path.stat().st_mtime_ns == first_mtime
    assert json.loads((tmp_path / "stages/run_0/train.json").read_text())["status"] == "complete"
    assert (tmp_path / "run_0/training.json").is_file()
    condition = dict(cost="latent", adapter_sha256=pilot.file_sha256(path))
    pilot.run_stage(tmp_path, "run_0/evaluate", cases, "test", result, condition=condition)
    report = pilot.make_report(tmp_path, manifest)
    assert report["runs"]["0"]["far"]["wins"] == 2
    assert report["runs"]["0"]["collection"]["attempts"] == 2


def test_interrupted_actions_remain_in_interaction_budget(tmp_path):
    from src.mylonas_ijepa_wm import curriculum_runtime as runtime
    assert hasattr(runtime, "InteractionJournal"), "durable interaction journal missing"
    with pytest.raises(RuntimeError):
        with runtime.InteractionJournal(tmp_path, "test", "run_0/collect", "train_0") as journal:
            journal.record(1, [1., 0.])
            journal.record(2, [1., 0.])
            raise RuntimeError("lost session")
    with runtime.InteractionJournal(tmp_path, "test", "run_0/collect", "train_0") as journal:
        journal.record(1, [1., 0.])
    budget = runtime.interaction_budget(tmp_path, "test")
    assert budget["primitive_interactions"] == 3
    assert budget["by_stage"]["run_0/collect"] == 3
    assert budget["interrupted_attempts"] == 1


def test_source_archive_keeps_original_revision_without_git_metadata(tmp_path):
    from src.mylonas_ijepa_wm import curriculum_runtime as runtime
    assert hasattr(runtime, "repository_provenance"), "source archive provenance missing"
    source = tmp_path / "src"
    source.mkdir()
    (source / "example.py").write_text("value = 1\n")
    (tmp_path / "pilot_source.json").write_text(json.dumps({"commit": "abc123", "source_digest": runtime.source_digest(source)}))
    metadata = runtime.repository_provenance(tmp_path)
    assert metadata == {"source_commit": "abc123", "source_dirty": False, "source_kind": "archive"}
    (source / "example.py").write_text("value = 2\n")
    assert runtime.repository_provenance(tmp_path)["source_dirty"]


def test_retention_screen_counts_lost_tasks_even_when_other_near_tasks_improve():
    pilot = pilot_module()
    cases = pilot.make_manifest(eval_per_band=10)["evaluation"]
    raw = [result(case, index < 6) for index, case in enumerate(cases)]
    adapted = [result(case, index in range(2, 9) or index in (10, 11)) for index, case in enumerate(cases)]
    report = pilot.paired_report(raw, adapted)
    assert report["near"]["success_change"] == 1
    assert report["near"]["losses"] == 2
    assert not report["screen_passed"]


def test_interrupted_diagnostic_replays_are_durably_counted(tmp_path):
    from src.mylonas_ijepa_wm import curriculum_diagnostics as diagnostics
    from src.mylonas_ijepa_wm.curriculum_runtime import interaction_budget

    class Environment:
        fail = True

        def reset(self, *args, **kwargs):
            self.calls = 0

        def step(self, action):
            self.calls += 1
            if self.fail and self.calls == 3:
                raise RuntimeError("interrupted simulator audit")
            return SimpleNamespace(state=np.array([self.calls, 0.]), success=False)

        def render(self):
            return np.zeros((1, 1, 2))

    runtime = SimpleNamespace(root=tmp_path, runtime_id="test", env=Environment(), device=torch.device("cpu"),
                              goal_tokens=lambda: torch.zeros(1, 2), encode_frames=lambda frames: torch.zeros(len(frames), 1, 2))
    case = dict(seed=3, case_id="eval_near_0", layout=[32, 30], start=[0, 0], goal=[2, 0])
    candidates = [dict(source="final_elite", predicted_cost=0., actions=np.array([[1., 0.] * 5]))]
    with pytest.raises(RuntimeError, match="interrupted"):
        diagnostics.replay_audit(runtime, case, candidates)
    runtime.env.fail = False
    diagnostics.replay_audit(runtime, case, candidates)
    assert interaction_budget(tmp_path, "test")["primitive_interactions"] == 7


@pytest.mark.skipif(not (torch.cuda.is_available() or torch.backends.mps.is_available()), reason="GPU regression")
def test_diagnostics_accept_cpu_checkpoint_statistics_on_gpu(tmp_path):
    from src.mylonas_ijepa_wm import curriculum_diagnostics as diagnostics
    from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter
    from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    class Encoder(torch.nn.Module):
        def forward(self, images):
            return torch.zeros(len(images), 1, 2, device=images.device)

    class Model(torch.nn.Module):
        def forward(self, latents, states, actions):
            predicted = states + actions.reshape(*actions.shape[:-1], 5, 2).sum(dim=-2)
            return predicted.unsqueeze(-2), predicted

    class Environment:
        def _render_at(self, position):
            return np.zeros((224, 224, 3), np.uint8)

        def reset(self, seed, **kwargs):
            self.state = np.asarray(kwargs["start"], dtype=np.float32)
            return SimpleNamespace(state=self.state.copy(), frame=self._render_at(None))

        def step(self, action):
            self.state += action
            return SimpleNamespace(state=self.state.copy(), success=False)

        def render(self):
            return self._render_at(None)

    planning = dict(PLANNING, planning_horizon=1, population=4, num_elites=2, cem_iterations=1, candidate_batch_size=2)
    manifest = pilot_module().make_manifest(eval_per_band=1)
    manifest["planning"] = planning
    (tmp_path / "run_0").mkdir()
    (tmp_path / "run_0/adapter.pt").write_bytes(b"audit-identity")
    runtime = SimpleNamespace(
        root=tmp_path, runtime_id="test", device=device, precision="fp32", env=Environment(),
        encoder=Encoder().to(device), model=Model().to(device), manifest=manifest,
        metadata=dict(state_mean=torch.zeros(2), state_std=torch.ones(2)),
        goal_tokens=lambda: torch.zeros(1, 2, device=device),
        encode_frames=lambda frames: torch.zeros(len(frames), 1, 2), load_world_model=lambda: None,
    )
    metric = ResidualTokenAdapter(2, 2).to(device).eval().requires_grad_(False)
    report = diagnostics.run_diagnostics(runtime, metric, 0, sample_count=1)
    assert report["extra_diagnostic_primitive_interactions"] == 60


def test_adapted_videos_have_distinct_paths_for_shared_eval_cases(tmp_path, monkeypatch):
    from src.mylonas_ijepa_wm import curriculum_runtime as runtime_module
    from src.mylonas_ijepa_wm.cost_adapter import ResidualTokenAdapter

    frame = np.zeros((224, 224, 3), np.uint8)
    case = pilot_module().make_manifest(eval_per_band=1)["evaluation"][0]
    args = SimpleNamespace(videos=True, run_seed=0)
    runtime = SimpleNamespace(
        root=tmp_path, runtime_id="test", device=torch.device("cpu"), precision="fp32", args=args,
        manifest={"planning": {}}, config=None, encoder=None, model=None, metadata=None,
        load_encoder=lambda: None, load_world_model=lambda: None, goal_tokens=lambda: torch.zeros(1, 2),
        env=SimpleNamespace(reset=lambda *a, **kw: SimpleNamespace(state=np.array(kw["start"]), frame=frame, success=False)),
    )

    def episode(*positional, **kwargs):
        positional[-1].append(frame)
        return dict(success=True, micro_steps=1, primitive_actions=[[1., 0.]], macro_durations=[1])

    monkeypatch.setattr(runtime_module.plan_proprio, "run_episode", episode)
    monkeypatch.setattr(runtime_module.visual, "save_episode_video", lambda frames, directory, *rest: directory / "same-case.mp4")
    metric = ResidualTokenAdapter(2, 2)
    first = runtime_module.PilotRuntime.run_case(runtime, case, metric=metric)
    args.run_seed = 1
    second = runtime_module.PilotRuntime.run_case(runtime, case, metric=metric)
    assert first["video_path"] != second["video_path"]
