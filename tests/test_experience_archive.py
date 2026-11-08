"""Fit from a completed raw-only band without executing its cases again."""
from types import SimpleNamespace

import torch

from src.mylonas_ijepa_wm import experience_curriculum as exp
from src.mylonas_ijepa_wm.curriculum_protocol import digest, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, save_tensors


def test_archived_band_one_is_read_only_and_keeps_the_original_goal_tokens(tmp_path):
    source, output = tmp_path/"source", tmp_path/"new"
    case = dict(case_id="b1_g00_s00", goal=[50, 42], goal_index=0)
    manifest = dict(stages=[[case]], goal_ids=[0], pair_cap=8)
    config = dict(manifest=manifest)
    source_config = dict(config, purpose="raw_band_1")
    common = dict(models={}, dino_commit="pin", dino_source="hash", precision="fp32", encoder_batch_size=8)
    sig = dict(common, manifest=digest(manifest), runtime_variant=dict(experiment_digest=digest(source_config)))
    source_id = digest(sig)
    write_json(source/"experiment.json", source_config)
    write_json(source/"runtime.json", dict(runtime_id=source_id, signature=sig))
    targets = torch.zeros(1, 1, 2)
    save_tensors(source/"collection_goals.pt", dict(runtime_id=source_id, tokens=targets))
    save_tensors(source/"raw/episode.pt", dict(case=case, runtime_id=source_id, success=True,
        states=torch.tensor([[50, 36], [50, 37], [50, 38]]), primitive_steps=torch.arange(3),
        frames=torch.tensor([[4, 0], [3, 0], [2, 0]])))
    run_stage(source, "band_1", [case], source_id, lambda _: dict(success=True, micro_steps=2,
        raw_path="raw/episode.pt", raw_sha256=file_sha256(source/"raw/episode.pt")),
        condition=dict(collector="raw", goal_sha256=file_sha256(source/"collection_goals.pt")))
    before = {p.relative_to(source):file_sha256(p) for p in source.rglob("*") if p.is_file()}
    archive = exp._read_raw_band1(source, config)
    runtime = SimpleNamespace(root=output, runtime_id="current", device="cpu",
                              encode_frames=lambda frames:frames[:, None].float())
    write_json(output/"runtime.json", dict(signature=common))
    design = dict(points=[[50, 42]], goal_ids=[0], heldout_mask=[False], frame_sha256=["goal"])
    report = exp.collect_and_fit(runtime, manifest, design, dict(updates=1, batch_size=2, bottleneck=2),
                                 .1, 0, filter_steps=exp._temporal_filter, raw_band1=archive)
    assert report["retained_pairs"] == 2
    assert report["source_runtime_id"] == source_id
    saved = torch.load(output/"collection_goals.pt", weights_only=True)
    assert saved["runtime_id"] == "current" and torch.equal(saved["tokens"], targets)
    assert not (output/"raw").exists()
    assert before == {p.relative_to(source):file_sha256(p) for p in source.rglob("*") if p.is_file()}
