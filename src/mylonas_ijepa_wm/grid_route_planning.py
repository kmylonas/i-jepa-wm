"""Compare raw and adapted visual costs on paired full-map cases."""
import argparse
from math import dist
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_square, geodesic_radial as wall
from src.mylonas_ijepa_wm import ranking_planning
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, save_tensors
from src.mylonas_ijepa_wm.geodesic_multigoal import TEST_GOALS


def make_cases():
    """Five starts in each room for both same-room and opposite-room goals."""
    groups = (
        ("same_room_left", 0, ((8.5, 10.5), (10.5, 49.5), (15.5, 19.5), (23.5, 52.5), (26.5, 35.5))),
        ("same_room_right", 4, ((38.5, 8.5), (47.5, 27.5), (54.5, 44.5), (55.5, 24.5), (43.5, 51.5))),
        ("left_to_right", 4, ((9.5, 8.5), (13.5, 24.5), (22.5, 29.5), (24.5, 49.5), (10.5, 54.5))),
        ("right_to_left", 0, ((55.5, 10.5), (51.5, 22.5), (42.5, 28.5), (43.5, 49.5), (54.5, 53.5))),
    )
    cases = []
    for route, first_goal, starts in groups:
        for i, start in enumerate(starts):
            goal_id = first_goal + i % 4
            goal = TEST_GOALS[goal_id]
            if not wall.valid(start) or dist(start, goal) < 4.5:
                raise ValueError(f"invalid evaluation start {start} for {goal}")
            index = len(cases)
            cases.append(dict(case_id=f"g{goal_id:02d}_s{index:02d}", goal_id=goal_id, route=route,
                              split="eval", start=list(start), goal=list(goal),
                              layout=list(wall.LAYOUT), seed=20261110 + index))
    return cases


def load_source(source, device):
    experiment = read(source / "experiment.json")
    runtime = read(source / "runtime.json")
    manifest = experiment["manifest"]
    if (experiment["kind"] != "experience_mpc_route_feedback_full_grid_starts_v1"
            or runtime["signature"]["manifest"] != digest(manifest)):
        raise ValueError("expected a matching full-map grid-start training run")
    band, number = len(manifest["bands"]), manifest["rounds_per_band"]
    stage = source / f"band_{band}_round_{number}"
    checkpoint = stage / "adapter.pt"
    complete = read(stage / "complete.json")
    checksum = file_sha256(checkpoint)
    if (complete["runtime_id"] != runtime["runtime_id"] or complete["band"] != band
            or complete["round"] != number or complete["checkpoint_sha256"] != checksum):
        raise ValueError("adapter checksum or final-round provenance differs from the saved run")
    adapter, saved = load_adapter(checkpoint, device, expected_runtime_id=runtime["runtime_id"])
    if saved["provenance"]["band"] != band or saved["provenance"]["round"] != number:
        raise ValueError("adapter metadata does not identify the final round")
    return adapter, dict(checkpoint=str(checkpoint), sha256=checksum, runtime_id=runtime["runtime_id"],
                         models=runtime["signature"]["models"]), manifest


def references(runtime, path, experiment):
    key = digest(experiment)
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["experiment"] != key:
            raise ValueError("evaluation goal references belong to a different experiment")
        return saved
    frames = []
    for goal_id, goal in enumerate(TEST_GOALS):
        case = next(case for case in runtime.manifest["evaluation"] if case["goal_id"] == goal_id)
        runtime.env.reset(case["seed"], layout=case["layout"], start=case["start"], goal=goal)
        frames.append(runtime.env.render_goal())
    saved = dict(experiment=key, tokens=runtime.encode_frames(np.stack(frames)),
                 frame_sha256=[ranking_planning._frame_hash(frame) for frame in frames])
    save_tensors(path, saved)
    return saved


def run(source, output, encoder, world_model, dino_root, device):
    source, output = source.resolve(), output.resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("evaluation output must be separate from the source run")
    adapter, origin, training = load_source(source, device)
    cases = make_cases()
    trained_starts = {tuple(case["start"]) for stage in training["stages"] for case in stage}
    if ({tuple(case["start"]) for case in cases} & trained_starts
            or set(TEST_GOALS) & {tuple(goal) for goal in training["goals"]}):
        raise ValueError("evaluation starts and goals must be held out from training")
    manifest = dict(evaluation=cases, planning=dict(training["planning"], max_macro_steps=20))
    experiment = dict(kind="grid_route_planning_v1", source=origin, manifest=manifest)
    config_path = output / "experiment.json"
    if config_path.exists() and read(config_path) != experiment:
        raise ValueError("evaluation output already belongs to a different experiment")
    write_json(config_path, experiment)
    reference_path = output / "goals.pt"
    paired = {}
    for condition in ("raw", "adapted"):
        root = output / condition
        args = SimpleNamespace(
            output_dir=root, encoder_checkpoint=encoder, world_model_checkpoint=world_model,
            dino_wm_root=dino_root, device=device, precision="fp32", encoder_batch_size=8,
            expected_encoder_sha256=origin["models"]["encoder"],
            expected_world_model_sha256=origin["models"]["world_model"],
            videos=True, run_seed=20261110,
            runtime_signature_extra=ranking_planning.runtime_variant(experiment, condition, device, True),
        )
        runtime = ranking_planning.PlanningRuntime(args, manifest)
        try:
            refs = references(runtime, reference_path, experiment)
            inputs = dict(experiment=digest(experiment), runtime_id=runtime.runtime_id,
                          reference_tokens_sha256=file_sha256(reference_path))
            records = ranking_planning._run_cases(root, cases, condition, runtime, refs,
                                                  adapter if condition == "adapted" else None, inputs)
            if any(not row.get("video_path") or not Path(row["video_path"]).is_file() for row in records):
                raise RuntimeError(f"missing {condition} rollout video")
            paired[condition] = records
        finally:
            runtime.close()
        del runtime
    result = []
    for raw, adapted in zip(paired["raw"], paired["adapted"], strict=True):
        case = raw["case"]
        if case != adapted["case"]:
            raise ValueError("raw and adapted rollouts used different evaluation cases")
        row = dict(case)
        for condition, record in (("raw", raw), ("adapted", adapted)):
            row.update({f"{condition}_{field}": record[field] for field in
                        ("success", "micro_steps", "path_length", "final_goal_distance", "video_path")})
            row[f"{condition}_crossed"] = case["route"] in experience_square.crossings(
                record["primitive_positions"])
        result.append(row)
    write_json(output / "comparison.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source-run", "output-dir", "encoder-checkpoint", "world-model-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    run(args.source_run, args.output_dir, args.encoder_checkpoint, args.world_model_checkpoint,
        args.dino_wm_root, args.device)


if __name__ == "__main__":
    main()
