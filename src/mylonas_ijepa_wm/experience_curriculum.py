"""Fit a multi-goal ranking cost from real planner rollouts in expanding annuli."""
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING, digest, load_stage, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256, save_tensors
from src.mylonas_ijepa_wm.multigoal_ranking import GAP, fit, ordinal_rows
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash, runtime_variant
from src.mylonas_ijepa_wm.ranking_planning_protocol import in_region, load_ranking_source


BANDS = ((6, 12), (12, 20), (20, 28), (28, 34))
GOAL_ANCHORS = ((50, 42), (44, 40), (53, 48), (42, 20), (50, 12), (55, 14))
SEED = 20261107
PAIR_CAP = 8


def _goal_ids(design):
    points = np.asarray(design["points"])
    candidates = np.asarray(design["train_ids"])
    candidate_xy = points[candidates]
    candidates = candidates[(candidate_xy >= [38, 8]).all(axis=1) & (candidate_xy <= [56, 55]).all(axis=1)]
    if len(candidates) < len(GOAL_ANCHORS):
        raise ValueError("not enough reset-safe handcrafted training references")
    goals = []
    for anchor in GOAL_ANCHORS:
        order = candidates[np.argsort(np.linalg.norm(points[candidates] - anchor, axis=1), kind="stable")]
        goals.append(next(int(i) for i in order if i not in goals))
    return goals


def make_manifest(design, starts_per_goal=6):
    if starts_per_goal < 1:
        raise ValueError("need at least one start per goal and band")
    points = np.asarray(design["points"])
    goals = _goal_ids(design)
    rng = np.random.default_rng(SEED)
    stages, used = [], []
    for band, (lo, hi) in enumerate(BANDS):
        cases = []
        for gi, goal_id in enumerate(goals):
            goal = points[goal_id]
            for j in range(starts_per_goal):
                for _ in range(50000):
                    angle = rng.uniform(0, 2 * np.pi)
                    start = goal + rng.uniform(lo, hi) * np.array([np.cos(angle), np.sin(angle)])
                    if not (in_region(start) and 38 <= start[0] <= 56 and 8 <= start[1] <= 55):
                        continue
                    if used and np.linalg.norm(np.asarray(used) - start, axis=1).min() < .25:
                        continue
                    break
                else:
                    raise ValueError(f"cannot sample starts for goal {goal.tolist()} in band {band + 1}")
                used.append(start.tolist())
                cases.append(dict(case_id=f"b{band + 1}_g{gi:02d}_s{j:02d}", split="train",
                                  band=band + 1, goal_index=gi, goal=goal.tolist(),
                                  start=start.tolist(), layout=[32, 30], run_seed=SEED,
                                  seed=int(rng.integers(0, 2**31 - 1))))
        stages.append(cases)
    return dict(bands=[list(band) for band in BANDS], goal_ids=goals, stages=stages,
                starts_per_goal=starts_per_goal, planning=dict(PLANNING), seed=SEED, pair_cap=PAIR_CAP)


def _protected(design):
    ids = set(design["goal_ids"][1:]) | set(np.flatnonzero(design["heldout_mask"]))
    return (np.asarray([design["points"][i] for i in sorted(ids)]).reshape(-1, 2),
            {design["frame_sha256"][i] for i in ids})


def _observations(raw):
    xy = raw["states"].numpy()[:, :2]
    steps = raw["primitive_steps"].numpy()
    if (len(xy) != len(raw["frames"]) or not np.array_equal(steps, np.arange(len(xy)))
            or not np.isfinite(xy).all()):
        raise ValueError("raw observations must be finite and consecutive")
    return xy


def _select_steps(xy, frames, candidates, inside, case, design, cap):
    points, hashes = _protected(design)
    if len(points):
        inside &= (np.linalg.norm(xy[:, None, :] - points[None, :, :], axis=-1) > 1e-5).all(axis=1)
    hashes.add(design["frame_sha256"][case["goal_id"]])
    eligible = []
    for i in candidates:
        if inside[i] and inside[i + 1]:
            before, after = (_frame_hash(frames[j]) for j in (i, i + 1))
            if before != after and not ({before, after} & hashes):
                eligible.append(int(i))
    selected = [eligible[k] for k in np.linspace(0, len(eligible) - 1, min(cap, len(eligible)), dtype=int)]
    return selected, dict(steps=len(xy) - 1, eligible_pairs=len(eligible), retained_pairs=len(selected),
                          stationary_steps=int((np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-5).sum()))


def _filter(raw, case, design, cap, *, geometry=None):
    xy = _observations(raw)
    if geometry is None:
        dist = np.linalg.norm(xy - np.asarray(case["goal"]), axis=1)
        inside = in_region(xy) & (dist <= BANDS[-1][1] + 1e-5)
    else:
        dist, inside = geometry(xy, case["goal"])
    gains = dist[:-1] - dist[1:]
    progress = np.flatnonzero(gains >= GAP)
    selected, stats = _select_steps(xy, raw["frames"], progress, inside, case, design, cap)
    stats.update(progress_steps=len(progress), away_steps=int((gains < -1e-5).sum()))
    return selected, stats


def _temporal_filter(raw, case, design, cap, *, domain=None):
    xy = _observations(raw)
    if not raw["success"]:
        return [], dict(steps=len(xy) - 1, eligible_pairs=0, retained_pairs=0, skipped_failed=True,
                        stationary_steps=int((np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-5).sum()))
    # Coordinates preserve the earlier experiment's region. Frame order supplies the label.
    if domain is None:
        inside = in_region(xy) & (np.linalg.norm(xy - np.asarray(case["goal"]), axis=1) <= BANDS[-1][1] + 1e-5)
    else:
        inside = domain(xy)
    selected, stats = _select_steps(xy, raw["frames"], range(len(xy) - 1), inside, case, design, cap)
    return selected, dict(stats, skipped_failed=False)


def _encode_positions(runtime, design, ids):
    runtime.env.reset(0, layout=[32, 30], start=[44, 34], goal=[50, 42])
    parts = []
    for offset in range(0, len(ids), 8):
        selected = ids[offset:offset + 8]
        frames = np.stack([runtime.env._render_at(design["points"][i]) for i in selected])
        if any(_frame_hash(frame) != design["frame_sha256"][i] for i, frame in zip(selected, frames)):
            raise ValueError("rendered reference or grid pixels differ from the saved design")
        parts.append(runtime.encode_frames(frames))
    return torch.cat(parts)


def _replay(runtime, record, design, manifest, *, filter_steps=_filter, raw_root=None):
    case = record["case"]
    root = runtime.root
    path = root / "replay" / f"{case['case_id']}.pt"
    lineage = dict(runtime_id=runtime.runtime_id, case=case, raw_sha256=record["raw_sha256"])
    source_runtime_id = runtime.runtime_id if raw_root is None else record["runtime_id"]
    if raw_root is not None:
        lineage["source_runtime_id"] = source_runtime_id
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["inputs"] != lineage:
            raise ValueError(f"replay provenance differs for {case['case_id']}")
        return saved
    source = root if raw_root is None else raw_root
    raw_path = (source / record["raw_path"]).resolve()
    raw_path.relative_to(source.resolve())
    if file_sha256(raw_path) != record["raw_sha256"]:
        raise ValueError(f"raw observations changed for {case['case_id']}")
    raw = torch.load(raw_path, map_location="cpu", weights_only=True)
    if (raw["case"] != case or raw["runtime_id"] != source_runtime_id
            or (raw_root is not None and raw["success"] != record["success"])
            or len(raw["frames"]) != record["micro_steps"] + 1):
        raise ValueError("raw observations do not match the completed rollout")
    starts, stats = filter_steps(raw, dict(case, goal_id=manifest["goal_ids"][case["goal_index"]]),
                           design, manifest["pair_cap"])
    ids = sorted({j for i in starts for j in (i, i + 1)})
    local = {i: j for j, i in enumerate(ids)}
    payload = dict(inputs=lineage, stats=stats, positions=raw["states"][ids, :2],
                   pairs=[[local[i], local[i + 1]] for i in starts],
                   tokens=runtime.encode_frames(raw["frames"][ids]).half() if ids else None)
    save_tensors(path, payload)
    return payload


def _checkpoint(runtime, index):
    root = runtime.root / f"band_{index + 1}"
    report = read(root / "complete.json")
    if report["runtime_id"] != runtime.runtime_id or report["checkpoint_sha256"] != file_sha256(root / "adapter.pt"):
        raise ValueError(f"checkpoint provenance differs in band {index + 1}")
    adapter, _ = load_adapter(root / "adapter.pt", runtime.device, expected_runtime_id=runtime.runtime_id)
    return adapter, report


def _raw_cost(target):
    @torch.no_grad()
    def score(tokens):
        return (tokens.float() - target).square().mean(dim=(1, 2))

    return score


def _initial_scale(tokens, population):
    endpoints = sorted({(g, i) for g, a, b in population for i in (a, b)})
    values = []
    for batch in torch.as_tensor(endpoints).split(8):
        ref, point = batch.T
        values.append((tokens[ref].float() - tokens[point].float()).square().mean(dim=(1, 2)))
    values = torch.cat(values)
    positive = values[values > 0]
    if not len(positive):
        raise ValueError("first band has no positive raw goal cost")
    return float(positive.median())


def _collect_band(runtime, manifest, design, index, initial, prior, encode_positions):
    root = runtime.root
    band = f"band_{index + 1}"
    goal_path = root / "collection_goals.pt"
    if not goal_path.exists():
        save_tensors(goal_path, dict(runtime_id=runtime.runtime_id,
                                    tokens=encode_positions(runtime, design, manifest["goal_ids"])))
    payload = torch.load(goal_path, map_location="cpu", weights_only=True)
    if payload["runtime_id"] != runtime.runtime_id:
        raise ValueError("collection goal tokens come from a different runtime")
    goals = payload["tokens"]
    targets = goals.to(runtime.device)
    scorers = [adapter_goal_cost(initial, t) if initial is not None else _raw_cost(t) for t in targets]
    condition = dict(goal_sha256=file_sha256(goal_path), collector=prior["checkpoint_sha256"] if prior else "raw")

    def execute(case):
        record = runtime.run_case(case, capture=True, latent_cost_fn=scorers[case["goal_index"]], journal_stage=band)
        record["raw_sha256"] = file_sha256(root / record["raw_path"])
        return record

    records = run_stage(root, band, manifest["stages"][index], runtime.runtime_id, execute, condition=condition)
    return records, goals, condition


def _read_raw_band1(root, config):
    manifest = config["manifest"]
    source_config = read(root / "experiment.json")
    source_manifest = source_config["manifest"]
    # Later expansion bands can change independently of the archived first band.
    expected_manifest = dict(manifest, stages=source_manifest["stages"], bands=source_manifest["bands"])
    if (source_config != dict(config, manifest=expected_manifest, purpose="raw_band_1")
            or manifest["stages"][0] != source_manifest["stages"][0]
            or manifest["bands"][0] != source_manifest["bands"][0]):
        raise ValueError("saved raw collection differs in its cases, planner settings or inputs")
    saved = read(root / "runtime.json")
    signature, source_id = saved["signature"], saved["runtime_id"]
    if (source_id != digest(signature) or signature["manifest"] != digest(source_manifest)
            or signature["runtime_variant"]["experiment_digest"] != digest(source_config)):
        raise ValueError("saved raw runtime provenance differs")
    stage = read(root / "stages" / "band_1.json")
    if stage["condition"]["collector"] != "raw":
        raise ValueError("saved Band 1 was not collected with raw MSE")
    records = load_stage(root, "band_1", manifest["stages"][0], source_id)
    provenance = dict(root=str(root), runtime_id=source_id, records_digest=digest(records),
                      goal_sha256=stage["condition"]["goal_sha256"])
    return dict(root=root, signature=signature, records=records, provenance=provenance)


def _reuse_band1(runtime, manifest, archive):
    source, provenance = archive["root"], archive["provenance"]
    current = read(runtime.root / "runtime.json")["signature"]
    for key in ("models", "dino_commit", "dino_source", "precision", "encoder_batch_size"):
        if current[key] != archive["signature"][key]:
            raise ValueError(f"saved raw collection runtime differs for {key}")
    original = source / "collection_goals.pt"
    if file_sha256(original) != provenance["goal_sha256"]:
        raise ValueError("saved raw collection goal checksum differs")
    targets = torch.load(original, map_location="cpu", weights_only=True)
    goals = targets["tokens"]
    if (targets["runtime_id"] != provenance["runtime_id"] or goals.ndim != 3
            or len(goals) != len(manifest["goal_ids"]) or not torch.isfinite(goals).all()):
        raise ValueError("saved raw collection goal tensors differ")
    goal_path = runtime.root / "collection_goals.pt"
    if goal_path.exists():
        saved = torch.load(goal_path, map_location="cpu", weights_only=True)
        if saved["runtime_id"] != runtime.runtime_id or not torch.equal(saved["tokens"], goals):
            raise ValueError("local collection goal tensors differ from the imported targets")
    else:
        save_tensors(goal_path, dict(runtime_id=runtime.runtime_id, tokens=goals))
    print(f"Reusing {len(archive['records'])} archived raw rollouts from {source}", flush=True)
    condition = dict(goal_sha256=file_sha256(goal_path), collector="raw", source_runtime_id=provenance["runtime_id"])
    return archive["records"], goals, condition


def collect_and_fit(runtime, manifest, design, fitting, scale, index, *,
                    filter_steps=_filter, encode_positions=_encode_positions, raw_band1=None):
    root = runtime.root
    band = f"band_{index + 1}"
    folder = root / band
    if (folder / "complete.json").exists():
        return _checkpoint(runtime, index)[1]
    initial, prior = _checkpoint(runtime, index - 1) if index else (None, None)
    raw_root = None
    if index == 0 and raw_band1 is not None:
        records, goals, condition = _reuse_band1(runtime, manifest, raw_band1)
        raw_root = raw_band1["root"]
    else:
        records, goals, condition = _collect_band(runtime, manifest, design, index, initial, prior, encode_positions)
    rows = []
    for record in records:
        saved = _replay(runtime, record, design, manifest, filter_steps=filter_steps, raw_root=raw_root)
        rows.append(dict(case=record["case"], success=record["success"],
                         **saved["stats"], retained_positions=saved["positions"].tolist()))

    new_pairs = sum(r["retained_pairs"] for r in rows)
    derive_scale = scale is None
    if derive_scale and prior is not None:
        scale = prior["scale"]
    if new_pairs:
        token_parts, population, offset = [goals.half()], [], len(goals)
        for cases in manifest["stages"][:index + 1]:
            for case in cases:
                saved = torch.load(root / "replay" / f"{case['case_id']}.pt", map_location="cpu", weights_only=True)
                if saved["pairs"]:
                    token_parts.append(saved["tokens"])
                    population.extend([case["goal_index"], offset + a, offset + b] for a, b in saved["pairs"])
                    offset += len(saved["tokens"])
        tokens = torch.cat(token_parts)
        del token_parts
        if scale is None:
            scale = _initial_scale(tokens, population)
        adapter, training = fit(tokens, population, scale, **fitting, device=runtime.device,
                                initial_adapter=initial, progress=lambda row: print(row, flush=True))
    elif initial is not None:
        adapter, training = initial, dict(updates=0, reason="no eligible pairs in this band")
    else:
        write_json(folder / "collection.json", rows)
        raise ValueError("first band yielded no eligible transitions; see band_1/collection.json")
    lineage = dict(runtime_id=runtime.runtime_id, band=index + 1, **condition)
    save_adapter(folder / "adapter.pt", adapter, training, lineage)
    write_json(folder / "training.json", training)
    report = dict(**lineage, collection=rows, retained_pairs=new_pairs, updates=training["updates"],
                  checkpoint_sha256=file_sha256(folder / "adapter.pt"))
    if derive_scale:
        report["scale"] = scale
    write_json(folder / "complete.json", report)
    print(f"{band}: {sum(r['success'] for r in rows)}/{len(rows)} successful, "
          f"{new_pairs} retained pairs, {training['updates']} optimizer updates", flush=True)
    return report


def compare(runtime, design, handcrafted, *, band_count=len(BANDS), full_half_map=False, filename="comparison.json"):
    points = design["points"]
    goal_ids = design["goal_ids"]
    nx, ny = len(design["grid"]["xs"]), len(design["grid"]["ys"])
    ids = np.asarray(design["grid"]["point_ids"]).reshape(ny, nx)
    edges = [*zip(ids[:, :-1].ravel(), ids[:, 1:].ravel()), *zip(ids[:-1].ravel(), ids[1:].ravel())]
    goals = [points[i] for i in goal_ids]
    tokens = _encode_positions(runtime, design, list(range(len(points))))
    raw = _scores(None, tokens, goal_ids, runtime.device)
    methods, legacy_methods, checkpoints = {}, {}, {}
    for name in ("raw", "handcrafted", *(f"band_{i + 1}" for i in range(band_count))):
        if name == "raw":
            costs = raw
        else:
            if name == "handcrafted":
                adapter = handcrafted
            else:
                adapter, checkpoints[name] = _checkpoint(runtime, int(name.split("_")[1]) - 1)
            costs = _scores(adapter, tokens, goal_ids, runtime.device)
        legacy = ordinal_rows(points, goals, raw, costs, design["heldout_mask"], edges)
        if full_half_map:
            legacy_methods[name] = legacy
            methods[name] = ordinal_rows(points, goals, raw, costs, design["heldout_mask"], edges, radius=np.inf)
        else:
            methods[name] = legacy
    report = dict(methods=methods, checkpoints=checkpoints, goal_ids=goal_ids,
                  grid=design["grid"], heldout_mask=design["heldout_mask"])
    if full_half_map:
        report["legacy_methods"] = legacy_methods
    write_json(runtime.root / filename, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "compare", "collect-raw"))
    for field in ("source-run", "output-dir", "world-model-checkpoint", "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--band", type=int)
    parser.add_argument("--band1-source", type=Path, help="Reuse the completed rollouts of a matching raw Band 1 run.")
    parser.add_argument("--starts-per-goal", type=int, default=6)
    planner_lengths = range(1, PLANNING["planning_horizon"] + 1)
    parser.add_argument("--actions-per-plan", type=int, choices=planner_lengths)
    parser.add_argument("--planning-horizon", type=int, choices=planner_lengths)
    parser.add_argument("--labels", choices=("progress", "temporal"), default="progress")
    parser.add_argument("--sampling", choices=("radial", "half-square"), default="radial")
    parser.add_argument("--wide-square-bands", action="store_true", help="Use 6-unit square strips to half-width 48.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    half_square = args.sampling == "half-square"
    if args.wide_square_bands and not half_square:
        parser.error("wide-square-bands requires half-square sampling")
    if half_square:
        if args.labels != "temporal":
            parser.error("half-square sampling requires temporal labels")
        from src.mylonas_ijepa_wm import experience_half_square
        bands = experience_half_square.WIDE_BANDS if args.wide_square_bands else experience_half_square.BANDS
        build_manifest = partial(experience_half_square.make_manifest, bands=bands)
        label_filter = experience_half_square.filter_steps
    else:
        bands = BANDS
        build_manifest = make_manifest
        label_filter = _temporal_filter if args.labels == "temporal" else _filter
    if args.operation == "stage" and args.band not in range(1, len(bands) + 1):
        parser.error(f"stage requires --band between 1 and {len(bands)}")
    if args.operation == "compare" and args.band is not None and args.band not in range(1, len(bands) + 1):
        parser.error(f"compare --band must be between 1 and {len(bands)}")
    if args.operation == "collect-raw" and (args.band not in (None, 1) or args.band1_source is not None):
        parser.error("collect-raw generates its own Band 1 rollouts")
    if (args.actions_per_plan or PLANNING["actions_per_plan"]) > (args.planning_horizon or PLANNING["planning_horizon"]):
        parser.error("actions-per-plan cannot exceed planning-horizon")
    args.output_dir, source = locations(args.output_dir, args.source_run,
                                      (args.world_model_checkpoint, args.encoder_checkpoint, args.dino_wm_root))
    handcrafted, snapshot, design = load_ranking_source(source)
    source_config = read(source / "config.json")
    fitting = dict(source_config["fitting"], updates=500)
    if snapshot["fitting_signature"]["encoder_sha256"] != EXPECTED_ENCODER:
        raise ValueError("encoder differs from handcrafted benchmark")
    manifest = build_manifest(design, args.starts_per_goal)
    if args.actions_per_plan is not None:
        manifest["planning"]["actions_per_plan"] = args.actions_per_plan
    if args.planning_horizon is not None:
        manifest["planning"]["planning_horizon"] = args.planning_horizon
    if half_square:
        kind = "experience_temporal_half_square_v1"
    else:
        kind = "experience_temporal_v1" if args.labels == "temporal" else "experience_curriculum_v1"
    config = dict(kind=kind, manifest=manifest, handcrafted=snapshot,
                  fitting=fitting, scale=source_config["scale"])
    raw_band1 = None
    if args.operation == "collect-raw":
        config["purpose"] = "raw_band_1"
    elif args.band1_source is not None:
        _, raw_source = locations(args.output_dir, args.band1_source)
        raw_band1 = _read_raw_band1(raw_source, config)
        config["band1_source"] = raw_band1["provenance"]
    path = args.output_dir / "experiment.json"
    if path.exists() and read(path) != config:
        raise ValueError("experiment configuration changed; choose a new output directory")
    write_json(path, config)
    args.precision = "fp32"
    args.encoder_batch_size = 8
    args.run_seed = SEED
    args.videos = False
    args.expected_world_model_sha256 = EXPECTED_WM
    args.expected_encoder_sha256 = EXPECTED_ENCODER
    args.runtime_signature_extra = runtime_variant(config, "experience", args.device)
    runtime = PilotRuntime(args, manifest)
    try:
        if args.operation == "collect-raw":
            records, _, _ = _collect_band(runtime, manifest, design, 0, None, None, _encode_positions)
            print(f"Raw band 1: {sum(r['success'] for r in records)}/{len(records)} successful", flush=True)
            return records
        if args.operation == "stage":
            return collect_and_fit(runtime, manifest, design, fitting, config["scale"], args.band - 1,
                                   filter_steps=label_filter, raw_band1=raw_band1)
        filename = "comparison.json" if args.band is None else f"band_{args.band}_comparison.json"
        return compare(runtime, design, handcrafted, band_count=args.band or len(bands),
                       full_half_map=half_square, filename=filename)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
