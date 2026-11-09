"""Refit one archived round with capped failure feedback, then replay fixed starts."""

import argparse
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_curriculum as experience
from src.mylonas_ijepa_wm import experience_exploration, experience_square
from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256
from src.mylonas_ijepa_wm.multigoal_ranking import fit
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores
from src.mylonas_ijepa_wm.ranking_planning import runtime_variant


def capped_schedule(schedule, triples, failed, cap, *, seed):
    """Replace only failure draws exceeding the allowed slots in a minibatch."""
    draws = np.asarray(schedule, dtype=int).copy()
    refs = np.asarray(triples)[:, 0]
    successes = np.flatnonzero(~failed)
    if not len(successes):
        raise ValueError("failure cap requires successful replay pairs")
    by_goal = {g: successes[refs[successes] == g] for g in np.unique(refs[successes])}
    rng = np.random.default_rng([seed, 917])
    slots = Fraction(str(cap)) * draws.shape[1]
    for update, batch in enumerate(draws):
        allowed = int((update + 1) * slots) - int(update * slots)
        failures = np.flatnonzero(failed[batch])
        if len(failures) > allowed:
            for i in rng.choice(failures, len(failures) - allowed, replace=False):
                batch[i] = rng.choice(by_goal.get(refs[batch[i]], successes))
    return draws


def replay_pool(archive, reports, targets):
    rows = [row for stage in reports for row in stage["collection"] if row["retained_pairs"]]
    size = len(targets) + sum(len(row["retained_positions"]) for row in rows)
    tokens = torch.empty((size, *targets.shape[1:]), dtype=torch.float16)
    tokens[:len(targets)] = targets
    triples, failed, offset = [], [], len(targets)
    for i, row in enumerate(rows, 1):
        case = row["case"]
        path = archive.root / "replay" / f"{case['case_id']}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        chunk, pairs = saved["tokens"], saved["pairs"]
        if (saved["inputs"]["runtime_id"] != archive.runtime_id or saved["inputs"]["case"] != case
                or len(pairs) != row["retained_pairs"] or len(chunk) != len(row["retained_positions"])):
            raise ValueError(f"archived replay differs for {case['case_id']}")
        tokens[offset:offset + len(chunk)] = chunk
        triples.extend((case["goal_index"], offset + a, offset + b) for a, b in pairs)
        failed.extend([not row["success"]] * len(pairs))
        offset += len(chunk)
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(f"Loaded replay {i}/{len(rows)} episodes ({len(triples)} pairs)", flush=True)
    return tokens, triples, np.asarray(failed, dtype=bool)


def load_trial(runtime, archive, manifest, fitting, cap):
    rounds = manifest["rounds_per_band"]
    first, second = 6 * rounds, 6 * rounds + 1
    incoming, prior = experience._checkpoint(archive, first - 1, rounds)
    control, original = experience._checkpoint(archive, first, rounds)
    test = read(archive.root / experience._stage_name(second, rounds) / "complete.json")
    if (incoming is None or control is None or original["collector"] != prior["checkpoint_sha256"]
            or test["runtime_id"] != archive.runtime_id or test["collector"] != original["checkpoint_sha256"]
            or [row["case"] for row in test["collection"]] != manifest["stages"][second]):
        raise ValueError("archived Band 7 Round 1 and Round 2 do not form the expected comparison")

    target_path = archive.root / "collection_goals.pt"
    saved = torch.load(target_path, map_location="cpu", weights_only=True)
    if saved["runtime_id"] != archive.runtime_id or file_sha256(target_path) != original["goal_sha256"]:
        raise ValueError("archived collection goal tensors differ")
    targets = saved["tokens"]
    provenance = dict(runtime_id=runtime.runtime_id, incoming=prior["checkpoint_sha256"],
                      control=original["checkpoint_sha256"], targets=original["goal_sha256"], fraction=cap)
    path = runtime.root / "adapter.pt"
    marker = runtime.root / "fit_complete.json"
    if marker.exists():
        meta = read(marker)
        if meta["provenance"] != provenance or meta["sha256"] != file_sha256(path):
            raise ValueError("capped checkpoint provenance differs")
        capped, _ = load_adapter(path, runtime.device, expected_runtime_id=runtime.runtime_id)
    else:
        reports = [read(archive.root / experience._stage_name(i, rounds) / "complete.json")
                   for i in range(first)] + [original]
        if any(report["runtime_id"] != archive.runtime_id for report in reports):
            raise ValueError("archived replay rounds belong to another run")
        tokens, triples, failed = replay_pool(archive, reports, targets)
        old_training = read(archive.root / experience._stage_name(first, rounds) / "training.json")
        if (len(triples) != old_training["eligible_triples"] or old_training["scale"] != prior["scale"]
                or any(old_training[key] != value for key, value in fitting.items())):
            raise ValueError("archived fit does not match reconstructed replay")
        original_draws = np.asarray(old_training["sampled_triplet_ids"]).reshape(
            fitting["updates"], fitting["batch_size"])
        draws = capped_schedule(original_draws, triples, failed, cap, seed=fitting["seed"])
        goal_indices = np.asarray(triples)[:, 0]
        stats = dict(pool_pairs=len(triples), failure_pairs=int(failed.sum()), total_draws=draws.size,
                     original_failure_draws=int(failed[original_draws].sum()),
                     capped_failure_draws=int(failed[draws].sum()), replacements=int((draws != original_draws).sum()),
                     goal_changes=int((goal_indices[draws] != goal_indices[original_draws]).sum()))
        print("Replay sample counts", stats, flush=True)
        capped, training = fit(tokens, triples, prior["scale"], initial_adapter=incoming,
                               device=runtime.device, sample_schedule=draws, **fitting,
                               progress=lambda row: print(row, flush=True))
        save_adapter(path, capped, training, provenance)
        meta = dict(provenance=provenance, sha256=file_sha256(path), **stats)
        write_json(marker, meta)
    return control, capped, targets, test, meta


def evaluate(runtime, manifest, design, control, capped, targets, test, meta):
    cases = [dict(row["case"], split="eval") for row in test["collection"]]
    scorers = [adapter_goal_cost(capped, goal) for goal in targets.to(runtime.device)]
    stage = "band_7_round_2_eval"
    inner = manifest["bands"][6][0]

    def execute(case):
        record = runtime.run_case(case, latent_cost_fn=scorers[case["goal_index"]],
                                  record_positions=True, journal_stage=stage)
        record["boundary"] = experience_exploration.measure(
            dict(record, case=case), inner, runtime.config.actions_per_model_step, right_room_only=False)
        record["crossed"] = case["route"] in experience_square.crossings(record["primitive_positions"])
        return record

    condition = dict(adapter=meta["sha256"], baseline=digest(test["collection"]))
    records = run_stage(runtime.root, stage, cases, runtime.runtime_id, execute, condition=condition)
    rows = [dict(case=case, original_success=before["success"], capped_success=after["success"],
                 original_crossed=before["reached_other_room"], capped_crossed=after["crossed"],
                 original=before["boundary"], capped=after["boundary"])
            for case, before, after in zip(cases, test["collection"], records, strict=True)]
    tokens = experience._encode_positions(runtime, design, list(range(len(design["points"]))))
    goal_ids = manifest["goal_ids"]
    tokens[goal_ids] = targets

    def grid_cost(adapter):
        return [[None if i is None else float(row[i]) for i in design["grid"]]
                for row in _scores(adapter, tokens, goal_ids, runtime.device)]

    result = dict(sampling=meta, cases=rows, axis=design["axis"],
                  control_grid=grid_cost(control), capped_grid=grid_cost(capped))
    write_json(runtime.root / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source-run", "output-dir", "world-model-checkpoint", "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=.15)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if not 0 <= args.fraction <= 1:
        parser.error("--fraction must be between zero and one")
    args.output_dir, source = locations(args.output_dir, args.source_run,
                                       (args.world_model_checkpoint, args.encoder_checkpoint, args.dino_wm_root))
    original = read(source / "experiment.json")
    manifest, fitting = original["manifest"], original["fitting"]
    saved = read(source / "runtime.json")
    if (original["kind"] != "experience_mpc_failure_feedback_full_square_v1"
            or saved["runtime_id"] != digest(saved["signature"])
            or saved["signature"]["manifest"] != digest(manifest)):
        raise ValueError("expected a completed archived failure-feedback run")
    config = dict(kind="experience_failure_cap_v1", archive=str(source), source_id=saved["runtime_id"],
                  fraction=args.fraction)
    config_path = args.output_dir / "experiment.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("failure-cap experiment configuration changed")
    write_json(config_path, config)
    args.precision, args.encoder_batch_size, args.run_seed, args.videos = "fp32", 8, manifest["seed"], False
    args.expected_world_model_sha256, args.expected_encoder_sha256 = EXPECTED_WM, EXPECTED_ENCODER
    args.runtime_signature_extra = runtime_variant(config, "experience_failure_cap", args.device)
    runtime = PilotRuntime(args, manifest)
    try:
        prior, current = saved["signature"], read(runtime.root / "runtime.json")["signature"]
        keys = ("models", "dino_commit", "dino_source", "torch_version", "device_type", "precision")
        variant = ("device_name", "matmul_precision")
        if (any(prior[key] != current[key] for key in keys)
                or any(prior["runtime_variant"][key] != current["runtime_variant"][key] for key in variant)):
            raise ValueError("use the same GPU, Torch version and numerical settings as the archived run")
        archive = SimpleNamespace(root=source, runtime_id=saved["runtime_id"], device=runtime.device)
        trial = load_trial(runtime, archive, manifest, fitting, args.fraction)
        return evaluate(runtime, manifest, read(source / "design.json")["design"], *trial)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
