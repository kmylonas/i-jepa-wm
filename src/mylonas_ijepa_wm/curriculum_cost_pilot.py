"""Staged near-to-far planning-cost adapter pilot (run with python -m)."""

import argparse
import json
from pathlib import Path
import sys

from src.mylonas_ijepa_wm.curriculum_protocol import (
    PLANNING, TRAINING, competence_gate, digest, load_stage, make_manifest,
    paired_report, run_stage, write_json,
)
from src.mylonas_ijepa_wm.curriculum_replay import fit_adapter, load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256, interaction_budget,
)


def require_baseline(root, manifest, runtime_id):
    records = load_stage(root, "baseline", manifest["evaluation"], runtime_id)
    gate = competence_gate(records)
    write_json(Path(root) / "gate.json", gate)
    if not gate["passed"]:
        raise ValueError(f"near-goal competence gate failed ({gate['successes']}/{gate['episodes']}); run position diagnostic")
    return records


def train_stage(root, manifest, run_seed, runtime_id, *, goal, encode_episode, device):
    root = Path(root)
    require_baseline(root, manifest, runtime_id)
    cases = manifest["training"][str(run_seed)]
    records = load_stage(root, f"run_{run_seed}/collect", cases, runtime_id)
    status_path = root / "stages" / f"run_{run_seed}/train.json"
    checkpoint = root / f"run_{run_seed}/adapter.pt"
    if checkpoint.exists():
        _, saved = load_adapter(checkpoint, device, expected_runtime_id=runtime_id)
        training = saved["training"]
        write_json(root / f"run_{run_seed}/training.json", {key: value for key, value in training.items() if key != "optimizer"})
        write_json(status_path, dict(status="complete", runtime_id=runtime_id,
                                      successful_episodes=sum(record["success"] for record in records), attempted_episodes=len(records),
                                      primitive_interactions=sum(record["micro_steps"] for record in records),
                                      checkpoint_sha256=file_sha256(checkpoint)))
        return checkpoint
    try:
        successes = [record for record in records if record["success"]]
        if not successes:
            raise ValueError("no successful training episodes; adaptation stopped")
        replay = [encode_episode(record) for record in successes]
        if any(episode["case_id"] not in {case["case_id"] for case in cases}
               or episode.get("runtime_id") != runtime_id for episode in replay):
            raise ValueError("training replay provenance mismatch")
        target = goal() if callable(goal) else goal
        adapter, training = fit_adapter(
            replay, target, seed=run_seed, device=device, **manifest["fitting"],
            progress=lambda row: print(f"update {row['update']} loss={row['loss']:.5f}", flush=True)
            if row["update"] == 1 or row["update"] % 50 == 0 else None,
        )
        provenance = dict(runtime_id=runtime_id, manifest_digest=digest(manifest),
                          run_seed=run_seed, collection_case_ids=[case["case_id"] for case in cases])
        save_adapter(checkpoint, adapter, training, provenance)
        write_json(root / f"run_{run_seed}/training.json", {key: value for key, value in training.items() if key != "optimizer"})
        write_json(status_path, dict(status="complete", runtime_id=runtime_id,
                                      successful_episodes=len(successes), attempted_episodes=len(records),
                                      primitive_interactions=sum(record["micro_steps"] for record in records),
                                      checkpoint_sha256=file_sha256(checkpoint)))
    except ValueError as error:
        write_json(status_path, dict(status="stopped", runtime_id=runtime_id, reason=str(error)))
        raise
    return checkpoint


def make_report(root, manifest):
    root = Path(root)
    runtime = json.loads((root / "runtime.json").read_text())
    runtime_id = runtime["runtime_id"]
    if runtime["signature"]["manifest"] != digest(manifest):
        raise ValueError("report manifest provenance mismatch")
    raw = load_stage(root, "baseline", manifest["evaluation"], runtime_id)
    report = dict(runtime_id=runtime_id, protocol_variant=manifest["protocol_variant"],
                  gate=competence_gate(raw), runs={}, recorded_interaction_budget=interaction_budget(root, runtime_id))
    for seed, cases in manifest["training"].items():
        status = root / "stages" / f"run_{seed}/evaluate.json"
        if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
            continue
        adapted = load_stage(root, f"run_{seed}/evaluate", manifest["evaluation"], runtime_id)
        adapter_sha = file_sha256(root / f"run_{seed}/adapter.pt")
        if any(item["condition"] != {"cost": "latent", "adapter_sha256": adapter_sha} for item in adapted):
            raise ValueError("evaluation adapter provenance mismatch")
        result = paired_report(raw, adapted)
        collected = load_stage(root, f"run_{seed}/collect", cases, runtime_id)
        result["collection"] = dict(attempts=len(collected), successes=sum(item["success"] for item in collected),
                                     primitive_interactions=sum(item["micro_steps"] for item in collected))
        report["runs"][seed] = result
    write_json(root / "report.json", report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "baseline", "position", "collect", "train", "evaluate", "report", "diagnostics"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--dino-wm-root", type=Path)
    parser.add_argument("--expected-world-model-sha256", default=EXPECTED_WM)
    parser.add_argument("--expected-encoder-sha256", default=EXPECTED_ENCODER)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="fp32")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[0, 1, 2], help="prepare only")
    parser.add_argument("--manifest-seed", type=int, default=20261105)
    parser.add_argument("--eval-per-band", type=int, default=20, help="prepare only")
    parser.add_argument("--train-per-run", type=int, default=40, help="prepare only")
    for key in ("planning_horizon", "actions_per_plan", "population", "num_elites", "cem_iterations", "candidate_batch_size", "max_macro_steps"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, help="prepare only; overrides mark protocol modified")
    parser.add_argument("--updates", type=int, help="prepare only")
    parser.add_argument("--batch-size", type=int, help="prepare only; adapter fitting")
    parser.add_argument("--audit-candidates", type=int, default=8, help="diagnostics: samples per initial/elite subset")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.output_dir
    manifest_path = root / "manifest.json"
    if args.stage == "prepare":
        manifest = make_manifest(args.eval_per_band, args.train_per_run, tuple(args.run_seeds), args.manifest_seed)
        for key in manifest["planning"]:
            if hasattr(args, key) and getattr(args, key) is not None:
                manifest["planning"][key] = getattr(args, key)
        for key in ("updates", "batch_size"):
            if getattr(args, key) is not None:
                manifest["fitting"][key] = getattr(args, key)
        planning = manifest["planning"]
        if (min(planning[key] for key in ("planning_horizon", "actions_per_plan", "population", "num_elites", "cem_iterations", "candidate_batch_size", "max_macro_steps")) < 1
                or planning["actions_per_plan"] > planning["planning_horizon"] or planning["num_elites"] > planning["population"]
                or manifest["fitting"]["updates"] < 1 or manifest["fitting"]["batch_size"] < 1):
            raise ValueError("invalid pilot counts or CEM configuration")
        if manifest["planning"] != PLANNING or manifest["fitting"] != TRAINING:
            manifest["protocol_variant"] = "modified"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("existing manifest differs; use a new output directory")
        write_json(manifest_path, manifest)
        print(f"Prepared {len(manifest['evaluation'])} evaluation cases; variant={manifest['protocol_variant']}")
        return manifest
    manifest = json.loads(manifest_path.read_text())
    if args.stage == "report":
        result = make_report(root, manifest)
        from src.mylonas_ijepa_wm.curriculum_diagnostics import plot_report
        plot_report(root, manifest, result)
        print(json.dumps(result, indent=2))
        return result
    if any(getattr(args, name) is None for name in ("world_model_checkpoint", "encoder_checkpoint", "dino_wm_root")):
        raise ValueError("model, encoder and pinned DINO-WM paths are required for execution stages")
    if args.encoder_batch_size < 1 or args.audit_candidates < 1:
        raise ValueError("encoding and audit batch counts must be positive")
    runtime = PilotRuntime(args, manifest)
    try:
        run_seed = args.run_seed
        if str(run_seed) not in manifest["training"]:
            raise ValueError("run seed was not declared in the manifest")
        raw_condition = dict(cost="latent", adapter_sha256=None)
        if args.stage in ("baseline", "position"):
            cost = "position" if args.stage == "position" else "latent"
            records = run_stage(root, args.stage, manifest["evaluation"], runtime.runtime_id,
                                lambda case: runtime.run_case(case, cost=cost),
                                condition=dict(cost=cost, adapter_sha256=None))
            if args.stage == "baseline":
                require_baseline(root, manifest, runtime.runtime_id)
            return records
        require_baseline(root, manifest, runtime.runtime_id)
        if args.stage == "collect":
            return run_stage(root, f"run_{run_seed}/collect", manifest["training"][str(run_seed)], runtime.runtime_id,
                             lambda case: runtime.run_case(case, capture=True), condition=raw_condition)
        if args.stage == "train":
            return train_stage(root, manifest, run_seed, runtime.runtime_id,
                               goal=runtime.goal_tokens, encode_episode=runtime.encode_episode, device=runtime.device)
        adapter_path = root / f"run_{run_seed}/adapter.pt"
        adapter, _ = load_adapter(adapter_path, runtime.device, expected_runtime_id=runtime.runtime_id)
        if args.stage == "evaluate":
            return run_stage(root, f"run_{run_seed}/evaluate", manifest["evaluation"], runtime.runtime_id,
                             lambda case: runtime.run_case(case, metric=adapter),
                             condition=dict(cost="latent", adapter_sha256=file_sha256(adapter_path)))
        from src.mylonas_ijepa_wm.curriculum_diagnostics import run_diagnostics
        return run_diagnostics(runtime, adapter, run_seed, args.audit_candidates)
    finally:
        runtime.close()


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"Pilot stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
