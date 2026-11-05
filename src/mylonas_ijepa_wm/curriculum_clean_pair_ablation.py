"""Replay-only oracle clean-pair ablation with a separate evaluation runtime."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.mylonas_ijepa_wm.curriculum_protocol import digest, load_stage, paired_report, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_replay import fit_adapter, load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256, interaction_budget, save_tensors,
)


RUN_SEED = 0
MIN_GOAL_PROGRESS = .5


def _read(path):
    return json.loads(Path(path).read_text())


def _parent_snapshot(parent, margin):
    parent = Path(parent).resolve()
    required = [parent / name for name in (
        "manifest.json", "runtime.json", "report.json", "run_0/adapter.pt",
        "run_0/training.json", "stages/run_0/train.json",
    )]
    if (not all(path.is_file() for path in required)
            or _read(parent / "stages/run_0/train.json").get("status") != "complete"):
        raise ValueError("complete parent pilot run 0 is required")
    manifest, runtime = _read(parent / "manifest.json"), _read(parent / "runtime.json")
    runtime_id = runtime.get("runtime_id")
    if not runtime_id or runtime.get("signature", {}).get("manifest") != digest(manifest):
        raise ValueError("complete parent pilot has invalid manifest provenance")
    if str(RUN_SEED) not in manifest.get("training", {}):
        raise ValueError("complete parent pilot does not contain run 0")
    baseline = load_stage(parent, "baseline", manifest["evaluation"], runtime_id)
    collection = load_stage(parent, "run_0/collect", manifest["training"]["0"], runtime_id)
    original = load_stage(parent, "run_0/evaluate", manifest["evaluation"], runtime_id)
    adapter_sha = file_sha256(parent / "run_0/adapter.pt")
    if any(record.get("condition") != {"cost": "latent", "adapter_sha256": adapter_sha}
           for record in original):
        raise ValueError("complete parent pilot original adapter condition differs")
    training = _read(parent / "run_0/training.json")
    expected_ids = [record["case_id"] for record in collection if record["success"]]
    if set(training.get("training_case_ids", [])) != set(expected_ids):
        raise ValueError("complete parent pilot training IDs differ from successful collection")
    replay_files = []
    total_frames = 0
    cases = {case["case_id"]: case for case in manifest["training"]["0"]}
    for case_id in expected_ids:
        path = parent / "replay" / "run_0" / f"{case_id}.pt"
        if not path.is_file():
            raise ValueError("complete parent pilot is missing successful replay")
        replay = torch.load(path, map_location="cpu", weights_only=True)
        if (replay.get("runtime_id") != runtime_id or replay.get("case_id") != case_id
                or replay.get("case") != cases[case_id] or replay.get("split") != "train"
                or not replay.get("success") or len(replay.get("latents", [])) != len(replay.get("states", []))
                or len(replay.get("states", [])) != len(replay.get("primitive_steps", []))):
            raise ValueError(f"complete parent pilot replay provenance differs: {case_id}")
        total_frames += len(replay["states"])
        replay_files.append(dict(case_id=case_id,
                                 path=path.relative_to(parent).as_posix(), sha256=file_sha256(path)))
    models = runtime.get("signature", {}).get("models")
    if not isinstance(models, dict) or set(models) != {"world_model", "encoder"}:
        raise ValueError("complete parent pilot model provenance is missing")
    return dict(schema_version=1, parent_dir=str(parent), parent_runtime_id=runtime_id,
                parent_manifest_digest=digest(manifest), parent_adapter_sha256=adapter_sha,
                parent_models=models,
                run_seed=RUN_SEED,
                pair_filter=dict(kind="oracle_goal_progress", min_goal_progress=float(margin)),
                replay_episodes=len(replay_files), replay_frames=total_frames,
                replay_files=replay_files, replay_digest=digest(replay_files),
                parent_baseline_episodes=len(baseline), parent_original_evaluation_episodes=len(original))


def prepare_ablation(parent, root, min_goal_progress=MIN_GOAL_PROGRESS):
    if not np.isfinite(min_goal_progress) or min_goal_progress <= 0:
        raise ValueError("minimum goal progress must be finite and positive")
    parent, root = Path(parent).resolve(), Path(root).resolve()
    config = _parent_snapshot(parent, min_goal_progress)
    manifest = _read(parent / "manifest.json")
    root.mkdir(parents=True, exist_ok=True)
    for path, value, label in ((root / "manifest.json", manifest, "manifest"),
                               (root / "ablation.json", config, "clean-pair configuration")):
        if path.exists() and _read(path) != value:
            raise ValueError(f"existing {label} differs; use a new output directory")
        if not path.exists():
            write_json(path, value)
    return config


def _config(parent, root):
    parent, root = Path(parent).resolve(), Path(root).resolve()
    config = _read(root / "ablation.json")
    if config.get("parent_dir") != str(parent):
        raise ValueError("clean-pair parent directory differs")
    manifest = _read(root / "manifest.json")
    if config.get("parent_manifest_digest") != digest(manifest):
        raise ValueError("clean-pair manifest differs from parent provenance")
    return config, manifest


def _verify_replay_hashes(parent, config):
    parent = Path(parent)
    current = []
    for item in config["replay_files"]:
        path = parent / item["path"]
        if not path.is_file():
            raise ValueError("parent replay changed or is missing")
        current.append(dict(case_id=item["case_id"], path=item["path"], sha256=file_sha256(path)))
    if current != config["replay_files"] or digest(current) != config["replay_digest"]:
        raise ValueError("parent replay changed after clean-pair prepare")


def load_parent_replay(parent, root):
    parent = Path(parent).resolve()
    config, manifest = _config(parent, root)
    _verify_replay_hashes(parent, config)
    runtime_id = _read(parent / "runtime.json")["runtime_id"]
    cases = {case["case_id"]: case for case in manifest["training"]["0"]}
    replay = []
    for item in config["replay_files"]:
        saved = torch.load(parent / item["path"], map_location="cpu", weights_only=True)
        if (saved.get("runtime_id") != runtime_id or saved.get("case_id") != item["case_id"]
                or saved.get("case") != cases[item["case_id"]] or not saved.get("success")):
            raise ValueError("parent replay provenance changed")
        replay.append(saved)
    return replay


def _child_runtime(root, runtime_id):
    runtime = _read(Path(root) / "runtime.json")
    manifest = _read(Path(root) / "manifest.json")
    if (runtime.get("runtime_id") != runtime_id
            or runtime.get("signature", {}).get("manifest") != digest(manifest)
            or runtime.get("signature", {}).get("models") != _read(Path(root) / "ablation.json").get("parent_models")):
        raise ValueError("child runtime provenance differs")


def _training_json(root, training):
    write_json(Path(root) / "run_0/training.json",
               {key: value for key, value in training.items() if key != "optimizer"})


def train_clean_stage(parent, root, runtime_id, *, goal, device):
    parent, root = Path(parent).resolve(), Path(root).resolve()
    config, manifest = _config(parent, root)
    _child_runtime(root, runtime_id)
    replay = load_parent_replay(parent, root)
    checkpoint = root / "run_0/adapter.pt"
    status_path = root / "stages/run_0/train.json"
    goal_path = root / "run_0/goal_tokens.pt"
    config_digest = digest(config)
    if checkpoint.exists():
        _, saved = load_adapter(checkpoint, device, expected_runtime_id=runtime_id)
        provenance = saved["provenance"]
        if (provenance.get("parent_runtime_id") != config["parent_runtime_id"]
                or provenance.get("ablation_digest") != config_digest
                or provenance.get("replay_digest") != config["replay_digest"]):
            raise ValueError("completed clean adapter provenance differs")
        _training_json(root, saved["training"])
        write_json(status_path, dict(status="complete", runtime_id=runtime_id,
                                     parent_runtime_id=config["parent_runtime_id"],
                                     checkpoint_sha256=file_sha256(checkpoint),
                                     goal_tokens_sha256=file_sha256(goal_path)))
        return checkpoint
    try:
        if goal_path.exists():
            goal_saved = torch.load(goal_path, map_location="cpu", weights_only=True)
            if (goal_saved.get("runtime_id") != runtime_id
                    or goal_saved.get("ablation_digest") != config_digest):
                raise ValueError("saved goal token provenance differs")
            target = goal_saved["tokens"]
        else:
            target = goal() if callable(goal) else goal
            target = torch.as_tensor(target).detach().cpu().float().clone()
            save_tensors(goal_path, dict(runtime_id=runtime_id, ablation_digest=config_digest,
                                         tokens=target, source="frozen_encoder_goal_render"))
        margin = config["pair_filter"]["min_goal_progress"]
        adapter, training = fit_adapter(
            replay, target, seed=RUN_SEED, device=device, min_goal_progress=margin,
            **manifest["fitting"],
            progress=lambda row: print(f"update {row['update']} loss={row['loss']:.5f}", flush=True)
            if row["update"] == 1 or row["update"] % 50 == 0 else None,
        )
        provenance = dict(runtime_id=runtime_id, parent_runtime_id=config["parent_runtime_id"],
                          parent_adapter_sha256=config["parent_adapter_sha256"],
                          parent_manifest_digest=config["parent_manifest_digest"],
                          replay_digest=config["replay_digest"], ablation_digest=config_digest,
                          run_seed=RUN_SEED, pair_filter=config["pair_filter"])
        save_adapter(checkpoint, adapter, training, provenance)
        _training_json(root, training)
        write_json(status_path, dict(
            status="complete", runtime_id=runtime_id, parent_runtime_id=config["parent_runtime_id"],
            checkpoint_sha256=file_sha256(checkpoint), goal_tokens_sha256=file_sha256(goal_path),
            replay_episodes=len(replay), temporal_pairs=sum(x["temporal_pairs"] for x in training["trajectories"]),
            eligible_pairs=sum(x["eligible_pairs"] for x in training["trajectories"]),
        ))
    except ValueError as error:
        write_json(status_path, dict(status="stopped", runtime_id=runtime_id, reason=str(error)))
        raise
    return checkpoint


def clean_condition(config, adapter_sha256):
    return dict(cost="clean_pair_adapter", adapter_sha256=adapter_sha256,
                parent_runtime_id=config["parent_runtime_id"], pair_filter=config["pair_filter"])


def _without_old_screen(report):
    report = dict(report)
    report.pop("screen_passed", None)
    return report


def _plot_report(parent, root, report, raw, original, clean):
    root = Path(root)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, band in zip(axes, ("near", "far")):
        block = report["clean_vs_raw"][band]
        original_block = report["original_vs_raw"][band]
        count = block["episodes"]
        axis.bar(["Raw", "Original\nadapter", "Clean-pair\nadapter"],
                 [block["raw_successes"] / count, original_block["adapted_successes"] / count,
                  block["adapted_successes"] / count], color=["#475569", "#dc2626", "#16a34a"])
        axis.set(title=band, ylim=(0, 1), ylabel="success fraction")
    figure.savefig(root / "success_by_band.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, band in zip(axes, ("near", "far")):
        for label, records, color in (("Raw", raw, "#475569"), ("Original adapter", original, "#dc2626"),
                                      ("Clean-pair adapter", clean, "#16a34a")):
            subset = [record for record in records if record["case"]["band"] == band]
            for index, record in enumerate(subset):
                path = np.asarray(record["achieved_positions"])
                axis.plot(path[:, 0], path[:, 1], color=color, alpha=.45,
                          label=label if index == 0 else None)
        axis.set(title=band, xlim=(0, 64), ylim=(0, 64), xlabel="x", ylabel="y", aspect="equal")
        axis.legend(fontsize=8)
    figure.savefig(root / "trajectories.png", dpi=150)
    plt.close(figure)

    history = _read(root / "run_0/training.json")["history"]
    figure, axis = plt.subplots(figsize=(7, 3), constrained_layout=True)
    for key in ("rank_loss", "anchor_loss"):
        axis.plot([row["update"] for row in history], [row[key] for row in history], label=key)
    axis.set(xlabel="adapter update", ylabel="loss"); axis.legend()
    figure.savefig(root / "training_loss.png", dpi=150)
    plt.close(figure)


def make_clean_report(parent, root, *, render=True):
    parent, root = Path(parent).resolve(), Path(root).resolve()
    config, manifest = _config(parent, root)
    _verify_replay_hashes(parent, config)
    parent_runtime = _read(parent / "runtime.json")["runtime_id"]
    child_runtime = _read(root / "runtime.json")["runtime_id"]
    _child_runtime(root, child_runtime)
    raw = load_stage(parent, "baseline", manifest["evaluation"], parent_runtime)
    original = load_stage(parent, "run_0/evaluate", manifest["evaluation"], parent_runtime)
    clean = load_stage(root, "run_0/evaluate", manifest["evaluation"], child_runtime)
    adapter_sha = file_sha256(root / "run_0/adapter.pt")
    expected_condition = clean_condition(config, adapter_sha)
    if any(record.get("condition") != expected_condition for record in clean):
        raise ValueError("clean evaluation adapter provenance differs")
    clean_vs_raw = _without_old_screen(paired_report(raw, clean))
    original_vs_raw = _without_old_screen(paired_report(raw, original))
    near, far = clean_vs_raw["near"], clean_vs_raw["far"]
    thresholds = dict(max_near_losses=2, min_near_successes=max(0, near["raw_successes"] - 2),
                      min_far_successes=far["raw_successes"])
    passed = (near["losses"] <= thresholds["max_near_losses"]
              and near["adapted_successes"] >= thresholds["min_near_successes"]
              and far["adapted_successes"] >= thresholds["min_far_successes"])
    if not passed:
        outcome = "clean_labels_did_not_prevent_degradation"
    elif far["adapted_successes"] > far["raw_successes"]:
        outcome = "improved_fixed_far_cases"
    else:
        outcome = "avoided_degradation_without_far_gain"
    report = dict(schema_version=1, runtime_id=child_runtime, parent_runtime_id=parent_runtime,
                  ablation=config, clean_vs_raw=clean_vs_raw, original_vs_raw=original_vs_raw,
                  clean_screen=dict(passed=passed, thresholds=thresholds,
                                    note="Operational fixed-case screen, not a significance test."),
                  outcome=outcome,
                  interaction_budgets=dict(
                      parent_raw_completed=sum(item["micro_steps"] for item in raw),
                      parent_original_adapter_completed=sum(item["micro_steps"] for item in original),
                      clean_evaluation_completed=sum(item["micro_steps"] for item in clean),
                      clean_journal=interaction_budget(root, child_runtime),
                      training_simulator_interactions=0,
                  ))
    write_json(root / "report.json", report)
    if render:
        _plot_report(parent, root, report, raw, original, clean)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "report"])
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-goal-progress", type=float, default=MIN_GOAL_PROGRESS, help="prepare only")
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--dino-wm-root", type=Path)
    parser.add_argument("--expected-world-model-sha256", default=EXPECTED_WM)
    parser.add_argument("--expected-encoder-sha256", default=EXPECTED_ENCODER)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="fp32")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--videos", action="store_true")
    parser.set_defaults(run_seed=RUN_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    parent, root = args.parent_dir, args.output_dir
    if args.stage == "prepare":
        result = prepare_ablation(parent, root, args.min_goal_progress)
        print(json.dumps(result, indent=2))
        return result
    if args.stage == "report":
        result = make_clean_report(parent, root)
        print(json.dumps(result, indent=2))
        return result
    if any(getattr(args, name) is None for name in
           ("world_model_checkpoint", "encoder_checkpoint", "dino_wm_root")):
        raise ValueError("model, encoder and pinned DINO-WM paths are required for execution stages")
    config, manifest = _config(parent, root)
    runtime = PilotRuntime(args, manifest)
    try:
        if args.stage == "train":
            return train_clean_stage(parent, root, runtime.runtime_id,
                                     goal=runtime.goal_tokens, device=runtime.device)
        checkpoint = root / "run_0/adapter.pt"
        adapter, saved = load_adapter(checkpoint, runtime.device, expected_runtime_id=runtime.runtime_id)
        if saved["provenance"].get("ablation_digest") != digest(config):
            raise ValueError("clean adapter configuration differs")
        condition = clean_condition(config, file_sha256(checkpoint))
        return run_stage(root, "run_0/evaluate", manifest["evaluation"], runtime.runtime_id,
                         lambda case: runtime.run_case(case, metric=adapter), condition=condition)
    finally:
        runtime.close()


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        print(f"Clean-pair ablation stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
