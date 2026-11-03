"""Run matched DotWall CEM evaluations across held-out layouts and action modes."""

import argparse
from hashlib import sha256
import json
import math
import numpy as np
from pathlib import Path
import subprocess
import sys

import torch

from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.wall_layout_manifest import (
    TRAIN_LAYOUTS, load_manifest, manifest_digest, validate_checkpoint_manifest,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_identity(path: Path, expected: dict) -> None:
    if not path.is_file():
        raise ValueError(f"Planning result identity is missing: {path}")
    if json.loads(path.read_text()) != expected:
        raise ValueError(f"Planning result identity differs from current inputs: {path}")


def wilson_interval(successes: int, episodes: int) -> tuple[float, float]:
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError("Invalid success count")
    z = 1.959963984540054
    proportion = successes / episodes
    denominator = 1 + z * z / episodes
    center = (proportion + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / episodes + z * z / (4 * episodes * episodes)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_invocations(
    manifest: dict, configs: dict, output_root: Path,
    encoder_checkpoint: Path, multi_checkpoint: Path, fixed_checkpoint: Path,
    base_seed: int = 10000, num_episodes: int = 20,
    device: str = "auto", precision: str = "auto",
    matched: bool = False,
) -> list[dict]:
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    output_root = Path(output_root)
    runs = []
    for layout in manifest["test_layouts"]:
        layout_key = tuple(layout)
        if layout_key not in configs:
            raise ValueError(f"Missing environment config for layout {layout}")
        for model_name, checkpoint in (
            ("multi", multi_checkpoint), ("fixed", fixed_checkpoint),
        ):
            modes = (("cap_train", float(manifest["max_training_action_length"])),) if matched else (
                ("unbounded", None), ("cap4", 4.0),
                ("cap_train", float(manifest["max_training_action_length"])),
            )
            for mode, cap in modes:
                stem = f"{model_name}_wall{layout[0]}_door{layout[1]}_{mode}"
                output = output_root / f"{stem}.json"
                argv = [
                    "--env", "wall", "--env-config", str(configs[layout_key]),
                    "--encoder-checkpoint", str(encoder_checkpoint),
                    "--world-model-checkpoint", str(checkpoint),
                    "--output", str(output),
                    "--cost", "position", "--num-episodes", str(num_episodes),
                    "--base-seed", str(base_seed),
                    "--max-macro-steps", "20", "--planning-horizon", "5",
                    "--actions-per-plan", "5", "--population", "300",
                    "--num-elites", "30", "--cem-iterations", "10",
                    "--candidate-batch-size", "64", "--device", device,
                    "--precision", precision, "--unbounded-actions",
                ]
                if not matched:
                    argv.extend(("--video-dir", str(output_root / f"{stem}_videos")))
                if cap is not None:
                    argv.extend(("--max-action-length", str(cap)))
                runs.append(dict(
                    model=model_name, mode=mode, layout=list(layout),
                    checkpoint=str(checkpoint), output=str(output), argv=argv,
                ))
    return runs


def paired_success_summary(runs: list[dict], bootstrap_samples: int = 10000) -> dict:
    """Pair model outcomes, then resample seed clusters across all layouts."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    outcomes = {}
    for run in runs:
        key = (run["model"], tuple(run["wall_layout"]))
        if key in outcomes:
            raise ValueError(f"Duplicate planning run: {key}")
        outcomes[key] = {episode["seed"]: bool(episode["success"])
                         for episode in run["episodes"]}
    layouts = sorted({layout for _, layout in outcomes})
    if not layouts or any((model, layout) not in outcomes
                          for model in ("multi", "fixed") for layout in layouts):
        raise ValueError("Missing paired model/layout results")
    seeds = sorted(outcomes[("multi", layouts[0])])
    if not seeds or any(sorted(outcomes[(model, layout)]) != seeds
                        for model in ("multi", "fixed") for layout in layouts):
        raise ValueError("Planning seed sets differ")
    differences = np.asarray([
        [int(outcomes[("multi", layout)][seed]) - int(outcomes[("fixed", layout)][seed])
         for layout in layouts]
        for seed in seeds
    ], dtype=np.int8)
    rng = np.random.default_rng(20261103)
    samples = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    boot = differences[samples].mean(axis=(1, 2))
    return dict(
        paired_episode_count=int(differences.size), seed_clusters=len(seeds),
        success_difference=float(differences.mean()),
        discordant_multi_only=int((differences == 1).sum()),
        discordant_fixed_only=int((differences == -1).sum()),
        bootstrap_seed=20261103, bootstrap_samples=bootstrap_samples,
        bootstrap_95=np.quantile(boot, [0.025, 0.975]).tolist(),
        by_layout={f"{x},{y}": float(differences[:, i].mean())
                   for i, (x, y) in enumerate(layouts)},
    )


def summarize_run(payload: dict, layout: list[int], expected_seeds: list[int]) -> dict:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(expected_seeds):
        raise ValueError("Planning output episode count disagrees with request")
    if [record.get("seed") for record in episodes] != expected_seeds:
        raise ValueError("Planning output seeds disagree with request")
    if any(record.get("wall_layout") != layout for record in episodes):
        raise ValueError("Planning output layout disagrees with request")
    successes = sum(bool(record["success"]) for record in episodes)
    failure_counts = dict(step_limit=0, terminated=0, truncated=0, other=0)
    for record in episodes:
        if record["success"]:
            continue
        if record.get("failed_by_step_limit"):
            failure_counts["step_limit"] += 1
        elif record.get("terminated"):
            failure_counts["terminated"] += 1
        elif record.get("truncated"):
            failure_counts["truncated"] += 1
        else:
            failure_counts["other"] += 1
    return dict(
        episodes=len(episodes), successes=successes,
        success_rate=successes / len(episodes),
        wilson_95=list(wilson_interval(successes, len(episodes))),
        mean_final_distance=sum(float(r["final_goal_distance"]) for r in episodes) / len(episodes),
        failure_counts=failure_counts,
    )


def expected_run_configuration(item: dict) -> dict:
    """Read the exact scientific settings requested for one planning run."""
    from src.mylonas_ijepa_wm.plan_proprio import parse_args

    args = parse_args(item["argv"])
    return dict(
        environment=load_environment_config(args.env, args.env_config).portable_metadata(),
        checkpoint=str(args.world_model_checkpoint),
        encoder_checkpoint=str(args.encoder_checkpoint),
        cost=args.cost, num_episodes=args.num_episodes, base_seed=args.base_seed,
        max_macro_steps=args.max_macro_steps, planning_horizon=args.planning_horizon,
        actions_per_plan=args.actions_per_plan, population=args.population,
        num_elites=args.num_elites, cem_iterations=args.cem_iterations,
        candidate_batch_size=args.candidate_batch_size, min_std=args.min_std,
        unbounded_actions=args.unbounded_actions,
        max_action_length=args.max_action_length,
    )


def validate_planning_result(payload: dict, item: dict, expected_seeds: list[int]) -> dict:
    expected = expected_run_configuration(item)
    saved = payload.get("configuration", {})
    mismatches = [key for key, value in expected.items() if saved.get(key) != value]
    if mismatches:
        raise ValueError(f"Planning result configuration differs for {mismatches}: {item['output']}")
    return summarize_run(payload, item["layout"], expected_seeds)


def run_matrix(
    manifest: dict, env_config: Path, output_root: Path,
    encoder_checkpoint: Path, multi_checkpoint: Path, fixed_checkpoint: Path,
    base_seed: int = 10000, num_episodes: int = 20,
    device: str = "auto", precision: str = "auto", resume: bool = False,
    fixed_manifest: dict | None = None, matched: bool = False,
) -> dict:
    multi_metadata = torch.load(
        multi_checkpoint, map_location="cpu", weights_only=True,
    ).get("metadata", {})
    validate_checkpoint_manifest(multi_metadata, manifest)
    if matched:
        if manifest["train_layouts"] != TRAIN_LAYOUTS:
            raise ValueError("Multi manifest must use the nine planned training layouts")
        if fixed_manifest is None:
            raise ValueError("Matched comparison requires a fixed manifest")
        if fixed_manifest["train_layouts"] != [[32, 30]]:
            raise ValueError("Fixed manifest must train on default [32, 30]")
        if manifest.get("collection_profile") != "wide_only" or fixed_manifest.get("collection_profile") != "wide_only":
            raise ValueError("Matched datasets must use the wide_only policy")
        if manifest["max_training_action_length"] != 3.5 or fixed_manifest["max_training_action_length"] != 3.5:
            raise ValueError("Matched action maxima must be 3.5")
        for split in ("val", "test"):
            left = [entry for entry in manifest["episodes"] if entry["split"] == split]
            right = [entry for entry in fixed_manifest["episodes"] if entry["split"] == split]
            if left != right:
                raise ValueError(f"Matched {split} assignments differ")
        fixed_metadata = torch.load(fixed_checkpoint, map_location="cpu", weights_only=True).get("metadata", {})
        validate_checkpoint_manifest(fixed_metadata, fixed_manifest)
    file_identities = {
        "encoder": file_sha256(encoder_checkpoint),
        "multi": file_sha256(multi_checkpoint),
        "fixed": file_sha256(fixed_checkpoint),
    }
    config = load_environment_config("wall", env_config)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "layout_configs"
    config_dir.mkdir(exist_ok=True)
    configs = {}
    for layout in manifest["test_layouts"]:
        local = {**config.portable_metadata(), "dino_wm_root": config.dino_wm_root}
        local.update(
            wall_location=layout[0], door_location=layout[1],
            wall_layouts=[list(layout)],
        )
        path = config_dir / f"wall{layout[0]}_door{layout[1]}.json"
        if path.exists() and json.loads(path.read_text()) != local:
            raise ValueError(f"Existing layout config differs: {path}")
        path.write_text(json.dumps(local, indent=2))
        configs[tuple(layout)] = path
    invocations = build_invocations(
        manifest, configs, output_root, encoder_checkpoint,
        multi_checkpoint, fixed_checkpoint, base_seed, num_episodes,
        device, precision, matched,
    )
    expected_seeds = list(range(base_seed, base_seed + num_episodes))
    results = []
    paired_runs = []
    for item in invocations:
        path = Path(item["output"])
        identity_path = path.with_suffix(".identity.json")
        identity = dict(
            manifest_sha256=manifest_digest(manifest),
            checkpoint_sha256=file_identities[item["model"]],
            encoder_sha256=file_identities["encoder"],
        )
        if path.exists() and not resume:
            raise FileExistsError(f"Planning result already exists: {path}")
        if path.exists():
            validate_run_identity(identity_path, identity)
        else:
            subprocess.run([
                sys.executable, "-m", "src.mylonas_ijepa_wm.plan_proprio",
                *item["argv"],
            ], check=True)
            identity_path.write_text(json.dumps(identity, indent=2))
        payload = json.loads(path.read_text())
        if matched:
            paired_runs.append(dict(model=item["model"], wall_layout=item["layout"], episodes=payload["episodes"]))
        results.append({
            "model": item["model"], "mode": item["mode"],
            "wall_layout": item["layout"], "result_path": str(path),
            "identity_path": str(identity_path),
            **validate_planning_result(payload, item, expected_seeds),
        })
    for name, checkpoint in (
        ("encoder", encoder_checkpoint), ("multi", multi_checkpoint),
        ("fixed", fixed_checkpoint),
    ):
        if file_sha256(checkpoint) != file_identities[name]:
            raise ValueError(f"Checkpoint identity changed during planning: {checkpoint}")
    summary = dict(
        base_seed=base_seed, num_episodes=num_episodes,
        file_identities=file_identities,
        action_modes=(dict(cap_train=3.5) if matched else
                      dict(unbounded=None, cap4=4.0,
                           cap_train=manifest["max_training_action_length"])),
        baseline_note=("New fixed and multi checkpoints use matched collection and evaluation settings."
                       if matched else "Fixed-layout checkpoint is a historical, non-isolated transfer baseline."),
        runs=results,
    )
    if matched:
        summary["fixed_manifest_sha256"] = manifest_digest(fixed_manifest)
        summary["multi_manifest_sha256"] = manifest_digest(manifest)
        summary["paired_comparison"] = paired_success_summary(paired_runs)
    (output_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--multi-checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run_matrix(
        load_manifest(args.episode_manifest), args.env_config, args.output_root,
        args.encoder_checkpoint, args.multi_checkpoint, args.fixed_checkpoint,
        args.base_seed, args.num_episodes or (50 if args.matched else 20),
        args.device, args.precision, args.resume,
        load_manifest(args.fixed_manifest) if args.fixed_manifest else None,
        args.matched,
    )
    print(f"Saved {len(summary['runs'])} matched planning runs to {args.output_root}")


if __name__ == "__main__":
    main()
