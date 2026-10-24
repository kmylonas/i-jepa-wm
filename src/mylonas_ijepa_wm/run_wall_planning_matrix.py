"""Run matched DotWall CEM evaluations across held-out layouts and action modes."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.wall_layout_manifest import load_manifest


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
            for mode, cap in (
                ("unbounded", None), ("cap4", 4.0),
                ("cap_train", float(manifest["max_training_action_length"])),
            ):
                stem = f"{model_name}_wall{layout[0]}_door{layout[1]}_{mode}"
                output = output_root / f"{stem}.json"
                argv = [
                    "--env", "wall", "--env-config", str(configs[layout_key]),
                    "--encoder-checkpoint", str(encoder_checkpoint),
                    "--world-model-checkpoint", str(checkpoint),
                    "--output", str(output),
                    "--video-dir", str(output_root / f"{stem}_videos"),
                    "--cost", "position", "--num-episodes", str(num_episodes),
                    "--base-seed", str(base_seed),
                    "--max-macro-steps", "20", "--planning-horizon", "5",
                    "--actions-per-plan", "5", "--population", "300",
                    "--num-elites", "30", "--cem-iterations", "10",
                    "--candidate-batch-size", "64", "--device", device,
                    "--precision", precision, "--unbounded-actions",
                ]
                if cap is not None:
                    argv.extend(("--max-action-length", str(cap)))
                runs.append(dict(
                    model=model_name, mode=mode, layout=list(layout),
                    checkpoint=str(checkpoint), output=str(output), argv=argv,
                ))
    return runs


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


def run_matrix(
    manifest: dict, env_config: Path, output_root: Path,
    encoder_checkpoint: Path, multi_checkpoint: Path, fixed_checkpoint: Path,
    base_seed: int = 10000, num_episodes: int = 20,
    device: str = "auto", precision: str = "auto", resume: bool = False,
) -> dict:
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
        device, precision,
    )
    expected_seeds = list(range(base_seed, base_seed + num_episodes))
    results = []
    for item in invocations:
        path = Path(item["output"])
        if path.exists() and not resume:
            raise FileExistsError(f"Planning result already exists: {path}")
        if not path.exists():
            subprocess.run([
                sys.executable, "-m", "src.mylonas_ijepa_wm.plan_proprio",
                *item["argv"],
            ], check=True)
        payload = json.loads(path.read_text())
        if payload.get("configuration", {}).get("checkpoint") != item["checkpoint"]:
            raise ValueError(f"Planning result checkpoint differs: {path}")
        results.append({
            "model": item["model"], "mode": item["mode"],
            "wall_layout": item["layout"], "result_path": str(path),
            **summarize_run(payload, item["layout"], expected_seeds),
        })
    summary = dict(
        base_seed=base_seed, num_episodes=num_episodes,
        action_modes=dict(unbounded=None, cap4=4.0,
                          cap_train=manifest["max_training_action_length"]),
        baseline_note="Fixed-layout checkpoint is a historical, non-isolated transfer baseline.",
        runs=results,
    )
    (output_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--multi-checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run_matrix(
        load_manifest(args.episode_manifest), args.env_config, args.output_root,
        args.encoder_checkpoint, args.multi_checkpoint, args.fixed_checkpoint,
        args.base_seed, args.num_episodes, args.device, args.precision, args.resume,
    )
    print(f"Saved {len(summary['runs'])} matched planning runs to {args.output_root}")


if __name__ == "__main__":
    main()
