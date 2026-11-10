"""Paired online DotWall planning with one world model and two frozen costs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm import plan_proprio, wm, wm_proprio
from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost
from src.mylonas_ijepa_wm.curriculum_protocol import write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
from src.mylonas_ijepa_wm.environment_adapters import create_adapter
from src.mylonas_ijepa_wm.environment_configs import load_environment_config
from src.mylonas_ijepa_wm.wall_layout_manifest import (
    manifest_digest, validate_checkpoint_manifest,
)


CONDITIONS = ("fixed", "multi")
PLANNING = dict(cost="latent", cem_selection="best", max_macro_steps=20,
                planning_horizon=5, actions_per_plan=5, population=300,
                num_elites=30, cem_iterations=10, candidate_batch_size=64,
                min_std=0.05, unbounded_actions=True, max_action_length=3.5)


def ensure_identity(path, expected):
    """Create an immutable run identity or reject a changed input on resume."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"planning identity differs: {path}")
    else:
        write_json(path, expected)


def summarize(records, layouts, seeds, *, bootstrap_samples=10000):
    """Pair outcomes by layout and seed; resample seed clusters across layouts."""
    if bootstrap_samples < 1 or not layouts or not seeds:
        raise ValueError("nonempty cases and positive bootstrap count required")
    indexed = {}
    for row in records:
        key = (row["condition"], tuple(row["layout"]), row["seed"])
        if key in indexed:
            raise ValueError(f"duplicate planning case: {key}")
        indexed[key] = row
    wanted = {(condition, tuple(layout), seed) for condition in CONDITIONS
              for layout in layouts for seed in seeds}
    if set(indexed) != wanted:
        raise ValueError("planning cases are incomplete or mismatched")
    layout_keys = [tuple(layout) for layout in layouts]
    differences = np.empty((len(seeds), len(layout_keys)), dtype=np.int8)
    for i, seed in enumerate(seeds):
        for j, layout in enumerate(layout_keys):
            fixed = indexed["fixed", layout, seed]
            multi = indexed["multi", layout, seed]
            if (not np.allclose(fixed["achieved_positions"][0], multi["achieved_positions"][0], atol=1e-6)
                    or not np.allclose(fixed["goal_position"], multi["goal_position"], atol=1e-6)):
                raise ValueError(f"paired start or goal differs for {layout}, seed {seed}")
            differences[i, j] = int(bool(multi["success"])) - int(bool(fixed["success"]))

    def stats(rows):
        return dict(episodes=len(rows), successes=sum(bool(row["success"]) for row in rows),
                    success_rate=float(np.mean([row["success"] for row in rows])),
                    mean_final_distance=float(np.mean([row["final_goal_distance"] for row in rows])))

    overall = {condition: stats([indexed[condition, layout, seed] for layout in layout_keys
                                 for seed in seeds]) for condition in CONDITIONS}
    by_layout = {f"{layout[0]},{layout[1]}": {
        condition: stats([indexed[condition, layout, seed] for seed in seeds])
        for condition in CONDITIONS} for layout in layout_keys}
    rng = np.random.default_rng(20261110)
    sample_ids = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    bootstrap = differences[sample_ids].mean(axis=(1, 2))
    paired = dict(episodes=int(differences.size), seed_clusters=len(seeds),
                  difference=float(differences.mean()),
                  multi_only=int((differences == 1).sum()),
                  fixed_only=int((differences == -1).sum()),
                  bootstrap_95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                  bootstrap_seed=20261110, bootstrap_samples=bootstrap_samples)
    return dict(overall=overall, by_layout=by_layout, paired=paired)


def _checked_inputs(wall_run, adapter_run, encoder_checkpoint, world_model_checkpoint, dino_wm_root):
    wall_run, adapter_run = Path(wall_run), Path(adapter_run)
    manifest = json.loads((wall_run / "multi/episode_manifest.json").read_text())
    adapter_manifest = json.loads((adapter_run / "layout_goal_manifest.json").read_text())
    complete = json.loads((wall_run / "multi/complete.json").read_text())
    encoder_sha = file_sha256(encoder_checkpoint)
    model_sha = file_sha256(world_model_checkpoint)
    if (adapter_manifest["runtime"]["encoder_sha256"] != encoder_sha
            or complete["identity"]["encoder_sha256"] != encoder_sha
            or complete["best_sha256"] != model_sha
            or complete["identity"]["manifest_sha256"] != manifest_digest(manifest)
            or adapter_manifest["train_layouts"] != manifest["train_layouts"]
            or adapter_manifest["unseen_layouts"] != manifest["test_layouts"]):
        raise ValueError("world-model and adapter source identities or layout splits differ")
    if not (Path(dino_wm_root) / "env/wall/envs/wall.py").is_file():
        raise FileNotFoundError("DotWall renderer checkout is missing")
    payload = torch.load(world_model_checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint_manifest(payload["metadata"], manifest)
    checkpoints, adapters = {}, {}
    for condition in CONDITIONS:
        path = adapter_run / condition / "band_08.pt"
        adapter, saved = load_adapter(path, "cpu")
        if saved["schema_version"] != 1 or saved["dim"] != 1280 or saved["provenance"]["band"] != 8:
            raise ValueError(f"unexpected {condition} adapter checkpoint")
        checkpoints[condition] = file_sha256(path)
        adapters[condition] = adapter
    if checkpoints["fixed"] == checkpoints["multi"]:
        raise ValueError("fixed and multi adapter checkpoints are identical")
    config = load_environment_config("wall", wall_run / "multi/environment.json")
    config = replace(config, dino_wm_root=str(Path(dino_wm_root).resolve()))
    identity = dict(kind="wall_multi_world_adapter_planning_v1", model_sha256=model_sha,
                    model_manifest_sha256=manifest_digest(manifest), encoder_sha256=encoder_sha,
                    adapter_sha256=checkpoints, adapter_run=str(adapter_run),
                    wall_run=str(wall_run), renderer_commit=adapter_manifest["runtime"]["renderer_commit"],
                    source_sha256=file_sha256(Path(__file__)), planning=PLANNING,
                    layouts=adapter_manifest["unseen_layouts"], base_seed=10000,
                    seeds_per_layout=50)
    return config, identity, adapters


@torch.no_grad()
def _run_case(env, config, encoder, model, metadata, metric, condition, layout, seed,
              device, precision):
    initial = env.reset(seed, layout=layout)
    goal_tokens = visual.encode_frame(encoder, env.render_goal(), device, precision).float()
    scorer = adapter_goal_cost(metric, goal_tokens)
    args = SimpleNamespace(**PLANNING)
    record = plan_proprio.run_episode(env, config, encoder, model, metadata, args,
                                      device, precision, seed, None, initial=initial,
                                      latent_cost_fn=scorer)
    record.update(condition=condition, layout=list(layout))
    return record


def run(wall_run, adapter_run, encoder_checkpoint, world_model_checkpoint, dino_wm_root,
        output, *, smoke=False, device="cuda", precision="fp32"):
    config, source_identity, adapters = _checked_inputs(
        wall_run, adapter_run, encoder_checkpoint, world_model_checkpoint, dino_wm_root)
    layouts = [tuple(x) for x in source_identity["layouts"][:1 if smoke else None]]
    seeds = list(range(10000, 10001 if smoke else 10050))
    output = Path(output) / ("smoke" if smoke else "full")
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(source_identity, mode="smoke" if smoke else "full",
                    selected_layouts=[list(x) for x in layouts], selected_seeds=seeds,
                    device=device, precision=precision)
    ensure_identity(output / "identity.json", identity)
    resolved_device = wm.resolve_device(device)
    resolved_precision = wm.resolve_precision(precision, resolved_device)
    model, metadata, _ = wm_proprio.load_model_and_metadata(world_model_checkpoint, resolved_device)
    model.eval().requires_grad_(False)
    if metadata["model_configuration"]["num_patches"] != 256 or metadata["model_configuration"]["ijepa_dim"] != 1280:
        raise ValueError("world model does not predict I-JEPA 256x1280 tokens")
    encoder = visual.load_frozen_ijepa_encoder(encoder_checkpoint, resolved_device)
    adapters = {name: metric.to(resolved_device).eval().requires_grad_(False)
                for name, metric in adapters.items()}
    records = []
    for layout in layouts:
        for seed in seeds:
            for condition in CONDITIONS:
                path = output / condition / f"wall{layout[0]}_door{layout[1]}_seed{seed}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    if (record.get("condition") != condition or record.get("layout") != list(layout)
                            or record.get("seed") != seed):
                        raise ValueError(f"resumed case identity differs: {path}")
                else:
                    env = create_adapter(config, PLANNING["max_macro_steps"] * config.actions_per_model_step)
                    try:
                        record = _run_case(env, config, encoder, model, metadata, adapters[condition],
                                           condition, layout, seed, resolved_device, resolved_precision)
                    finally:
                        env.close()
                    write_json(path, record)
                records.append(record)
                print(f"{condition} wall={layout[0]} door={layout[1]} seed={seed} "
                      f"success={record['success']} final_d={record['final_goal_distance']:.2f}", flush=True)
    summary = summarize(records, layouts, seeds)
    write_json(output / "summary.json", summary)
    print("Summary:", output / "summary.json", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wall-run", "adapter-run", "encoder-checkpoint", "world-model-checkpoint",
                 "dino-wm-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "auto", "bf16"), default="fp32")
    args = parser.parse_args(argv)
    run(args.wall_run, args.adapter_run, args.encoder_checkpoint, args.world_model_checkpoint,
        args.dino_wm_root, args.output, smoke=args.smoke, device=args.device,
        precision=args.precision)


if __name__ == "__main__":
    main()
