"""Evaluate three frozen planning costs on one adjacent band of held-out starts."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from src.mylonas_ijepa_wm import curriculum_clean_pair_ablation as clean_runner
from src.mylonas_ijepa_wm.curriculum_protocol import digest, load_stage, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256, interaction_budget,
)


CONDITIONS = ("raw", "original", "clean")
LABELS = {"raw": "Raw cost", "original": "Original adapter", "clean": "Clean-pair adapter"}
COLORS = {"raw": "#475569", "original": "#dc2626", "clean": "#16a34a"}


def _read(path):
    return json.loads(Path(path).read_text())


def _paths(parent, clean, output):
    parent, clean, output = (Path(path).resolve() for path in (parent, clean, output))
    for source in (parent, clean):
        if output == source or source in output.parents:
            raise ValueError("evaluation output must be outside both source directories")
    # Include dangling links: a journal or plot may create its linked target.
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("evaluation output must not contain descendant symlinks")
    return parent, clean, output


def _condition_root(output, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown evaluation condition")
    output = Path(output).resolve()
    root = (output / "conditions" / condition).resolve()
    if output not in root.parents:
        raise ValueError("condition directory must remain inside evaluation output")
    return root


def _source_snapshot(parent, clean):
    config, manifest = clean_runner._config(parent, clean)
    clean_runner._validate_parent_snapshot(parent, config)
    clean_runtime = _read(clean / "runtime.json")["runtime_id"]
    clean_runner._child_runtime(clean, clean_runtime)
    clean_runner._load_completed_training(clean, config, clean_runtime, "cpu")
    load_adapter(parent / "run_0/adapter.pt", "cpu", expected_runtime_id=config["parent_runtime_id"])
    common_files = ("manifest.json", "runtime.json", "run_0/adapter.pt",
                    "run_0/training.json", "stages/run_0/train.json")
    return dict(
        models=config["parent_models"],
        original=dict(directory=str(parent), runtime_id=config["parent_runtime_id"],
                      adapter_sha256=config["parent_adapter_sha256"],
                      files={name: file_sha256(parent / name) for name in common_files}),
        clean=dict(directory=str(clean), runtime_id=clean_runtime,
                   adapter_sha256=file_sha256(clean / "run_0/adapter.pt"),
                   files={name: file_sha256(clean / name) for name in
                          (*common_files, "ablation.json", "run_0/goal_tokens.pt")}),
    ), manifest


def _manifest(parent_manifest):
    rng = np.random.default_rng(20261105)
    cases = [dict(case_id=f"eval_adjacent_{i:03d}", split="eval", band="adjacent",
                  start=[float(rng.uniform(40, 54)), float(rng.uniform(22, 30))],
                  goal=[50., 42.], layout=[32, 30], seed=int(rng.integers(0, 2**31 - 1)),
                  run_seed=None) for i in range(20)]
    return dict(schema_version=1, seed=20261105, protocol_variant="adjacent_band_evaluation",
                evaluation=cases, training={}, planning=deepcopy(parent_manifest["planning"]),
                fitting=deepcopy(parent_manifest["fitting"]),
                bands=dict(near=dict(x=[40, 54], y=[30, 36]), adjacent=dict(x=[40, 54], y=[22, 30])))


def prepare_evaluation(parent, clean, output):
    parent, clean, output = _paths(parent, clean, output)
    sources, parent_manifest = _source_snapshot(parent, clean)
    manifest = _manifest(parent_manifest)
    config = dict(schema_version=1, sources=sources, manifest_digest=digest(manifest),
                  conditions=list(CONDITIONS), raw_required_successes=18, max_paired_losses=2)
    # Check both before writing either, including on a partially prepared run.
    for name, value in (("manifest.json", manifest), ("experiment.json", config)):
        path = output / name
        if path.exists() and _read(path) != value:
            raise ValueError(f"existing {name} differs; use a new output directory")
    for name, value in (("manifest.json", manifest), ("experiment.json", config)):
        if not (output / name).exists():
            write_json(output / name, value)
    return config


def _validated(parent, clean, output):
    parent, clean, output = _paths(parent, clean, output)
    config, manifest = _read(output / "experiment.json"), _read(output / "manifest.json")
    sources, parent_manifest = _source_snapshot(parent, clean)
    if config.get("sources") != sources:
        raise ValueError("frozen source artifacts changed after evaluation prepare")
    if (manifest != _manifest(parent_manifest) or config.get("manifest_digest") != digest(manifest)
            or config.get("conditions") != list(CONDITIONS)
            or config.get("raw_required_successes") != 18 or config.get("max_paired_losses") != 2):
        raise ValueError("adjacent evaluation manifest or configuration differs")
    return config, manifest


def runtime_variant(config, condition):
    return dict(kind="adjacent_band_evaluation", experiment_digest=digest(config), condition=condition)


def _runtime(output, config, manifest, condition, runtime_id=None):
    root = _condition_root(output, condition)
    saved = _read(root / "runtime.json")
    signature = saved.get("signature", {})
    if (saved.get("runtime_id") != digest(signature)
            or (runtime_id is not None and saved["runtime_id"] != runtime_id)
            or signature.get("manifest") != digest(manifest)
            or signature.get("models") != config["sources"]["models"]
            or signature.get("runtime_variant") != runtime_variant(config, condition)):
        raise ValueError("evaluation runtime provenance differs")
    return root, saved


def _stage(condition):
    return "baseline" if condition == "raw" else "run_0/evaluate"


def _condition(config, condition):
    source = config["sources"].get(condition)
    return dict(cost=condition, adapter_sha256=source["adapter_sha256"] if source else None,
                training_runtime_id=source["runtime_id"] if source else None,
                sources_digest=digest(config["sources"]))


def _preflight_records(root, manifest, runtime_id, condition, identity):
    stage = _stage(condition)
    status_path = root / "stages" / f"{stage}.json"
    signature = dict(runtime_id=runtime_id, cases_digest=digest(manifest["evaluation"]), condition=identity)
    if status_path.exists():
        prior = _read(status_path)
        if any(prior.get(key) != value for key, value in signature.items()):
            raise ValueError("cached stage provenance differs")
    folder = root / "episodes" / stage
    expected = {f"{case['case_id']}.json" for case in manifest["evaluation"]}
    if any(path.name not in expected for path in folder.glob("*.json")):
        raise ValueError("cached case provenance differs")
    for case in manifest["evaluation"]:
        path = folder / f"{case['case_id']}.json"
        if not path.exists():
            continue
        record = _read(path)
        if (record.get("case") != case or record.get("case_id") != case["case_id"]
                or record.get("runtime_id") != runtime_id or record.get("condition") != identity
                or record.get("raw_path")):
            raise ValueError("cached episode provenance differs")
    interaction_budget(root, runtime_id)


def evaluate_condition(parent, clean, output, condition, runtime):
    config, manifest = _validated(parent, clean, output)
    root, _ = _runtime(output, config, manifest, condition, runtime.runtime_id)
    if Path(runtime.root).resolve() != root or runtime.manifest != manifest:
        raise ValueError("evaluation runtime root or manifest differs")
    identity = _condition(config, condition)
    _preflight_records(root, manifest, runtime.runtime_id, condition, identity)
    metric = None
    if condition != "raw":
        source = config["sources"][condition]
        metric, _ = load_adapter(Path(source["directory"]) / "run_0/adapter.pt", runtime.device,
                                 expected_runtime_id=source["runtime_id"])
    # All conditions share exactly the frozen target used for clean training.
    goal = torch.load(Path(clean) / "run_0/goal_tokens.pt", map_location="cpu", weights_only=True)
    runtime.goal = goal["tokens"].detach().to(runtime.device)
    target = runtime.goal.clone()

    @torch.no_grad()
    def raw_score(tokens):
        return (tokens - target.unsqueeze(0)).square().mean(dim=(1, 2))

    return run_stage(root, _stage(condition), manifest["evaluation"], runtime.runtime_id,
                     lambda case: runtime.run_case(case, metric=metric,
                         **({"latent_cost_fn": raw_score} if condition == "raw" else {})),
                     condition=identity)


def _comparison(raw, adapted, competent):
    original = {row["case_id"]: row for row in raw}
    learned = {row["case_id"]: row for row in adapted}
    if (len(original) != len(raw) or len(learned) != len(adapted) or original.keys() != learned.keys()
            or any(original[key]["case"] != learned[key]["case"] for key in original)):
        raise ValueError("paired evaluation requires identical unique cases and CEM seeds")
    wins = [key for key in original if not original[key]["success"] and learned[key]["success"]]
    losses = [key for key in original if original[key]["success"] and not learned[key]["success"]]
    return dict(wins=len(wins), losses=len(losses), won_case_ids=wins, lost_case_ids=losses,
                success_change=len(wins) - len(losses),
                retention_passed=len(losses) <= 2 if competent else None,
                interpretation="Fixed-case operational screen; inspect distances and actions even at 20/20.")


def _plot(output, manifest, records, report):
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    labels = [LABELS[key] for key in CONDITIONS]
    for axis, field, title in zip(axes, ("successes", "mean_final_distance", "primitive_steps"),
                                  ("Successes / 20", "Mean final goal distance", "Total primitive actions")):
        axis.bar(labels, [report["conditions"][key][field] for key in CONDITIONS],
                 color=[COLORS[key] for key in CONDITIONS])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
        if field == "successes":
            axis.set_ylim(0, 20)
    figure.savefig(Path(output) / "success_comparison.png", dpi=140)
    plt.close(figure)
    # Each held-out case gets a separate panel, avoiding an unreadable path overlay.
    figure, axes = plt.subplots(4, 5, figsize=(16, 14), constrained_layout=True)
    for index, (axis, case) in enumerate(zip(axes.flat, manifest["evaluation"])):
        axis.add_patch(Rectangle((40, 22), 14, 8, color="#cbd5e1", alpha=.25))
        axis.add_patch(Rectangle((40, 30), 14, 6, color="#93c5fd", alpha=.18))
        axis.axhline(30, color="#64748b", linestyle="--", linewidth=.6)
        axis.add_patch(Circle((50, 42), 4.5, fill=False, color="#64748b", linewidth=.6))
        axis.scatter(*case["goal"], marker="*", color="black", s=35, zorder=4)
        axis.scatter(*case["start"], marker="x", color="black", s=20, zorder=4)
        outcomes = []
        for key in CONDITIONS:
            record = records[key][index]
            path = np.asarray(record["achieved_positions"])
            axis.plot(path[:, 0], path[:, 1], color=COLORS[key], alpha=.8, linewidth=1,
                      label=LABELS[key])
            axis.scatter(*path[-1], color=COLORS[key], s=12)
            outcomes.append(f"{key[0].upper()}: {'S' if record['success'] else 'F'}")
        axis.set_title(f"case {index:03d}\n" + " | ".join(outcomes), fontsize=9)
        axis.set(xlim=(32, 64), ylim=(0, 64), aspect="equal")
        axis.tick_params(labelsize=7)
        if index == 0:
            axis.legend(fontsize=6, loc="lower left")
    figure.suptitle("Adjacent starts (gray), near band (blue), goal star; macro-boundary trajectories", fontsize=14)
    figure.savefig(Path(output) / "trajectories.png", dpi=140)
    plt.close(figure)


def make_report(parent, clean, output, *, render=True):
    config, manifest = _validated(parent, clean, output)
    records, summaries, signatures = {}, {}, []
    for condition in CONDITIONS:
        root, runtime = _runtime(output, config, manifest, condition)
        _preflight_records(root, manifest, runtime["runtime_id"], condition, _condition(config, condition))
        signature = dict(runtime["signature"])
        signature.pop("runtime_variant")
        signatures.append(signature)
        rows = load_stage(root, _stage(condition), manifest["evaluation"], runtime["runtime_id"])
        if any(row.get("condition") != _condition(config, condition) for row in rows):
            raise ValueError("evaluation condition provenance differs")
        records[condition] = rows
        summaries[condition] = dict(
            successes=sum(bool(row["success"]) for row in rows), episodes=len(rows),
            mean_final_distance=float(np.mean([row["final_goal_distance"] for row in rows])),
            primitive_steps=sum(row["micro_steps"] for row in rows),
            failed_case_ids=[row["case_id"] for row in rows if not row["success"]],
            runtime_id=runtime["runtime_id"], journal=interaction_budget(root, runtime["runtime_id"]),
        )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("evaluation runtimes use different model, source or execution settings")
    competent = summaries["raw"]["successes"] >= 18
    report = dict(schema_version=1, experiment=config, band=manifest["bands"]["adjacent"],
                  conditions=summaries,
                  raw_competence=dict(passed=competent, required_successes=18,
                                      note="Interpret transfer only when raw achieves at least 18/20."),
                  comparisons={condition: _comparison(records["raw"], records[condition], competent)
                               for condition in ("original", "clean")},
                  per_case=[dict(case=case, results={key: dict(success=records[key][i]["success"],
                            final_goal_distance=records[key][i]["final_goal_distance"],
                            primitive_steps=records[key][i]["micro_steps"]) for key in CONDITIONS})
                            for i, case in enumerate(manifest["evaluation"])],
                  note="New evaluation starts may overlap states visited during near training. This tests adjacent-band performance, not guaranteed unseen-state generalization.")
    write_json(Path(output) / "report.json", report)
    if render:
        _plot(output, manifest, records, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "evaluate", "report"])
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--dino-wm-root", type=Path)
    parser.add_argument("--expected-world-model-sha256", default=EXPECTED_WM)
    parser.add_argument("--expected-encoder-sha256", default=EXPECTED_ENCODER)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="fp32")
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.set_defaults(run_seed=0, videos=False)
    args = parser.parse_args(argv)
    parent, clean, output = _paths(args.parent_dir, args.clean_dir, args.output_dir)
    if args.stage == "prepare":
        result = prepare_evaluation(parent, clean, output)
    elif args.stage == "report":
        result = make_report(parent, clean, output)
    else:
        if args.condition is None or any(getattr(args, name) is None for name in
                ("world_model_checkpoint", "encoder_checkpoint", "dino_wm_root")):
            raise ValueError("condition and frozen model/encoder/DINO paths are required")
        config, manifest = _validated(parent, clean, output)
        args.output_dir = _condition_root(output, args.condition)
        args.runtime_signature_extra = runtime_variant(config, args.condition)
        runtime = PilotRuntime(args, manifest)
        try:
            return evaluate_condition(parent, clean, output, args.condition, runtime)
        finally:
            runtime.close()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"Adjacent evaluation stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
