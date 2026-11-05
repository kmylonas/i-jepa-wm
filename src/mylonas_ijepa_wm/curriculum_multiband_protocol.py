"""Fixed cases, immutable source references and stage preflight for curricula."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import curriculum_adjacent_band as adjacent
from src.mylonas_ijepa_wm.curriculum_protocol import digest, load_stage, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256, interaction_budget, save_tensors


RECIPES = ("original", "clean")
BANDS = ("near", "band_1", "band_2", "band_3")
BOUNDS = {"near": [30., 36.], "band_1": [22., 30.], "band_2": [14., 22.], "band_3": [8., 14.]}


def read(path):
    return json.loads(Path(path).read_text())


def safe_paths(parent, clean, adjacent_dir, output):
    parent, clean, adjacent_dir, output = (Path(path).resolve() for path in (parent, clean, adjacent_dir, output))
    if any(output == source or source in output.parents for source in (parent, clean, adjacent_dir)):
        raise ValueError("curriculum output must be outside all source directories")
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("curriculum output cannot contain descendant symlinks")
    return parent, clean, adjacent_dir, output


def source_snapshot(parent, clean, adjacent_dir):
    # Read-only checks; do not invoke the adjacent report writer on a source run.
    config, adjacent_manifest = adjacent._validated(parent, clean, adjacent_dir)
    paths = ["manifest.json", "experiment.json", "report.json"]
    for condition in adjacent.CONDITIONS:
        root, runtime = adjacent._runtime(adjacent_dir, config, adjacent_manifest, condition)
        stage = adjacent._stage(condition)
        records = load_stage(root, stage, adjacent_manifest["evaluation"], runtime["runtime_id"])
        if any(row["condition"] != adjacent._condition(config, condition) for row in records):
            raise ValueError("adjacent source condition differs")
        paths.append((root / "runtime.json").relative_to(adjacent_dir).as_posix())
        paths.append((root / "stages" / f"{stage}.json").relative_to(adjacent_dir).as_posix())
        paths.extend((root / "episodes" / stage / f"{case['case_id']}.json").relative_to(adjacent_dir).as_posix()
                     for case in adjacent_manifest["evaluation"])
    sources = deepcopy(config["sources"])
    sources["adjacent"] = dict(directory=str(adjacent_dir),
                              files={name: file_sha256(adjacent_dir / name) for name in paths})
    return sources, read(parent / "manifest.json"), adjacent_manifest


def make_manifest(parent, adjacent_manifest):
    evaluation = {"near": [deepcopy(case) for case in parent["evaluation"] if case["band"] == "near"],
                  "band_1": deepcopy(adjacent_manifest["evaluation"])}
    if any(len(rows) != 20 for rows in evaluation.values()):
        raise ValueError("curriculum requires 20 source near and adjacent evaluation cases")
    rng = np.random.default_rng(20261106)

    def sample(split, band, index):
        return dict(case_id=f"{split}_curriculum_{band}_{index:03d}", split=split, band=band,
                    start=[float(rng.uniform(40, 54)), float(rng.uniform(*BOUNDS[band]))],
                    goal=[50., 42.], layout=[32, 30], seed=int(rng.integers(0, 2**31 - 1)),
                    run_seed=0 if split == "train" else None)

    for band in BANDS[2:]:
        evaluation[band] = [sample("eval", band, index) for index in range(20)]
    eval_points = np.array([case["start"] for rows in evaluation.values() for case in rows])
    collection = {}
    for band in BANDS[1:]:
        rows = []
        for index in range(40):
            for _ in range(10000):
                case = sample("train", band, index)
                if np.linalg.norm(eval_points - np.asarray(case["start"]), axis=1).min() >= .25:
                    rows.append(case)
                    break
            else:
                raise ValueError("cannot sample disjoint curriculum training starts")
        collection[band] = rows
    all_cases = [case for rows in (*evaluation.values(), *collection.values()) for case in rows]
    if (len({case["case_id"] for case in all_cases}) != len(all_cases)
            or any(case["goal"] != [50., 42.] or case["layout"] != [32, 30] for case in all_cases)):
        raise ValueError("source goal/layout or case identity differs from curriculum")
    return dict(schema_version=1, seed=20261106, evaluation=evaluation, collection=collection,
                planning=deepcopy(parent["planning"]), fitting=deepcopy(parent["fitting"]),
                bounds=deepcopy(BOUNDS), x_range=[40, 54], recipes=list(RECIPES))


def prepare(parent, clean, adjacent_dir, output):
    parent, clean, adjacent_dir, output = safe_paths(parent, clean, adjacent_dir, output)
    sources, parent_manifest, adjacent_manifest = source_snapshot(parent, clean, adjacent_dir)
    manifest = make_manifest(parent_manifest, adjacent_manifest)
    config = dict(schema_version=1, sources=sources, manifest_digest=digest(manifest),
                  pair_rules=dict(original=None, clean=.5), optimizer="fresh_adam_per_band",
                  sampling="uniform_band_then_episode_then_pair", goal_source="clean/run_0/goal_tokens.pt")
    for name, value in (("manifest.json", manifest), ("experiment.json", config)):
        path = output / name
        if path.exists() and read(path) != value:
            raise ValueError("existing curriculum configuration differs; use a new output directory")
    for name, value in (("manifest.json", manifest), ("experiment.json", config)):
        if not (output / name).exists():
            write_json(output / name, value)
    goal_path, goal_meta = output / "cache/goal.pt", output / "cache/goal.json"
    if not goal_path.exists() and not goal_meta.exists():
        source = torch.load(clean / "run_0/goal_tokens.pt", map_location="cpu", weights_only=True)
        save_tensors(goal_path, dict(tokens=source["tokens"].detach().cpu().float(), experiment_digest=digest(config)))
        write_json(goal_meta, dict(sha256=file_sha256(goal_path), experiment_digest=digest(config)))
    load_goal(output, config)
    return config


def load_goal(root, config):
    root = Path(root)
    meta, path = read(root / "cache/goal.json"), root / "cache/goal.pt"
    if meta.get("experiment_digest") != digest(config) or meta.get("sha256") != file_sha256(path):
        raise ValueError("curriculum goal cache hash/provenance differs")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    tokens = saved.get("tokens")
    if (saved.get("experiment_digest") != digest(config) or not isinstance(tokens, torch.Tensor)
            or tokens.ndim != 2 or not torch.isfinite(tokens).all()):
        raise ValueError("curriculum goal tokens differ")
    return tokens, meta["sha256"]


def load(parent, clean, adjacent_dir, output):
    parent, clean, adjacent_dir, output = safe_paths(parent, clean, adjacent_dir, output)
    config, manifest = read(output / "experiment.json"), read(output / "manifest.json")
    sources, pm, am = source_snapshot(parent, clean, adjacent_dir)
    expected = dict(schema_version=1, sources=sources, manifest_digest=digest(make_manifest(pm, am)),
                    pair_rules=dict(original=None, clean=.5), optimizer="fresh_adam_per_band",
                    sampling="uniform_band_then_episode_then_pair", goal_source="clean/run_0/goal_tokens.pt")
    if config != expected or manifest != make_manifest(pm, am):
        raise ValueError("curriculum source/configuration changed after prepare")
    load_goal(output, config)
    return config, manifest


def variant(config, recipe):
    return dict(kind="multiband_curriculum", experiment_digest=digest(config), recipe=recipe)


def runtime_root(root, recipe):
    if recipe not in (*RECIPES, "raw"):
        raise ValueError("unknown curriculum recipe")
    return Path(root) / ("raw" if recipe == "raw" else f"branches/{recipe}")


def check_runtime(root, recipe, config, manifest, runtime_id=None):
    saved = read(runtime_root(root, recipe) / "runtime.json")
    signature = saved.get("signature", {})
    if (saved.get("runtime_id") != digest(signature)
            or (runtime_id is not None and saved["runtime_id"] != runtime_id)
            or signature.get("manifest") != digest(manifest)
            or signature.get("models") != config["sources"]["models"]
            or signature.get("runtime_variant") != variant(config, recipe)):
        raise ValueError("curriculum execution runtime provenance differs")
    execution = {key: value for key, value in signature.items() if key != "runtime_variant"}
    path = Path(root) / "execution.json"
    if path.exists() and read(path) != execution:
        raise ValueError("curriculum execution settings changed across recipes/stages")
    if not path.exists():
        write_json(path, execution)
    return saved["runtime_id"]


def validated_budget(root, runtime_id):
    """Reconcile every committed episode with its complete journal attempt.

    Uncommitted interrupted attempts remain counted; a truncated crash line can
    only make their count a lower bound, never excuse a missing committed action.
    """
    root = Path(root)
    try:
        budget = interaction_budget(root, runtime_id)
    except (KeyError, TypeError) as error:
        raise ValueError("interaction journal entry is incomplete") from error
    attempts = {}
    path = root / "interactions.jsonl"
    for line in path.read_text().splitlines() if path.exists() else ():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # interaction_budget exposes these as a lower bound.
        try:
            identity = (entry["stage"], entry["case_id"])
            attempt_id, event = entry["attempt_id"], entry["event"]
            if event == "start":
                if attempt_id in attempts:
                    raise ValueError("duplicate journal attempt")
                attempts[attempt_id] = dict(identity=identity, steps=[], finish=None)
                continue
            attempt = attempts[attempt_id]
            if attempt["identity"] != identity or attempt["finish"] is not None:
                raise ValueError("journal attempt identity/order differs")
            if event == "primitive":
                attempt["steps"].append(entry["step"])
            elif event == "finish" and entry["status"] in ("complete", "interrupted"):
                attempt["finish"] = entry["status"]
            else:
                raise ValueError("unknown journal event")
        except (KeyError, TypeError) as error:
            raise ValueError("interaction journal attempt is incomplete") from error
    used = set()
    for path in (root / "episodes").rglob("*.json"):
        record = read(path)
        attempt_id = record.get("attempt_id")
        attempt = attempts.get(attempt_id)
        stage = path.parent.relative_to(root / "episodes").as_posix()
        if (attempt is None or attempt_id in used or record.get("runtime_id") != runtime_id
                or attempt["identity"] != (stage, record.get("case_id"))
                or attempt["finish"] != "complete"
                or attempt["steps"] != list(range(1, record["micro_steps"] + 1))):
            raise ValueError("interaction journal does not match committed episode attempt/stage/case/actions")
        used.add(attempt_id)
    budget.setdefault("count_is_lower_bound", False)
    budget["status"] = "validated"
    return budget


def preflight(root, stage, cases, runtime_id, condition, *, capture=False):
    root = Path(root)
    signature = dict(runtime_id=runtime_id, cases_digest=digest(cases), condition=condition)
    stage_path = root / "stages" / f"{stage}.json"
    if stage_path.exists() and any(read(stage_path).get(key) != value for key, value in signature.items()):
        raise ValueError("cached curriculum stage provenance differs")
    folder = root / "episodes" / stage
    expected = {f"{case['case_id']}.json" for case in cases}
    if any(path.name not in expected for path in folder.glob("*.json")):
        raise ValueError("cached curriculum case identity differs")
    for case in cases:
        path = folder / f"{case['case_id']}.json"
        if not path.exists():
            continue
        record = read(path)
        if (record.get("case") != case or record.get("case_id") != case["case_id"]
                or record.get("runtime_id") != runtime_id or record.get("condition") != condition):
            raise ValueError("cached curriculum episode provenance differs")
        if capture:
            expected_path = Path("raw/run_0") / f"{case['case_id']}.pt"
            if (record.get("raw_path") != expected_path.as_posix()
                    or not (root / expected_path).is_file()
                    or record.get("raw_sha256") != file_sha256(root / expected_path)):
                raise ValueError("cached collection raw hash/provenance differs")
        elif record.get("raw_path"):
            raise ValueError("evaluation cannot reference collection data")
    validated_budget(root, runtime_id)


def execute_stage(root, stage, cases, runtime_id, condition, execute, *, capture=False):
    preflight(root, stage, cases, runtime_id, condition, capture=capture)
    return run_stage(root, stage, cases, runtime_id, execute, condition=condition)


def completed_stage(root, stage, cases, runtime_id, condition, *, capture=False):
    preflight(root, stage, cases, runtime_id, condition, capture=capture)
    return load_stage(root, stage, cases, runtime_id)


def optional_stage(root, stage, cases, runtime_id, condition, *, capture=False):
    preflight(root, stage, cases, runtime_id, condition, capture=capture)
    path = Path(root) / "stages" / f"{stage}.json"
    if not path.exists() or read(path).get("status") != "complete":
        return None
    return load_stage(root, stage, cases, runtime_id)
