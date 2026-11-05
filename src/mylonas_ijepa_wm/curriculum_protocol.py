"""Fixed pilot cases, resumable records and paired evaluation summaries."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from src.mylonas_ijepa_wm.plan_pointmaze import _write_json_atomically


PLANNING = dict(planning_horizon=5, actions_per_plan=5, population=300, num_elites=30,
                cem_iterations=10, candidate_batch_size=64, min_std=0.05,
                cem_selection="elite_mean", unbounded_actions=True,
                max_action_length=3.5, max_macro_steps=20)
TRAINING = dict(updates=500, batch_size=16, lr=1e-4, bottleneck=64, anchor_weight=1.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomically(value, path)


def make_manifest(eval_per_band=20, train_per_run=40, run_seeds=(0, 1, 2), seed=20261105):
    if min(eval_per_band, train_per_run) < 1 or not run_seeds or len(set(run_seeds)) != len(run_seeds):
        raise ValueError("positive case counts and unique run seeds required")
    rng = np.random.default_rng(seed)
    evaluation = []

    def case(split, band, index, run_seed=None):
        bounds = (30., 36.) if band == "near" else (14., 24.)
        for _ in range(10000):
            start = [float(rng.uniform(40, 54)), float(rng.uniform(*bounds))]
            if split != "train" or all(np.linalg.norm(np.asarray(start) - item["start"]) >= .25 for item in evaluation):
                break
        else:
            raise ValueError("cannot sample disjoint training starts")
        prefix = "eval" if split == "eval" else f"train_{run_seed}"
        return dict(case_id=f"{prefix}_{band}_{index:03d}", split=split, band=band,
                    start=start, goal=[50., 42.], layout=[32, 30],
                    seed=int(rng.integers(0, 2**31 - 1)), run_seed=run_seed)

    for band in ("near", "far"):
        evaluation.extend(case("eval", band, index) for index in range(eval_per_band))
    training = {str(run_seed): [case("train", "near", i, run_seed) for i in range(train_per_run)]
                for run_seed in run_seeds}
    return dict(schema_version=1, seed=seed, evaluation=evaluation, training=training,
                planning=dict(PLANNING), fitting=dict(TRAINING),
                privileges="state resets, success detection, revisit filtering; proprioceptive dynamics",
                protocol_variant="agreed" if (eval_per_band, train_per_run, tuple(run_seeds)) == (20, 40, (0, 1, 2)) else "modified")


def run_stage(root, stage, cases, runtime_id, execute, *, condition=None):
    """Persist each finished case; interrupted attempts have no committed record."""
    root = Path(root)
    stage_file = root / "stages" / f"{stage}.json"
    signature = dict(runtime_id=runtime_id, cases_digest=digest(cases), condition=condition)
    if stage_file.exists():
        prior = json.loads(stage_file.read_text())
        if any(prior.get(key) != value for key, value in signature.items()):
            raise ValueError(f"stage provenance mismatch: {stage}")
    write_json(stage_file, {**signature, "status": "running"})
    records = []
    for index, case in enumerate(cases):
        path = root / "episodes" / stage / f"{case['case_id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if (record.get("runtime_id") != runtime_id or record.get("case") != case
                    or record.get("condition") != condition):
                raise ValueError(f"episode provenance mismatch: {path}")
            if record.get("raw_path") and not (root / record["raw_path"]).is_file():
                raise ValueError(f"missing raw companion: {path}")
        else:
            record = execute(case)
            record.update(case=case, case_id=case["case_id"], runtime_id=runtime_id, condition=condition)
            write_json(path, record)
        records.append(record)
        print(f"{stage} {index + 1}/{len(cases)} {case['case_id']} success={record['success']} steps={record['micro_steps']}", flush=True)
    write_json(stage_file, {**signature, "status": "complete", "episodes": len(records),
                            "primitive_interactions": sum(item["micro_steps"] for item in records)})
    return records


def load_stage(root, stage, cases, runtime_id):
    root = Path(root)
    status = json.loads((root / "stages" / f"{stage}.json").read_text())
    if (status.get("status") != "complete" or status.get("runtime_id") != runtime_id
            or status.get("cases_digest") != digest(cases)):
        raise ValueError(f"stage incomplete or provenance mismatch: {stage}")
    records = []
    for case in cases:
        record = json.loads((root / "episodes" / stage / f"{case['case_id']}.json").read_text())
        if (record.get("case") != case or record.get("runtime_id") != runtime_id
                or record.get("condition") != status.get("condition")):
            raise ValueError(f"episode provenance mismatch: {case['case_id']}")
        records.append(record)
    return records


def competence_gate(records):
    near = [item for item in records if item["case"]["band"] == "near"]
    successes = sum(bool(item["success"]) for item in near)
    required = math.ceil(.6 * len(near))
    return dict(passed=bool(near) and successes >= required, successes=successes,
                episodes=len(near), required=required, threshold_note="operational collection gate, not significance")


def _wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1-p) / total + z*z/(4*total*total)) / (1+z*z/total)
    return [center - half, center + half]


def paired_report(raw, adapted):
    original = {item["case_id"]: item for item in raw}
    learned = {item["case_id"]: item for item in adapted}
    if (not original or len(original) != len(raw) or len(learned) != len(adapted)
            or original.keys() != learned.keys()
            or any(original[key]["case"] != learned[key]["case"] for key in original)):
        raise ValueError("paired evaluation requires identical unique cases and CEM seeds")
    report = {}
    for band in ("near", "far"):
        keys = [key for key in original if original[key]["case"]["band"] == band]
        if not keys:
            raise ValueError("paired evaluation needs both bands")
        a = sum(bool(original[key]["success"]) for key in keys)
        b = sum(bool(learned[key]["success"]) for key in keys)
        report[band] = dict(episodes=len(keys), raw_successes=a, adapted_successes=b,
                            success_change=b-a, success_rate_change=(b-a)/len(keys),
                            wins=sum(not original[key]["success"] and learned[key]["success"] for key in keys),
                            losses=sum(original[key]["success"] and not learned[key]["success"] for key in keys),
                            raw_wilson_95=_wilson(a, len(keys)), adapted_wilson_95=_wilson(b, len(keys)))
        for label, records in (("raw", original), ("adapted", learned)):
            report[band][f"{label}_mean_final_distance"] = float(np.mean([records[key]["final_goal_distance"] for key in keys]))
            report[band][f"{label}_primitive_steps"] = sum(records[key]["micro_steps"] for key in keys)
    report["screen_passed"] = (report["far"]["success_change"] >= math.ceil(.2 * report["far"]["episodes"])
                               and report["near"]["losses"] <= math.floor(.1 * report["near"]["episodes"]))
    report["interpretation"] = "Mechanism screen only; fixed-goal tasks reused across run seeds; not independent extra tasks or a full curriculum comparison."
    return report
