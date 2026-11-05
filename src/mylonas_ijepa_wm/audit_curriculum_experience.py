"""CPU audit of saved experience; never loads models or executes simulator actions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from src.mylonas_ijepa_wm.curriculum_protocol import digest, load_stage
from src.mylonas_ijepa_wm.curriculum_replay import eligible_pairs


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _fraction(count, total):
    return float(count / total) if total else None


def _reversals(vectors):
    if len(vectors) < 2:
        return None
    valid = (np.linalg.norm(vectors[:-1], axis=1) > 1e-6) & (np.linalg.norm(vectors[1:], axis=1) > 1e-6)
    return _fraction(int(((vectors[:-1] * vectors[1:]).sum(axis=1)[valid] < 0).sum()), int(valid.sum()))


def _pair_stats(pairs, distances, tolerance):
    progress = np.array([distances[i] - distances[j] for i, j in pairs])
    result = dict(count=len(pairs), progress=int((progress > tolerance).sum()),
                  ambiguous=int((np.abs(progress) <= tolerance).sum()),
                  contradictory=int((progress < -tolerance).sum()),
                  strict_contradictory=int((progress < 0).sum()))
    for key in ("progress", "ambiguous", "contradictory", "strict_contradictory"):
        result[key + "_fraction"] = _fraction(result[key], len(pairs))
    return result


def analyze_episode(raw, *, success_radius=4.5, progress_tolerance=.25):
    if not np.isfinite(progress_tolerance) or progress_tolerance < 0 or success_radius <= 0:
        raise ValueError("invalid audit tolerance or success radius")
    states = np.asarray(raw["states"], dtype=np.float64)
    actions = np.asarray(raw["primitive_actions"], dtype=np.float64)
    steps = np.asarray(raw["primitive_steps"])
    if (states.ndim != 2 or states.shape[1] != 2 or not len(states)
            or actions.shape != (len(states) - 1, 2) or steps.shape != (len(states),)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()
            or not np.array_equal(steps, np.arange(len(states)))):
        raise ValueError("raw observations/actions must align with every actual primitive step")
    case = raw["case"]
    goal = np.asarray(case["goal"], dtype=np.float64)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("goal must contain finite x,y coordinates")
    distances = np.linalg.norm(states - goal, axis=1)
    progress = distances[:-1] - distances[1:]
    movements = np.diff(states, axis=0)
    lengths = np.linalg.norm(movements, axis=1)
    path_length = float(lengths.sum())
    shortest = float(max(distances[0] - success_radius, 0))
    retained, pairs = eligible_pairs(raw["states"], raw["primitive_steps"])
    state_tensor = torch.as_tensor(raw["states"], dtype=torch.float32)
    unfiltered = [(i, j) for i in range(len(states)) for j in range(i + 2, len(states))
                  if torch.linalg.vector_norm(state_tensor[i] - state_tensor[j]) >= 2]
    n = len(actions)
    near = (states[:, 0] >= 40) & (states[:, 0] <= 54) & (states[:, 1] >= 30) & (states[:, 1] <= 36)
    far = (states[:, 0] >= 40) & (states[:, 0] <= 54) & (states[:, 1] >= 14) & (states[:, 1] <= 24)
    bad_pairs = sorted([(i, j) for i, j in pairs if distances[j] - distances[i] > progress_tolerance],
                       key=lambda pair: distances[pair[1]] - distances[pair[0]], reverse=True)
    row = dict(case_id=case["case_id"], success=bool(raw["success"]), steps=n, frames=len(states),
               initial_distance=float(distances[0]), final_distance=float(distances[-1]),
               path_length=path_length, shortest_distance_to_success_region=shortest,
               success_path_efficiency=shortest / path_length if raw["success"] and path_length > 0 else None,
               closer_actions=int((progress > progress_tolerance).sum()),
               away_actions=int((progress < -progress_tolerance).sum()),
               neutral_actions=int((np.abs(progress) <= progress_tolerance).sum()),
               strict_away_actions=int((progress < 0).sum()),
               cumulative_distance_increase=float(np.maximum(-progress, 0).sum()),
               maximum_excursion_beyond_start=float(max(distances.max() - distances[0], 0)),
               stationary_actions=int((lengths <= 1e-6).sum()),
               action_reversal_fraction=_reversals(actions), movement_reversal_fraction=_reversals(movements),
               mean_action_change=float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean()) if n > 1 else None,
               far_rectangle_frames=int(far.sum()), outside_near_start_rectangle_frames=int((~near).sum()),
               retained_frames=len(retained), pairs_before=_pair_stats(unfiltered, distances, progress_tolerance),
               pairs_after=_pair_stats(pairs, distances, progress_tolerance))
    for name, count in (("closer", "closer_actions"), ("away", "away_actions"),
                        ("neutral", "neutral_actions"), ("stationary", "stationary_actions")):
        row[name + "_fraction"] = _fraction(row[count], n)
    row["contradictory_examples"] = [dict(indices=[i, j], primitive_steps=[int(steps[i]), int(steps[j])],
                                           distances=[float(distances[i]), float(distances[j])],
                                           distance_increase=float(distances[j] - distances[i])) for i, j in bad_pairs[:3]]
    row["trace"] = dict(states=states.tolist(), actions=actions.tolist(), steps=steps.tolist(),
                         distances=distances.tolist(), retained_indices=retained, goal=goal.tolist(),
                         success_radius=success_radius)
    return row


def summarize_pair_quality(rows, field):
    usable = [row[field] for row in rows if row["success"] and row[field]["count"]]
    total = sum(item["count"] for item in usable)
    result = dict(usable_successful_episodes=len(usable), total_pairs=total, episode_uniform={}, pooled_pairs={})
    for key in ("progress", "ambiguous", "contradictory"):
        result["episode_uniform"][key + "_fraction"] = float(np.mean([item[key] / item["count"] for item in usable])) if usable else None
        result["pooled_pairs"][key + "_fraction"] = _fraction(sum(item[key] for item in usable), total)
    return result


def _load_collection(root, manifest, runtime_id, seed, tolerance):
    records = load_stage(root, f"run_{seed}/collect", manifest["training"][str(seed)], runtime_id)
    rows = []
    for record in records:
        path = (root / record["raw_path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("raw episode path escapes pilot directory")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        if (raw.get("runtime_id") != runtime_id or raw.get("case") != record["case"]
                or raw.get("success") != record["success"]):
            raise ValueError(f"raw episode provenance mismatch: {path.name}")
        count = int(record["micro_steps"])
        frames = raw["frames"]
        durations = raw["macro_durations"]
        if (len(raw["states"]) != count + 1 or len(frames) != count + 1
                or frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8
                or sum(durations) != count or any(not 1 <= duration <= 5 for duration in durations)
                or not np.allclose(np.asarray(raw["states"])[0], record["case"]["start"], atol=1e-4, rtol=0)):
            raise ValueError(f"raw frame/state/action duration alignment mismatch: {path.name}")
        row = analyze_episode(raw, progress_tolerance=tolerance)
        row.update(raw_path=record["raw_path"], mpc_updates=record["mpc_updates"])
        rows.append(row)
    return rows


def _verify_training(root, seed, rows):
    path = root / f"run_{seed}/training.json"
    if not path.exists():
        return False, None
    training = json.loads(path.read_text())
    expected = {row["case_id"]: row for row in rows if row["success"]}
    saved = {item["case_id"]: item for item in training["trajectories"]}
    if saved.keys() != expected.keys():
        raise ValueError("saved training filter episode IDs differ from successful collection")
    for case_id, row in expected.items():
        if (saved[case_id]["frames"] != row["frames"] or saved[case_id]["retained"] != row["retained_frames"]
                or saved[case_id]["eligible_pairs"] != row["pairs_after"]["count"]):
            raise ValueError(f"saved training filter counts differ: {case_id}")
    usable = {key for key, row in expected.items() if row["pairs_after"]["count"]}
    if usable != set(training["training_case_ids"]):
        raise ValueError("saved training filter usable episode IDs differ")
    history = training.get("history", [])
    window = min(50, max(1, len(history) // 2))
    losses = dict(updates_recorded=len(history), averaging_window=window)
    if history:
        for key in ("rank_loss", "anchor_loss", "loss"):
            losses[key] = dict(first_mean=float(np.mean([item[key] for item in history[:window]])),
                               last_mean=float(np.mean([item[key] for item in history[-window:]])))
    return True, losses


def _evaluation_comparison(root, manifest, runtime_id, seed):
    paths = [root / "stages" / f"{name}.json" for name in ("baseline", f"run_{seed}/evaluate")]
    if not all(path.exists() and json.loads(path.read_text()).get("status") == "complete" for path in paths):
        return dict(status="not_available", lost_cases=[])
    cases = manifest["evaluation"]
    raw = load_stage(root, "baseline", cases, runtime_id)
    adapted = load_stage(root, f"run_{seed}/evaluate", cases, runtime_id)
    learned = {item["case_id"]: item for item in adapted}
    lost = []
    for original in raw:
        new = learned[original["case_id"]]
        if not original["success"] or new["success"]:
            continue
        item = dict(case_id=original["case_id"], case=original["case"], traces={})
        for label, record in (("raw", original), ("adapted", new)):
            states = np.asarray(record["achieved_positions"], dtype=np.float64)
            steps = np.r_[0, np.cumsum(record["macro_durations"])].tolist()
            if len(states) != len(steps) or steps[-1] != record["micro_steps"]:
                raise ValueError("evaluation macro durations do not align with observed states")
            item["traces"][label] = dict(states=states.tolist(), steps=steps,
                                         distances=np.linalg.norm(states - record["case"]["goal"], axis=1).tolist(),
                                         success=record["success"], mpc_updates=record["mpc_updates"])
        lost.append(item)
    return dict(status="available", lost_cases=lost, resolution="macro-boundary observations, not every primitive action")


def _saved_candidate_audits(root, seed, runtime_id):
    result = []
    for path in sorted((root / "diagnostics" / f"run_{seed}").glob("audit_*.json")):
        saved = json.loads(path.read_text())
        if saved.get("runtime_id") != runtime_id:
            raise ValueError("candidate audit provenance mismatch")
        result.append(dict(file=path.name, case_id=saved["case"]["case_id"], cost=saved["cost"], subsets=saved["subsets"]))
    return result


def _describe(report):
    after = report["pair_quality"]["after"]["episode_uniform"]["contradictory_fraction"]
    bad = "unavailable (no eligible pairs)" if after is None else f"{100*after:.1f}%"
    away = report["motion"]["away_fraction"]
    away_text = "unavailable (no actions)" if away is None else f"{100*away:.1f}%"
    return f"""# Saved-experience audit

Read-only analysis of {report['episodes']} collection episodes, {report['primitive_actions']}
executed primitive actions and {report['frames']} real frames. No encoder/world-model
load, new rollouts or adapter training occurred.

- Actions moving away by more than the diagnostic tolerance: {away_text}.
- Contradictory retained ranking pairs under the trainer's episode-uniform sampling: {bad}.
- Episodes visiting the farther test-start rectangle: {report['coverage']['episodes_visiting_far_rectangle']}.
- Saved fitting/filter counts verified: {report['saved_filter_verified']}.

`summary.json` contains definitions and all aggregates. `episodes.csv` gives one
row per episode. The before/after pair figure uses uniform episode then uniform
pair sampling, as the trainer does; pooled pair rates have a separate JSON field.
The tolerance is {report['progress_tolerance']} simulator units, with strict-sign
counts also retained. A neutral pair is not confidently ordered by this diagnostic.
Reversals mean consecutive nonzero vectors turn by more than 90 degrees.

Path efficiency is shortest distance to the 4.5-unit success region divided by
actual travelled distance, for successful episodes. Euclidean comparisons are
interpreted for the unobstructed same-side pilot, not doorway detours. Exiting
the near *start* rectangle toward the goal is expected, not a quality failure.

Raw files are unchanged. Image examples add a green goal/success-region overlay
for interpretation. Lost-case plots use coarse macro-boundary observations.
Exact learned-cost ordering is not computed because the original goal feature
was not saved; a terminal success image is not substituted for that goal.
"""


def run_audit(pilot_dir, run_seed=0, progress_tolerance=.25, *, render=True):
    root = Path(pilot_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    runtime = json.loads((root / "runtime.json").read_text())
    if runtime["signature"]["manifest"] != digest(manifest):
        raise ValueError("manifest provenance mismatch")
    rows = _load_collection(root, manifest, runtime["runtime_id"], run_seed, progress_tolerance)
    if not rows:
        raise ValueError("no collection episodes to audit")
    verified, losses = _verify_training(root, run_seed, rows)
    count = sum(row["steps"] for row in rows)
    frame_count = sum(row["frames"] for row in rows)
    report = dict(schema_version=1, runtime_id=runtime["runtime_id"], pilot_source_commit=runtime.get("source_commit"),
                  run_seed=run_seed, episodes=len(rows), successful_episodes=sum(row["success"] for row in rows),
                  primitive_actions=count, frames=frame_count, progress_tolerance=progress_tolerance,
                  pair_rules=dict(revisit_radius=.5, min_primitive_gap=2, min_position_separation=2.),
                  saved_filter_verified=verified, loss_history=losses,
                  mean_actions=float(np.mean([row["steps"] for row in rows])),
                  median_actions=float(np.median([row["steps"] for row in rows])),
                  motion={key + "_fraction": _fraction(sum(row[key + "_actions"] for row in rows), count)
                          for key in ("closer", "away", "neutral", "strict_away", "stationary")},
                  mean_success_path_efficiency=float(np.mean([row["success_path_efficiency"] for row in rows if row["success_path_efficiency"] is not None]))
                  if any(row["success_path_efficiency"] is not None for row in rows) else None,
                  pair_quality=dict(before=summarize_pair_quality(rows, "pairs_before"), after=summarize_pair_quality(rows, "pairs_after")),
                  coverage=dict(far_rectangle_frames=sum(row["far_rectangle_frames"] for row in rows),
                                episodes_visiting_far_rectangle=sum(row["far_rectangle_frames"] > 0 for row in rows),
                                note="The near start rectangle excludes the goal; leaving it is not inherently poor behavior."),
                  evaluation=_evaluation_comparison(root, manifest, runtime["runtime_id"], run_seed),
                  existing_candidate_audits=_saved_candidate_audits(root, run_seed, runtime["runtime_id"]),
                  adapter_ordering_audit=dict(status="not_computed", reason="Original goal embedding was not persisted; no goal proxy or encoder load."),
                  definitions=dict(progress="goal distance decreases by more than tolerance", contradictory_pair="later state farther from goal by more than tolerance",
                                   ambiguous_pair="distance difference within tolerance; not proven good supervision",
                                   episode_uniform="average per-usable-successful-episode pair fractions; matches trainer sampling",
                                   pooled_pairs="all pairs pooled; long episodes contribute more pairs"))
    source = Path(__file__)
    report["audit_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        report["audit_source_commit"] = subprocess.check_output(["git", "-C", str(source.parents[2]), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        report["audit_source_commit"] = None
    output = root / "experience_audit" / f"run_{run_seed}"
    output.mkdir(parents=True, exist_ok=True)
    traces = [{key: row[key] for key in ("case_id", "raw_path", "trace", "contradictory_examples")} for row in rows]
    _write(output / "summary.json", report)
    _write(output / "traces.json", traces)
    scalar_rows = []
    for row in rows:
        scalar = {key: value for key, value in row.items() if not isinstance(value, (list, dict))}
        for label in ("pairs_before", "pairs_after"):
            scalar.update({label + "_" + key: value for key, value in row[label].items()})
        scalar_rows.append(scalar)
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    (output / "README.md").write_text(_describe(report))
    if render:
        from src.mylonas_ijepa_wm.curriculum_audit_figures import render_audit
        render_audit(root, rows, report, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--progress-tolerance", type=float, default=.25)
    args = parser.parse_args(argv)
    report = run_audit(args.pilot_dir, args.run_seed, args.progress_tolerance)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
