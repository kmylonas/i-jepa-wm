"""Fit one geodesic ranking adapter for both rooms and test held-out goals."""
from itertools import combinations

import numpy as np
import torch

from src.mylonas_ijepa_wm import geodesic_radial as radial
from src.mylonas_ijepa_wm import geodesic_square as square
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
from src.mylonas_ijepa_wm.multigoal_ranking import GAP, fit
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash


BANDS = radial.BANDS
TRAIN_GOALS = ((15, 15), (19, 30), (15, 46), (49, 15), (45, 30), (49, 46))
TEST_GOALS = ((12, 23), (24, 12), (24, 31), (12, 42), (52, 23), (40, 12), (40, 31), (52, 42))


def make_design(*, geometry="radial"):
    if geometry not in ("radial", "square"):
        raise ValueError("unknown multigoal fitting geometry")
    points, lookup = [], {}

    def intern(point):
        key = tuple(map(float, point))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(list(key))
        return lookup[key]

    goal_ids = [intern(g) for g in TRAIN_GOALS]
    heldout_goal_ids = [intern(g) for g in TEST_GOALS]
    axis = [radial.BOUNDS[0], *range(6, 60, 2), radial.BOUNDS[1]]
    goals = (*TRAIN_GOALS, *TEST_GOALS)
    if geometry == "square":
        samples = [[dict(id=intern(node["point"]), band=node["band"], distance=node["distance"])
                    for node in square.training_nodes(goal)] for goal in TRAIN_GOALS]
    else:
        samples = [[dict(id=intern(point), band=band, distance=radial.distance(point, goal))
                    for band, radii in enumerate(radial.RADII)
                    for radius in radii for _, point in radial._ring(radius, 0, goal)]
                   for goal in TRAIN_GOALS]
    grid = [intern((x, y)) if radial.valid((x, y)) else None for y in axis for x in axis]
    if geometry == "square":
        grid_ids = [i for i in grid if i is not None]
        rings = [[group["ids"] for group in square.contours(points, grid_ids, goal)] for goal in goals]
        bands = square.BANDS
        extra = dict(geometry=geometry, spacing=square.SPACING)
    else:
        rings = [[[intern(point) for _, point in radial._ring(radius, 7.5, goal)]
                  for radii in radial.RADII for radius in radii] for goal in goals]
        bands = BANDS
        extra = {}
    return dict(points=points, goal_ids=goal_ids, heldout_goal_ids=heldout_goal_ids, samples=samples,
                axis=axis, grid=grid, rings=rings, bands=[list(b) for b in bands], layout=list(radial.LAYOUT), **extra)


def population(design, hashes, stage):
    protected = {hashes[i] for i in design["heldout_goal_ids"]}
    result = []
    for goal, nodes in zip(design["goal_ids"], design["samples"]):
        if hashes[goal] in protected:
            raise ValueError("training goal image aliases a held-out goal")
        seen = protected | {hashes[goal]}
        eligible = []
        for node in nodes:
            h = hashes[node["id"]]
            if node["band"] <= stage and h not in seen:
                eligible.append(node)
                seen.add(h)
        for a, b in combinations(eligible, 2):
            delta = a["distance"] - b["distance"]
            if abs(delta) >= GAP:
                far, near = (a, b) if delta > 0 else (b, a)
                result.append((goal, far["id"], near["id"]))
    return result


def _encode(env, encode, points):
    env.reset(0, layout=radial.LAYOUT, start=(18, 15), goal=radial.GOAL)
    tokens = None
    hashes = []
    for start in range(0, len(points), 8):
        frames = np.stack([env._render_at(p) for p in points[start:start + 8]])
        batch = encode(frames).detach().cpu().float()
        if tokens is None:
            tokens = batch.new_empty((len(points), *batch.shape[1:]))
        tokens[start:start + len(batch)].copy_(batch)
        hashes.extend(map(_frame_hash, frames))
    return tokens, hashes


def _local(truth, scores, edges, tolerance):
    if not len(edges):
        return dict(pairs=0, correct=None, uphill=None)
    delta = scores[edges[:, 0]] - scores[edges[:, 1]]
    delta *= np.sign(truth[edges[:, 0]] - truth[edges[:, 1]])
    return dict(pairs=len(delta), correct=float((delta > tolerance).mean()),
                uphill=float((delta < -tolerance).mean()))


def _goal_report(design, goal, truth, extent, raw, adapted, mask, edges, index, outer):
    points = np.asarray(design["points"])
    # Use only the two rooms here. Doorway pixels remain in the full-grid metrics.
    opposite = points[:, 0] > radial.WALL_RIGHT if points[goal, 0] < radial.WALL_LEFT else points[:, 0] < radial.WALL_LEFT
    ids = np.asarray([i for i in design["grid"] if i is not None and mask[i]], dtype=int)
    within, beyond, same = (("within_square", "beyond_square", "same_extent") if design.get("geometry") == "square"
                            else ("within_radius", "beyond_radius", "same_radius"))
    cohorts = {"full_map": [ids], within: [ids[extent[ids] <= outer + radial.TIE]],
               beyond: [ids[extent[ids] > outer + radial.TIE]], "opposite_room": [ids[opposite[ids]]],
               same: [[i for i in ring if mask[i]] for ring in design["rings"][index]]}
    metrics = {
        name: dict(raw=radial._metric(truth, raw, group), adapted=radial._metric(truth, adapted, group))
        for name, group in cohorts.items()
    }
    directed = edges[np.abs(truth[edges[:, 0]] - truth[edges[:, 1]]) >= GAP]
    for name, selected in (("local", directed), ("local_opposite_room", directed[opposite[directed].all(axis=1)])):
        metrics[name] = {
            label: _local(truth, scores, selected, max(1e-7, .001 * float(np.ptp(scores[ids]))))
            for label, scores in (("raw", raw), ("adapted", adapted))
        }
    grid = design["grid"]
    return dict(goal=points[goal].tolist(), status="withheld" if goal in design["heldout_goal_ids"] else "seen",
                metrics=metrics, grid_costs=[None if i is None else float(adapted[i]) for i in grid])


def run(output, env, encode, device, signature, *, updates=500, geometry="radial"):
    """Fit and evaluate cumulative, goal-balanced geodesic ranking checkpoints."""
    design = make_design(geometry=geometry)
    square_grid = geometry == "square"
    kind = "geodesic_multigoal_square_v1" if square_grid else "geodesic_multigoal_v1"
    config = dict(kind=kind, design=digest(design), runtime=signature, updates=updates)
    path = output / "config.json"
    if path.exists() and read(path) != config:
        raise ValueError("multigoal run configuration changed")
    write_json(path, config)
    write_json(output / "design.json", design)
    points = np.asarray(design["points"])
    goals = design["goal_ids"] + design["heldout_goal_ids"]
    tokens, hashes = _encode(env, encode, points)
    raw = _scores(None, tokens, goals, device)
    truth = np.asarray([[radial.distance(p, points[g]) for p in points] for g in goals])
    offsets = points[None, :, :] - points[goals, None, :]
    extent = np.abs(offsets).max(axis=2) if square_grid else np.linalg.norm(offsets, axis=2)

    bands = design["bands"]
    complete_population = population(design, hashes, len(bands) - 1)
    exposed = {hashes[i] for triplet in complete_population for i in triplet} | {hashes[i] for i in goals}
    mask = np.asarray([h not in exposed for h in hashes])
    n = len(design["axis"])
    grid = np.asarray(design["grid"], dtype=object).reshape(n, n)
    neighbors = [*zip(grid[:, :-1].flat, grid[:, 1:].flat), *zip(grid[:-1].flat, grid[1:].flat)]
    edges = np.asarray([(a, b) for a, b in neighbors if a is not None and b is not None and mask[a] and mask[b]
                        and radial._visible(points[a], points[b])], dtype=int).reshape(-1, 2)

    reports = []
    previous, previous_sha = None, None
    for stage, (_, outer) in enumerate(bands):
        triples = complete_population if stage == len(bands) - 1 else population(design, hashes, stage)
        ids = sorted({i for triplet in triples for i in triplet})
        folder = output / f"band_{stage + 1}"
        checkpoint, marker = folder / "adapter.pt", folder / "complete.json"
        inputs = dict(runtime_id=digest(config), incoming=previous_sha, triplets=digest(triples))
        if marker.exists():
            if read(marker) != dict(inputs=inputs, sha256=file_sha256(checkpoint)):
                raise ValueError("multigoal checkpoint provenance differs")
            adapter, payload = load_adapter(checkpoint, "cpu", expected_runtime_id=digest(config))
            training = payload["training"]
        else:
            goal_rows = {g: row for row, g in enumerate(goals)}
            exposures = {(goal_rows[g], i) for g, far, near in triples for i in (far, near)}
            scale_values = np.asarray([raw[g, i] for g, i in exposures])
            positive = scale_values[scale_values > 0]
            if not len(positive):
                raise ValueError("training raw MSE scale is zero")
            remap = np.full(len(points), -1, dtype=np.int32)
            remap[ids] = np.arange(len(ids), dtype=np.int32)
            adapter, training = fit(tokens[ids], remap[np.asarray(triples, dtype=np.int32)],
                                    float(np.median(positive)), initial_adapter=previous, seed=stage,
                                    updates=updates, batch_size=16, device=device,
                                    progress=lambda row: print(row, flush=True))
            save_adapter(checkpoint, adapter, training, inputs)
            write_json(marker, dict(inputs=inputs, sha256=file_sha256(checkpoint)))
        adapted = _scores(adapter, tokens, goals, device)
        previous, previous_sha = adapter, file_sha256(checkpoint)
        by_goal = [_goal_report(design, g, truth[j], extent[j], raw[j], adapted[j], mask, edges, j, outer)
                   for j, g in enumerate(goals)]
        counts = dict(zip(goals, [0] * len(goals)))
        for g, _, _ in triples:
            counts[g] += 1
        size = "half_width" if square_grid else "outer_radius"
        report = dict(band=stage + 1, pairs=len(triples),
                      pairs_per_training_goal=[counts[g] for g in design["goal_ids"]],
                      train_accuracy=training["probe_after"]["rank_accuracy"], by_goal=by_goal, **{size: outer})
        write_json(folder / "report.json", report)
        reports.append(report)
        print(f"Band {stage + 1} {'half-width' if square_grid else 'R'}={outer}: {len(triples)} triples across "
              f"{sum(counts[g] > 0 for g in design['goal_ids'])} training goals", flush=True)

    result = dict(goals=[points[g].tolist() for g in goals], seen_goals=len(TRAIN_GOALS),
                  axis=design["axis"], heldout_grid_points=int(sum(mask[i] for i in design["grid"] if i is not None)),
                  raw_grid=[[None if i is None else float(row[i]) for i in design["grid"]] for row in raw],
                  truth_grid=[[None if i is None else float(row[i]) for i in design["grid"]] for row in truth],
                  stages=reports)
    write_json(output / "comparison.json", result)
    return result


if __name__ == "__main__":
    radial.main(runner=run)
