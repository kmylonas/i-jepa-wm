"""Compare cumulative hand-crafted ranking checkpoints in the collection run's right room."""

import argparse
from pathlib import Path

import numpy as np

from src.mylonas_ijepa_wm import geodesic_multigoal as multi
from src.mylonas_ijepa_wm import geodesic_radial as wall
from src.mylonas_ijepa_wm import geodesic_square as square
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import file_sha256
from src.mylonas_ijepa_wm.multigoal_ranking import GAP, fit
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores


def make_design(source, manifest):
    points, index = [], {}

    def intern(point):
        key = tuple(map(float, point))
        if key not in index:
            index[key] = len(points)
            points.append(list(key))
        return index[key]

    bands = manifest["bands"]
    train_ids = manifest["goal_ids"]
    test_ids = [i for i in source["goal_ids"] if i not in train_ids]
    refs = [intern(source["points"][i]) for i in train_ids + test_ids]
    grid = source["grid"]
    grid_ids = [intern(source["points"][i]) for i in grid["point_ids"]]
    mid_x, mid_y = [(np.asarray(axis[:-1]) + axis[1:]) / 2 for axis in (grid["xs"], grid["ys"])]
    evaluation_ids = [intern((x, y)) for y in mid_y for x in mid_x]
    if any(not wall.valid(p) or p[0] <= wall.WALL_RIGHT for p in points):
        raise ValueError("goals and evaluation grid must be in the legal right room")

    samples = []
    for goal in refs[:len(train_ids)]:
        nodes = [dict(id=intern(node["point"]), band=node["band"], distance=node["distance"])
                 for node in square.training_nodes(points[goal], bands=bands)
                 if node["point"][0] > wall.WALL_RIGHT]
        samples.append(nodes)
    return dict(points=points, goal_ids=refs[:len(train_ids)], heldout_goal_ids=refs[len(train_ids):],
                samples=samples, bands=bands, grid=dict(grid, point_ids=grid_ids),
                evaluation=dict(point_ids=evaluation_ids, xs=mid_x.tolist(), ys=mid_y.tolist()))


def _summaries(costs, truth, grid, heldout, crossing):
    rows = []
    for values, distances, bands in zip(costs, truth, crossing):
        tolerance = max(1e-7, .001 * float(np.ptp(values[heldout])))
        rows.append(dict(
            ranking=wall._metric(distances, values, [np.flatnonzero(heldout)]),
            local=multi._local(distances, values, grid[np.abs(distances[grid[:, 0]] - distances[grid[:, 1]]) >= GAP],
                               tolerance),
            boundaries=[multi._local(distances, values, edges, tolerance) for edges in bands],
        ))
    return rows


def run(output, env, encode, device, signature, *, source, collection):
    """Fit cumulative right-room square ranking from the collection experiment's frozen inputs."""
    manifest = collection["manifest"]
    design = make_design(source, manifest)
    settings = dict(kind="handcrafted_half_square_v1", design=digest(design), source=digest(source),
                    signature=signature, fitting=collection["fitting"], scale=collection["scale"])
    config_path = output / "config.json"
    if config_path.exists() and read(config_path) != settings:
        raise ValueError("hand-crafted run configuration changed")
    write_json(config_path, settings)

    tokens, hashes = multi._encode(env, encode, design["points"])
    evaluation = design["evaluation"]
    evaluation_ids = evaluation["point_ids"]
    protected = {hashes[i] for i in evaluation_ids}
    protected.update(source["frame_sha256"][i] for i in source["goal_ids"] if i not in manifest["goal_ids"])
    training = dict(design, samples=[[node for node in nodes if hashes[node["id"]] not in protected]
                                     for nodes in design["samples"]])
    grid = design["grid"]
    grid_ids = grid["point_ids"]
    goals = design["goal_ids"] + design["heldout_goal_ids"]
    scored_ids = grid_ids + evaluation_ids
    eval_tokens = tokens[scored_ids + goals]
    tokens = tokens.half()
    ref_ids = list(range(len(scored_ids), len(eval_tokens)))
    points = np.asarray(design["points"])
    offsets = points[evaluation_ids][None, :, :] - points[goals][:, None, :]
    truth = np.linalg.norm(offsets, axis=-1)
    extent = np.abs(offsets).max(axis=-1)
    grid_layout = np.arange(len(evaluation_ids)).reshape(len(evaluation["ys"]), len(evaluation["xs"]))
    a = np.r_[grid_layout[:, :-1].ravel(), grid_layout[:-1, :].ravel()]
    b = np.r_[grid_layout[:, 1:].ravel(), grid_layout[1:, :].ravel()]
    reference_hashes = {hashes[g] for g in goals}
    heldout = np.asarray([hashes[i] not in reference_hashes for i in evaluation_ids])
    visible_edges = heldout[a] & heldout[b]
    a, b = a[visible_edges], b[visible_edges]
    edges = np.column_stack((a, b))
    crossing = []
    for distance, width in zip(truth, extent):
        # An inward edge must cross the square and also make physical progress.
        a_outside = width[a] > width[b]
        outside, inside = np.where(a_outside, a, b), np.where(a_outside, b, a)
        progress = distance[outside] - distance[inside] >= GAP
        inward_edges = np.column_stack((outside, inside))
        crossing.append([inward_edges[progress & (width[outside] > outer) & (width[inside] <= outer)]
                         for _, outer in design["bands"]])

    raw_cost = _scores(None, eval_tokens, ref_ids, device)
    result = dict(grid=grid, goals=points[goals].tolist(), seen_goals=len(design["goal_ids"]),
                  bands=design["bands"], heldout_points=int(heldout.sum()),
                  raw=dict(costs=raw_cost[:, :len(grid_ids)].tolist(),
                           metrics=_summaries(raw_cost[:, len(grid_ids):len(scored_ids)], truth,
                                              edges, heldout, crossing)), stages=[])
    previous, previous_sha = None, None
    runtime_id = digest(settings)
    for stage, (inner, outer) in enumerate(design["bands"]):
        triples = multi.population(training, hashes, stage)
        folder = output / f"band_{stage + 1}"
        checkpoint, marker = folder / "adapter.pt", folder / "complete.json"
        inputs = dict(runtime_id=runtime_id, incoming=previous_sha, pairs=digest(triples))
        if marker.exists():
            if read(marker) != dict(inputs=inputs, sha256=file_sha256(checkpoint)):
                raise ValueError(f"checkpoint provenance differs in Band {stage + 1}")
            adapter, payload = load_adapter(checkpoint, "cpu", expected_runtime_id=runtime_id)
            fitted = payload["training"]
        else:
            adapter, fitted = fit(tokens, triples, settings["scale"], initial_adapter=previous,
                                  device=device, **settings["fitting"])
            save_adapter(checkpoint, adapter, fitted, inputs)
            write_json(marker, dict(inputs=inputs, sha256=file_sha256(checkpoint)))
        previous, previous_sha = adapter, file_sha256(checkpoint)
        costs = _scores(adapter, eval_tokens, ref_ids, device)
        metrics = _summaries(costs[:, len(grid_ids):len(scored_ids)], truth, edges, heldout, crossing)
        for row in metrics:
            row["crossing_inward"] = row["boundaries"][stage]
        report = dict(band=stage + 1, pairs=len(triples), costs=costs[:, :len(grid_ids)].tolist(), metrics=metrics,
                      new_points_per_goal=[sum(node["band"] == stage for node in nodes) for nodes in training["samples"]],
                      train_before=fitted["probe_before"], train_after=fitted["probe_after"])
        result["stages"].append(report)
        write_json(folder / "report.json", report)
        write_json(output / "comparison.json", result)
        print(f"Band {stage + 1} ({inner}–{outer}): {len(triples)} cumulative pairs, "
              f"{sum(report['new_points_per_goal'])} new goal-position samples", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--collection-run", required=True, type=Path)
    args, rest = parser.parse_known_args(argv)
    collection = read(args.collection_run / "experiment.json")
    source_path = Path(collection["handcrafted"]["root"]) / "data" / "design.json"
    source = read(source_path)

    def runner(output, env, encode, device, signature):
        locations(output, args.collection_run, (source_path,))
        return run(output, env, encode, device, signature, source=source, collection=collection)

    return wall.main(rest, runner=runner)


if __name__ == "__main__":
    main()
