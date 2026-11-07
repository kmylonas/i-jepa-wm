"""Rank hand-positioned frames by legal path distance in Euclidean bands."""
import argparse
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.mylonas_ijepa_wm import plan_pointmaze as visual
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import digest, write_json
from src.mylonas_ijepa_wm.curriculum_replay import load_adapter, save_adapter
from src.mylonas_ijepa_wm.curriculum_runtime import (
    DINO_COMMIT, EXPECTED_ENCODER, PilotRuntime, file_sha256, git_output, source_digest,
)
from src.mylonas_ijepa_wm.environment_adapters import WallAdapter
from src.mylonas_ijepa_wm.environment_configs import EnvironmentConfig
from src.mylonas_ijepa_wm.multigoal_ranking import GAP, fit
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash

GOAL = (15., 15.)
LAYOUT = (32, 30)
BANDS = ((6, 12), (12, 20), (20, 28), (28, 34), (34, 42), (42, 50), (50, 58), (58, 64))
RADII = ((8, 10), (14, 18), (22, 26), (30, 34), (36, 40), (44, 48), (52, 56), (60, 63))
WALL_LEFT, WALL_RIGHT = 29., 35.
DOOR_LOW, DOOR_HIGH = 26., 34.
BOUNDS = (4.001, 59.999)
CORNERS = [(x, y) for x in (WALL_LEFT, WALL_RIGHT) for y in (DOOR_LOW, DOOR_HIGH)]
TIE = 1e-5


def valid(point):
    x, y = point
    return (BOUNDS[0] <= x <= BOUNDS[1] and BOUNDS[0] <= y <= BOUNDS[1]
            and (x < WALL_LEFT or x > WALL_RIGHT or DOOR_LOW < y < DOOR_HIGH))


def _visible(a, b):
    # Contact with a corner or travel along a face represents the limit
    # of paths with arbitrarily small clearance from the solid wall.
    dx, dy = b - a
    if dx == 0:
        return (not WALL_LEFT < a[0] < WALL_RIGHT
                or min(a[1], b[1]) >= DOOR_LOW and max(a[1], b[1]) <= DOOR_HIGH)
    entry, exit = sorted(((WALL_LEFT - a[0]) / dx, (WALL_RIGHT - a[0]) / dx))
    lo, hi = max(0., entry), min(1., exit)
    if lo >= hi:
        return True
    y1, y2 = a[1] + lo * dy, a[1] + hi * dy
    return min(y1, y2) >= DOOR_LOW - 1e-9 and max(y1, y2) <= DOOR_HIGH + 1e-9


def distance(point, goal=GOAL):
    """Shortest path for the ball center around the two solid wall sections."""
    points = np.asarray([point, goal, *CORNERS], dtype=float)
    if not valid(point) or not valid(goal):
        raise ValueError("shortest-path endpoints must be valid ball positions")
    if _visible(*points[:2]):
        return float(np.linalg.norm(points[0] - points[1]))
    lengths = np.full((len(points), len(points)), np.inf)
    np.fill_diagonal(lengths, 0.)
    for i, j in combinations(range(len(points)), 2):
        if _visible(points[i], points[j]):
            lengths[i, j] = lengths[j, i] = np.linalg.norm(points[i] - points[j])
    for k in range(len(points)):
        lengths = np.minimum(lengths, lengths[:, k, None] + lengths[None, k, :])
    return float(lengths[0, 1])


def _ring(radius, offset):
    for angle in range(0, 360, 15):
        theta = np.deg2rad(angle + offset)
        point = (np.asarray(GOAL) + radius * np.array([np.cos(theta), np.sin(theta)])).astype(np.float32)
        if valid(point):
            yield angle + offset, point.tolist()


def make_design():
    nodes = []
    for band, radii in enumerate(RADII):
        for radius in radii:
            for angle, point in _ring(radius, 0):
                nodes.append(dict(point=point, band=band, angle=angle,
                                  radius=radius, distance=distance(point)))
    axis = [BOUNDS[0], *range(6, 60, 2), BOUNDS[1]]
    points = [list(GOAL), *(n["point"] for n in nodes)]
    grid = []
    for y in axis:
        for x in axis:
            if valid((x, y)):
                grid.append(len(points))
                points.append([x, y])
            else:
                grid.append(None)
    rings = []
    for radii in RADII:
        for radius in radii:
            probes = [point for _, point in _ring(radius, 7.5)]
            rings.append(dict(radius=radius, ids=list(range(len(points), len(points) + len(probes)))))
            points.extend(probes)
    return dict(goal=list(GOAL), layout=list(LAYOUT), bands=[list(b) for b in BANDS],
                training=nodes, points=points, axis=axis, grid=grid, rings=rings)


def training_pairs(nodes, stage):
    eligible = [i for i, n in enumerate(nodes) if n["band"] <= stage]
    pairs = []
    for i, j in combinations(eligible, 2):
        a, b = nodes[i], nodes[j]
        delta = a["distance"] - b["distance"]
        if abs(delta) >= GAP:
            pairs.append((i, j) if delta > 0 else (j, i))
    return pairs


def _metric(truth, scores, groups):
    count, correct = 0, 0
    for group in groups:
        ids = np.asarray(group, dtype=int)
        if not len(ids):
            continue
        chosen = truth[ids, None] - truth[None, ids] >= GAP
        differences = scores[ids, None] - scores[None, ids]
        tolerance = max(1e-7, .001 * float(np.ptp(scores[ids])))
        count += int(chosen.sum())
        correct += int((differences[chosen] > tolerance).sum())
    return dict(pairs=count, correct=correct / count if count else None)


def run(output, env, encode, device, signature, *, updates=500):
    design = make_design()
    config = dict(kind="geodesic_radial_v2", design=digest(design), runtime=signature, updates=updates)
    config_path = output / "config.json"
    if config_path.exists() and read(config_path) != config:
        raise ValueError("geodesic radial run configuration changed")
    write_json(config_path, config)
    write_json(output / "design.json", design)
    env.reset(0, layout=LAYOUT, start=(18, 15), goal=GOAL)
    goal_frame = env.render_goal()
    hashes, chunks = [], []
    points = design["points"]
    for start in range(0, len(points), 8):
        frames = np.stack([goal_frame if i == 0 else env._render_at(points[i])
                           for i in range(start, min(start + 8, len(points)))])
        hashes.extend(map(_frame_hash, frames))
        chunks.append(encode(frames).detach().cpu().float())
    tokens = torch.cat(chunks)
    del chunks
    raw = _scores(None, tokens, [0], device)[0]
    truth = np.asarray([distance(p) for p in points])
    radii = np.linalg.norm(np.asarray(points) - GOAL, axis=1)
    reports = []
    previous, previous_sha = None, None
    nodes = design["training"]
    fitting_hashes = set(hashes[:len(nodes) + 1])
    heldout = [i for i in design["grid"] if i is not None and hashes[i] not in fitting_hashes]
    for index, (inner, outer) in enumerate(BANDS):
        pairs = training_pairs(nodes, index)
        point_ids = sorted({i + 1 for pair in pairs for i in pair})
        remap = {i: j + 1 for j, i in enumerate(point_ids)}
        triples = [[0, remap[a + 1], remap[b + 1]] for a, b in pairs]
        stage = output / f"band_{index + 1}"
        checkpoint, marker = stage / "adapter.pt", stage / "complete.json"
        provenance = dict(runtime_id=digest(config), incoming=previous_sha, pairs=digest(pairs))
        if marker.exists():
            if read(marker) != dict(inputs=provenance, sha256=file_sha256(checkpoint)):
                raise ValueError(f"checkpoint provenance differs in band {index + 1}")
            adapter, payload = load_adapter(checkpoint, "cpu", expected_runtime_id=digest(config))
            training = payload["training"]
        else:
            positive = raw[point_ids][raw[point_ids] > 0]
            if not len(positive):
                raise ValueError("training frames have zero raw MSE to goal")
            adapter, training = fit(tokens[[0, *point_ids]], triples, float(np.median(positive)),
                                    initial_adapter=previous, seed=index, updates=updates, batch_size=16,
                                    device=device, progress=lambda row: print(row, flush=True))
            save_adapter(checkpoint, adapter, training, provenance)
            write_json(marker, dict(inputs=provenance, sha256=file_sha256(checkpoint)))
        adapted = _scores(adapter, tokens, [0], device)[0]
        previous, previous_sha = adapter, file_sha256(checkpoint)
        cohorts = dict(within_radius=[[i for i in heldout if radii[i] <= outer + TIE]],
                       current_band=[[i for i in heldout if inner < radii[i] <= outer + TIE]],
                       beyond_radius=[[i for i in heldout if radii[i] > outer + TIE]],
                       opposite_side=[[i for i in heldout if points[i][0] > WALL_RIGHT]],
                       same_radius=[[i for i in ring["ids"] if hashes[i] not in fitting_hashes]
                                    for ring in design["rings"]])
        within_band = sum(nodes[a]["band"] == nodes[b]["band"] for a, b in pairs)
        counts = dict(within_band=within_band, across_bands=len(pairs) - within_band,
                      same_radius=sum(nodes[a]["radius"] == nodes[b]["radius"] for a, b in pairs),
                      across_wall=sum(nodes[a]["point"][0] > WALL_RIGHT and nodes[b]["point"][0] < WALL_LEFT
                                      or nodes[b]["point"][0] > WALL_RIGHT and nodes[a]["point"][0] < WALL_LEFT
                                      for a, b in pairs),
                      opposite_side_nodes=sum(points[i][0] > WALL_RIGHT for i in point_ids))
        report = dict(band=index + 1, outer_radius=outer, training_points=point_ids,
                      pair_counts=dict(total=len(pairs), **counts),
                      metrics={name: dict(raw=_metric(truth, raw, ids), adapted=_metric(truth, adapted, ids))
                               for name, ids in cohorts.items()},
                      adapted=adapted.tolist(), train_accuracy=training["probe_after"]["rank_accuracy"])
        write_json(stage / "report.json", report)
        reports.append(report)
        print(f"Band {index + 1} R={outer}: {report['pair_counts']}", flush=True)
    result = dict(design=design, truth=truth.tolist(), raw=raw.tolist(), stages=reports)
    write_json(output / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("output-dir", "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    output, _ = locations(args.output_dir, args.encoder_checkpoint, (args.dino_wm_root,))
    if file_sha256(args.encoder_checkpoint) != EXPECTED_ENCODER:
        raise ValueError("unexpected encoder checkpoint")
    if (git_output(args.dino_wm_root, "rev-parse", "HEAD") != DINO_COMMIT
            or git_output(args.dino_wm_root, "status", "--porcelain", "--untracked-files=no")):
        raise ValueError("renderer must match unmodified pinned DINO source")
    device = torch.device(args.device)
    encoder = visual.load_frozen_ijepa_encoder(args.encoder_checkpoint, device)
    runtime = SimpleNamespace(device=device, precision="fp32",
                              args=SimpleNamespace(encoder_batch_size=8), load_encoder=lambda: encoder)
    signature = dict(source=source_digest(Path(__file__).parents[1]), encoder=EXPECTED_ENCODER,
                     renderer=DINO_COMMIT, device=str(device), torch_version=str(torch.__version__),
                     device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                     matmul_precision=torch.get_float32_matmul_precision())
    env = WallAdapter(EnvironmentConfig(env="wall", wall_layouts=[list(LAYOUT)],
                                        dino_wm_root=str(args.dino_wm_root)), 100)
    try:
        return run(output, env, lambda frames: PilotRuntime.encode_frames(runtime, frames), device, signature)
    finally:
        env.close()


if __name__ == "__main__":
    main()
