"""Collect real planner experience across expanding square bands on the Wall map."""
import argparse
from pathlib import Path

import numpy as np

from src.mylonas_ijepa_wm import experience_curriculum as experience
from src.mylonas_ijepa_wm import geodesic_radial as wall
from src.mylonas_ijepa_wm import geodesic_multigoal as metrics
from src.mylonas_ijepa_wm import geodesic_square as square
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING, digest, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash, runtime_variant


SEED = 20261107
HALF_WIDTHS = (*square.HALF_WIDTHS[:-1], 48)
BANDS = tuple(zip((0, *HALF_WIDTHS[:-1]), HALF_WIDTHS))
SAFE_LOW, SAFE_HIGH = 8., 56.
FITTING = dict(updates=500, batch_size=32, bottleneck=64, lr=1e-4, seed=0)


def _room(x):
    return "left" if x < wall.WALL_LEFT else "right" if x > wall.WALL_RIGHT else "door"


def _rectangles(goal, inner, outer):
    """Non-overlapping legal portions of the shell as axis-aligned rectangles."""
    x, y = goal
    shell = [(x - outer, x - inner, y - outer, y + outer),
             (x + inner, x + outer, y - outer, y + outer),
             (x - inner, x + inner, y - outer, y - inner),
             (x - inner, x + inner, y + inner, y + outer)]
    regions = [(SAFE_LOW, wall.WALL_LEFT, SAFE_LOW, SAFE_HIGH),
               (wall.WALL_RIGHT, SAFE_HIGH, SAFE_LOW, SAFE_HIGH),
               (wall.WALL_LEFT, wall.WALL_RIGHT, wall.DOOR_LOW, wall.DOOR_HIGH)]
    rectangles = []
    for x0, x1, y0, y1 in shell:
        for a, b, c, d in regions:
            lower = [max(x0, a), max(y0, c)]
            upper = [min(x1, b), min(y1, d)]
            if lower[0] < upper[0] and lower[1] < upper[1]:
                rectangles.append((lower, upper))
    return np.asarray(rectangles).reshape(-1, 2, 2)


def make_manifest(starts_per_goal=6):
    if starts_per_goal < 1:
        raise ValueError("starts per goal must be positive")
    rng = np.random.default_rng(SEED)
    # Three randomly drawn targets in each room, one in each vertical third.
    goals = [rng.uniform([left, y], [right, y + 16]).tolist()
             for left, right in ((8, 24), (40, 56)) for y in (8, 24, 40)]
    stages = []
    for index, (inner, outer) in enumerate(BANDS):
        cases = []
        for g, goal in enumerate(goals):
            rectangles = _rectangles(goal, inner, outer)
            if not len(rectangles):
                continue
            areas = np.prod(rectangles[:, 1] - rectangles[:, 0], axis=1)
            for j in range(starts_per_goal):
                for _ in range(10000):
                    lo, hi = rectangles[rng.choice(len(areas), p=areas / areas.sum())]
                    start = rng.uniform(lo, hi)
                    if wall.valid(start) and np.linalg.norm(start - goal) >= 4.5:
                        break
                else:
                    raise ValueError("could not sample a legal start outside the success radius")
                a, b = _room(start[0]), _room(goal[0])
                route = "same_room" if a == b else "doorway_start" if a == "door" else f"{a}_to_{b}"
                cases.append(dict(case_id=f"b{index + 1}_g{g:02d}_s{j:02d}", split="train", band=index + 1,
                                  goal_index=g, start=start.tolist(), goal=goal, layout=list(wall.LAYOUT),
                                  route=route, seed=int(rng.integers(0, 2**31 - 1)), run_seed=SEED))
        stages.append(cases)
    return dict(bands=[list(b) for b in BANDS], goals=goals, goal_ids=list(range(len(goals))), stages=stages,
                starts_per_goal=starts_per_goal, planning=dict(PLANNING), seed=SEED, pair_cap=experience.PAIR_CAP)


def _doorway_step(a, b):
    return bool(min(a[0], b[0]) <= wall.WALL_RIGHT and max(a[0], b[0]) >= wall.WALL_LEFT
                and wall.valid(a) and wall.valid(b) and wall._visible(a, b))


def crossings(positions):
    """Count passages that go through the opening and reach the other room."""
    xy = np.asarray(positions)
    origin = _room(xy[0, 0])
    pending = False
    result = []
    center = wall.LAYOUT[0]
    for a, b in zip(xy[:-1], xy[1:]):
        if (a[0] < center <= b[0]) or (b[0] < center <= a[0]):
            destination = "right" if b[0] >= center else "left"
            pending = origin not in ("door", destination) and _doorway_step(a, b)
        room = _room(b[0])
        if room != "door":
            if pending and origin != room:
                result.append(f"{origin}_to_{room}")
            origin, pending = room, False
    return result


def _legal_distances(xy, goal):
    inside = np.asarray([wall.valid(point) for point in xy])
    distances = np.asarray([wall.distance(point, goal) if ok else np.nan for point, ok in zip(xy, inside)])
    return distances, inside


def filter_steps(raw, case, design, cap):
    selected, stats = experience._filter(raw, case, design, cap, geometry=_legal_distances)
    xy = raw["states"].numpy()[:, :2]
    events = crossings(xy)
    stats.update(crossings=events, reached_other_room=case["route"] in events,
                 retained_doorway_pairs=sum(_doorway_step(xy[i], xy[i + 1]) for i in selected))
    return selected, stats


def make_design(runtime, manifest):
    train = manifest["goals"]
    goals = [*train, *[list(g) for g in metrics.TEST_GOALS]]
    ids = list(range(len(train), len(goals)))
    axis = [wall.BOUNDS[0], *range(6, 60, 2), wall.BOUNDS[1]]
    points = list(goals)
    grid = []
    for y in axis:
        for x in axis:
            grid.append(len(points) if wall.valid((x, y)) else None)
            if grid[-1] is not None:
                points.append([x, y])
    grid_ids = [i for i in grid if i is not None]
    design = dict(geometry="square", points=points, axis=axis, grid=grid, goal_ids=ids, heldout_goal_ids=ids,
                  heldout_mask=[i >= len(train) for i in range(len(points))],
                  rings=[[group["ids"] for group in square.contours(points, grid_ids, g)] for g in goals[len(train):]])
    path = runtime.root / "design.json"
    if path.exists():
        saved = read(path)
        saved_design = saved["design"]
        saved_geometry = {k: v for k, v in saved_design.items() if k != "frame_sha256"}
        if (saved["runtime_id"] != runtime.runtime_id or saved["geometry_digest"] != digest(design)
                or saved_geometry != design or len(saved_design["frame_sha256"]) != len(points)):
            raise ValueError("evaluation geometry or runtime changed")
        return saved_design
    runtime.env.reset(0, layout=wall.LAYOUT, start=(18, 15), goal=wall.GOAL)
    design["frame_sha256"] = [_frame_hash(runtime.env._render_at(point)) for point in points]
    if set(design["frame_sha256"][:len(train)]) & set(design["frame_sha256"][i] for i in ids):
        raise ValueError("training goal image aliases a held-out goal")
    write_json(path, dict(runtime_id=runtime.runtime_id, geometry_digest=digest(
        {k: v for k, v in design.items() if k != "frame_sha256"}), design=design))
    return design


def compare(runtime, manifest, design):
    points = np.asarray(design["points"])
    goals = design["goal_ids"]
    tokens = experience._encode_positions(runtime, design, list(range(len(points))))
    raw = _scores(None, tokens, goals, runtime.device)
    truth = np.asarray([[wall.distance(p, points[g]) for p in points] for g in goals])
    extent = np.abs(points[None] - points[goals, None]).max(axis=2)
    mask = np.asarray(design["heldout_mask"])
    goal_hashes = {design["frame_sha256"][i] for i in [*manifest["goal_ids"], *goals]}
    mask &= [h not in goal_hashes for h in design["frame_sha256"]]
    n = len(design["axis"])
    grid = np.asarray(design["grid"], dtype=object).reshape(n, n)
    neighbors = [*zip(grid[:, :-1].flat, grid[:, 1:].flat), *zip(grid[:-1].flat, grid[1:].flat)]
    edges = np.asarray([(a, b) for a, b in neighbors if a is not None and b is not None and mask[a] and mask[b]
                        and wall._visible(points[a], points[b])], dtype=int).reshape(-1, 2)
    stages = []
    for i, (_, outer) in enumerate(manifest["bands"]):
        adapter, report = experience._checkpoint(runtime, i)
        adapted = _scores(adapter, tokens, goals, runtime.device)
        by_goal = [metrics._goal_report(design, g, truth[j], extent[j], raw[j], adapted[j], mask, edges, j, outer)
                   for j, g in enumerate(goals)]
        stages.append(dict(band=i + 1, half_width=outer, collection=report["collection"],
                           updates=report["updates"], by_goal=by_goal))
    result = dict(axis=design["axis"], goals=points[goals].tolist(), stages=stages,
                  heldout_grid_points=int(sum(mask[i] for i in design["grid"] if i is not None)),
                  raw_grid=[[None if i is None else float(row[i]) for i in design["grid"]] for row in raw],
                  truth_grid=[[None if i is None else float(row[i]) for i in design["grid"]] for row in truth])
    write_json(runtime.root / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "compare"))
    for field in ("output-dir", "world-model-checkpoint", "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--band", type=int, choices=range(1, len(BANDS) + 1))
    parser.add_argument("--starts-per-goal", type=int, default=6)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    if args.operation == "stage" and args.band is None:
        parser.error("stage requires --band")
    args.output_dir, _ = locations(args.output_dir, args.encoder_checkpoint,
                                   (args.world_model_checkpoint, args.dino_wm_root))
    manifest = make_manifest(args.starts_per_goal)
    config = dict(kind="experience_square_v1", manifest=manifest, fitting=FITTING)
    path = args.output_dir / "experiment.json"
    if path.exists() and read(path) != config:
        raise ValueError("experiment configuration changed; choose a new output directory")
    write_json(path, config)
    args.precision = "fp32"
    args.encoder_batch_size = 8
    args.run_seed = SEED
    args.videos = False
    args.expected_world_model_sha256 = EXPECTED_WM
    args.expected_encoder_sha256 = EXPECTED_ENCODER
    args.runtime_signature_extra = runtime_variant(config, "experience_square", args.device)
    runtime = PilotRuntime(args, manifest)
    try:
        design = make_design(runtime, manifest)
        if args.operation == "stage":
            return experience.collect_and_fit(runtime, manifest, design, FITTING, None, args.band - 1,
                                              filter_steps=filter_steps)
        return compare(runtime, manifest, design)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
