"""Collect real planner experience across expanding square bands on the Wall map."""
import argparse
from functools import partial
from math import dist
from pathlib import Path

import numpy as np
import torch

from src.mylonas_ijepa_wm import experience_curriculum as experience
from src.mylonas_ijepa_wm import experience_exploration
from src.mylonas_ijepa_wm import geodesic_radial as wall
from src.mylonas_ijepa_wm import geodesic_multigoal as metrics
from src.mylonas_ijepa_wm import geodesic_square as square
from src.mylonas_ijepa_wm.cost_adapter import adapter_goal_cost
from src.mylonas_ijepa_wm.curriculum_multiband_protocol import read
from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING, digest, run_stage, write_json
from src.mylonas_ijepa_wm.curriculum_runtime import EXPECTED_ENCODER, EXPECTED_WM, PilotRuntime, file_sha256, save_tensors
from src.mylonas_ijepa_wm.radial_experiment import locations
from src.mylonas_ijepa_wm.radial_multigoal_experiment import _scores
from src.mylonas_ijepa_wm.ranking_planning import _frame_hash, runtime_variant


SEED = 20261107
HALF_WIDTHS = (*square.HALF_WIDTHS[:-1], 48)
BANDS = tuple(zip((0, *HALF_WIDTHS[:-1]), HALF_WIDTHS))
SAFE_LOW, SAFE_HIGH = 8., 56.
FITTING = dict(updates=500, batch_size=32, bottleneck=64, lr=1e-4, seed=0)
FRESH_RIGHT_GOALS = (
    (50., 42.), (44.204444885253906, 40.447086334228516), (53., 47.19615173339844),
    (44.30598068237305, 20.749631881713867), (50., 12.), (50., 16.),
)
EXTRA_LEFT_GOALS = ((14., 12.), (23., 30.), (16., 51.))


def _room(x):
    return "left" if x < wall.WALL_LEFT else "right" if x > wall.WALL_RIGHT else "door"


def _rectangles(goal, inner, outer, *, regions=None):
    """Non-overlapping legal portions of the shell as axis-aligned rectangles."""
    x, y = goal
    shell = [(x - outer, x - inner, y - outer, y + outer),
             (x + inner, x + outer, y - outer, y + outer),
             (x - inner, x + inner, y - outer, y - inner),
             (x - inner, x + inner, y + inner, y + outer)]
    if regions is None:
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


def make_manifest(starts_per_goal=6, *, round_seed=0, goals=None, extra_goal_count=0, reset_region=None):
    if starts_per_goal < 1:
        raise ValueError("starts per goal must be positive")
    rng = np.random.default_rng(SEED)
    # Three randomly drawn targets in each room, one in each vertical third.
    original_goals = [rng.uniform([left, y], [right, y + 16]).tolist()
                      for left, right in ((8, 24), (40, 56)) for y in (8, 24, 40)]
    if goals is None:
        goals = original_goals
    if round_seed:
        rng = np.random.default_rng([SEED, round_seed])
    # Appended goals should not change the saved protocol's starts or CEM seeds.
    extra_rng = np.random.default_rng([SEED, round_seed, 1]) if extra_goal_count else rng
    regions = (reset_region,) if reset_region is not None else None
    stages = []
    for index, (inner, outer) in enumerate(BANDS):
        cases = []
        for g, goal in enumerate(goals):
            sampler = rng if g < len(goals) - extra_goal_count else extra_rng
            rectangles = _rectangles(goal, inner, outer, regions=regions)
            if not len(rectangles):
                continue
            areas = np.prod(rectangles[:, 1] - rectangles[:, 0], axis=1)
            for j in range(starts_per_goal):
                for _ in range(10000):
                    lo, hi = rectangles[sampler.choice(len(areas), p=areas / areas.sum())]
                    start = sampler.uniform(lo, hi)
                    if wall.valid(start) and np.linalg.norm(start - goal) >= 4.5:
                        break
                else:
                    raise ValueError("could not sample a legal start outside the success radius")
                a, b = _room(start[0]), _room(goal[0])
                route = "same_room" if a == b else "doorway_start" if a == "door" else f"{a}_to_{b}"
                cases.append(dict(case_id=f"b{index + 1}_g{g:02d}_s{j:02d}", split="train", band=index + 1,
                                  goal_index=g, start=start.tolist(), goal=goal, layout=list(wall.LAYOUT),
                                  route=route, seed=int(sampler.integers(0, 2**31 - 1)), run_seed=SEED))
        stages.append(cases)
    return dict(bands=[list(b) for b in BANDS], goals=goals, goal_ids=list(range(len(goals))), stages=stages,
                starts_per_goal=starts_per_goal, planning=dict(PLANNING), seed=SEED, pair_cap=experience.PAIR_CAP)


def expanded_goals(right_manifest, *, extra_left=()):
    right = {}
    for case in right_manifest["stages"][0]:
        i, point = case["goal_index"], case["goal"]
        if i in right and right[i] != point:
            raise ValueError("saved right-room goal differs across cases")
        right[i] = point
    if (set(right) != set(range(6)) or len({tuple(p) for p in right.values()}) != 6
            or any(not wall.valid(p) or p[0] <= wall.WALL_RIGHT for p in right.values())):
        raise ValueError("expected six distinct saved right-room goals")
    left = make_manifest()["goals"][:3]
    extra = [list(point) for point in extra_left]
    if (any(not (SAFE_LOW <= x < wall.WALL_LEFT and SAFE_LOW <= y <= SAFE_HIGH) for x, y in extra)
            or len({tuple(p) for p in left + extra}) != len(left) + len(extra)):
        raise ValueError("extra goals must be distinct reset-safe positions in the left room")
    return [*left, *(right[i] for i in range(6)), *extra]


def make_repeated_manifest(starts_per_goal, *, rounds, goals=None, extra_goal_count=0, reset_region=None):
    options = dict(goals=goals, extra_goal_count=extra_goal_count, reset_region=reset_region)
    first = make_manifest(starts_per_goal, **options)
    first["planning"].update(planning_horizon=1, actions_per_plan=1)
    later = (make_manifest(starts_per_goal, round_seed=number, **options)
             for number in range(1, rounds))
    return experience.repeat_manifest(first, later)


def make_grid_manifest(goals, *, rounds):
    """Use each legal goal-centered lattice point once across the rounds of its band."""
    batches = [[[] for _ in BANDS] for _ in range(rounds)]
    seeds = np.random.default_rng(SEED)
    for g, goal in enumerate(goals):
        for j, node in enumerate(square.lattice_nodes(goal, bands=BANDS)):
            point, band = node["point"], node["band"]
            if any(not SAFE_LOW <= value <= SAFE_HIGH for value in point) or dist(point, goal) < 4.5:
                continue
            x, y = [round((value - center) / square.SPACING) for value, center in zip(point, goal)]
            a, b = _room(point[0]), _room(goal[0])
            route = "same_room" if a == b else "doorway_start" if a == "door" else f"{a}_to_{b}"
            batches[(x + y) % rounds][band].append(
                dict(case_id=f"b{band + 1}_g{g:02d}_s{j:03d}", split="train", band=band + 1,
                     goal_index=g, start=point, goal=goal, layout=list(wall.LAYOUT), route=route,
                     seed=int(seeds.integers(0, 2**31 - 1)), run_seed=SEED))
    first = dict(bands=[list(band) for band in BANDS], goals=goals, goal_ids=list(range(len(goals))),
                 grid_start_spacing=square.SPACING, stages=batches[0], seed=SEED, pair_cap=experience.PAIR_CAP,
                 planning=dict(PLANNING, planning_horizon=1, actions_per_plan=1))
    return experience.repeat_manifest(first, (dict(stages=stages) for stages in batches[1:]))


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


def _legal_positions(xy):
    return np.asarray([wall.valid(point) for point in xy], dtype=bool)


def temporal_filter(raw, case, design, cap):
    return experience._temporal_filter(raw, case, design, cap, domain=_legal_positions)


def loop_filtered_steps(raw, case, design, cap, *, radius, domain=_legal_positions):
    from src.mylonas_ijepa_wm.experience_loop_filter import loop_erased_steps

    return experience._temporal_filter(raw, case, design, cap, domain=domain,
                                       erase_loops=partial(loop_erased_steps, radius=radius))


def sample_failed_frames(states, frames, case, design, cap):
    """Keep the start and up to cap distinct later images spread over the failed path."""
    xy = np.asarray(states)[:, :2]
    inside = _legal_positions(xy)
    protected = experience._exclude_protected(xy, inside, dict(case, goal_id=case["goal_index"]), design)
    start_hash = _frame_hash(frames[0])
    if not inside[0] or start_hash in protected:
        return []
    protected.add(start_hash)
    candidates = {}
    for i in np.flatnonzero(inside[1:]) + 1:
        h = _frame_hash(frames[i])
        if h not in protected:
            candidates[h] = int(i)
    eligible = sorted(candidates.values())
    if not eligible:
        return []
    count = min(cap, len(eligible))
    selected = np.linspace(0, len(eligible) - 1, count, dtype=int) if count > 1 else [-1]
    return [0, *(eligible[i] for i in selected)]


def summarize_round_episode(record, saved, inner, macro_steps):
    events = crossings(record["primitive_positions"])
    retained = saved.get("pairs", ()) if record.get("success") else ()
    xy = np.asarray(saved.get("positions", []))
    return dict(
        crossings=events,
        reached_other_room=record["case"]["route"] in events,
        retained_doorway_pairs=sum(_doorway_step(xy[a], xy[b]) for a, b in retained),
        boundary=experience_exploration.measure(record, inner, macro_steps, right_room_only=False),
    )


def make_design(runtime, manifest):
    train = manifest["goals"]
    goals = [*train, *[list(g) for g in metrics.TEST_GOALS]]
    ids = list(range(len(train), len(goals)))
    # Offset evaluation nodes from the even lattice used by several training goals.
    first_grid_point = 5 if "grid_start_spacing" in manifest else 6
    axis = [wall.BOUNDS[0], *range(first_grid_point, 60, 2), wall.BOUNDS[1]]
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


def compare(runtime, manifest, design, *, band_count=None, collection_maps=False, single_band=False):
    points = np.asarray(design["points"])
    goals = design["goal_ids"]
    cache = runtime.root / "evaluation_grid_tokens.pt"
    if single_band and cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["runtime_id"] != runtime.runtime_id or saved["design"] != digest(design):
            raise ValueError("evaluation grid tokens differ from the current design")
        tokens = saved["tokens"]
    else:
        tokens = experience._encode_positions(runtime, design, list(range(len(points))))
        if single_band:
            save_tensors(cache, dict(runtime_id=runtime.runtime_id, design=digest(design), tokens=tokens))
    collection_ids = manifest["goal_ids"] if collection_maps else []
    if collection_maps:
        target_path = runtime.root / "collection_goals.pt"
        rounds = manifest["rounds_per_band"]
        first = read(runtime.root / experience._stage_name(0, rounds) / "complete.json")
        target = torch.load(target_path, map_location="cpu", weights_only=True)
        if (target["runtime_id"] != runtime.runtime_id or file_sha256(target_path) != first["goal_sha256"]
                or target["tokens"].shape != tokens[collection_ids].shape):
            raise ValueError("collection goal tensors differ from the saved planner targets")
        tokens[collection_ids] = target["tokens"]
    score_goals = goals + collection_ids
    raw = _scores(None, tokens, score_goals, runtime.device)

    def grid_values(scores):
        return [[None if i is None else float(row[i]) for i in design["grid"]] for row in scores]

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
    rounds = manifest.get("rounds_per_band", 1)
    bands = manifest["bands"][:band_count]
    start = max(0, (band_count - 1) * rounds - 1) if single_band else 0
    for i in range(start, len(bands) * rounds):
        band, number = divmod(i, rounds)
        _, outer = bands[band]
        adapter, report = experience._checkpoint(runtime, i, rounds)
        adapted = _scores(adapter, tokens, score_goals, runtime.device)
        by_goal = [metrics._goal_report(design, g, truth[j], extent[j], raw[j], adapted[j], mask, edges, j, outer)
                   for j, g in enumerate(goals)]
        stage = dict(band=band + 1, half_width=outer, collection=report["collection"],
                     updates=report["updates"], by_goal=by_goal)
        if rounds > 1:
            stage["round"] = number + 1
        if collection_maps:
            stage["collection_grid"] = grid_values(adapted[len(goals):])
        stages.append(stage)
    result = dict(axis=design["axis"], goals=points[goals].tolist(), stages=stages,
                  heldout_grid_points=int(sum(mask[i] for i in design["grid"] if i is not None)),
                  raw_grid=grid_values(raw[:len(goals)]), truth_grid=grid_values(truth))
    if collection_maps:
        result["collection_raw_grid"] = grid_values(raw[len(goals):])
    name = f"comparison_band_{band_count}.json" if single_band else "comparison.json"
    write_json(runtime.root / name, result)
    return result


def benchmark_band4(runtime, manifest, reference_run):
    """Replay the earlier Band 4 Round 1 cases with the incoming Band 4 cost."""
    previous = read(reference_run / "experiment.json")["manifest"]
    if previous["planning"] != manifest["planning"]:
        raise ValueError("fixed benchmark requires matching planner settings")
    old_stage = read(reference_run / "band_4_round_1" / "complete.json")
    old = old_stage["collection"]
    cases = [dict(row["case"], split="eval") for row in old]
    stage_index = 3 * manifest["rounds_per_band"] - 1
    adapter, checkpoint = experience._checkpoint(runtime, stage_index, manifest["rounds_per_band"])
    target_path = reference_run / "collection_goals.pt"
    saved = torch.load(target_path, map_location="cpu", weights_only=True)
    if (saved["runtime_id"] != old_stage["runtime_id"] or file_sha256(target_path) != old_stage["goal_sha256"]
            or len(saved["tokens"]) != len(previous["goals"])
            or any(case["goal"] != previous["goals"][case["goal_index"]] for case in cases)):
        raise ValueError("fixed benchmark goal tokens or cases differ from the archived run")
    scorers = [adapter_goal_cost(adapter, t) if adapter is not None else experience._raw_cost(t)
               for t in saved["tokens"].to(runtime.device)]
    stage = "benchmark_band_4"
    condition = dict(collector=checkpoint["checkpoint_sha256"], reference=digest(old),
                     goal_sha256=old_stage["goal_sha256"])

    def execute(case):
        result = runtime.run_case(case, latent_cost_fn=scorers[case["goal_index"]],
                                  journal_stage=stage, record_positions=True)
        result["reached_other_room"] = case["route"] in crossings(result["primitive_positions"])
        return result

    records = run_stage(runtime.root, stage, cases, runtime.runtime_id, execute, condition=condition)
    rows = [
        dict(case=case, previous_success=before["success"], success=after["success"],
             previous_crossed=before["reached_other_room"], crossed=after["reached_other_room"],
             positions=after["primitive_positions"])
        for case, before, after in zip(cases, old, records, strict=True)
    ]
    write_json(runtime.root / "band_4_fixed_benchmark.json", dict(reference=str(reference_run), rows=rows))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("stage", "compare", "benchmark"))
    for field in ("output-dir", "world-model-checkpoint", "encoder-checkpoint", "dino-wm-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--band", type=int, choices=range(1, len(BANDS) + 1))
    parser.add_argument("--starts-per-goal", type=int, default=6)
    parser.add_argument("--right-goals-run", type=Path)
    parser.add_argument("--extra-left-goal", nargs=2, type=float, action="append", default=[], metavar=("X", "Y"),
                        help="Append a left-room goal to the expanded manifest. Repeat for each new goal.")
    parser.add_argument("--benchmark-run", type=Path)
    parser.add_argument("--failure-pairs", type=int, default=0,
                        help="Maximum later failure frames to rank above the start cost.")
    parser.add_argument("--rounds-per-band", type=int, default=1,
                        help="Use two or more rounds for one-macro MPC with success-only temporal labels.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fresh-loop-curriculum", action="store_true",
                      help="Use the fixed twelve goals, loop filtering and 15%% failure sampling cap.")
    mode.add_argument("--half-route-feedback", action="store_true",
                      help="Use the right room and rank failed observations against successful routes.")
    mode.add_argument("--full-route-feedback", action="store_true",
                      help="Use both rooms and rank failed observations against successful routes.")
    parser.add_argument("--grid-starts", action="store_true",
                        help="Use the hand-crafted lattice as starts for full-map route feedback.")
    args = parser.parse_args(argv)
    route_feedback = args.half_route_feedback or args.full_route_feedback
    fresh = args.fresh_loop_curriculum or route_feedback
    if args.grid_starts and not args.full_route_feedback:
        parser.error("--grid-starts requires --full-route-feedback")
    if args.operation == "stage" and args.band is None:
        parser.error("stage requires --band")
    rounds = args.rounds_per_band
    if rounds < 1:
        parser.error("--rounds-per-band must be positive")
    if args.right_goals_run is not None and rounds < 2:
        parser.error("expanded goals require at least two rounds per band")
    if args.extra_left_goal and args.right_goals_run is None:
        parser.error("--extra-left-goal requires --right-goals-run")
    if args.failure_pairs < 0 or (args.failure_pairs and rounds < 2):
        parser.error("--failure-pairs must be nonnegative and requires repeated rounds")
    if args.operation == "benchmark" and (args.right_goals_run is None or args.benchmark_run is None):
        parser.error("benchmark requires --right-goals-run and --benchmark-run")
    if fresh:
        if (args.band is None or rounds < 2 or args.right_goals_run is not None or args.extra_left_goal
                or args.failure_pairs or args.benchmark_run is not None or args.operation == "benchmark"):
            parser.error("fresh curricula require --band and repeated rounds, with their own goals and feedback")
        if args.fresh_loop_curriculum:
            args.failure_pairs = 4
    sources = [args.world_model_checkpoint, args.dino_wm_root]
    sources.extend(p for p in (args.right_goals_run, args.benchmark_run) if p is not None)
    args.output_dir, _ = locations(args.output_dir, args.encoder_checkpoint, sources)
    collection_maps = args.right_goals_run is not None
    goals = None
    reset_region = None
    if collection_maps:
        goals = expanded_goals(read(args.right_goals_run / "experiment.json")["manifest"],
                               extra_left=args.extra_left_goal)
    extra_goal_count = len(args.extra_left_goal)
    if args.fresh_loop_curriculum or args.full_route_feedback:
        goals = [*make_manifest()["goals"][:3], *map(list, FRESH_RIGHT_GOALS), *map(list, EXTRA_LEFT_GOALS)]
        extra_goal_count = len(EXTRA_LEFT_GOALS)
        collection_maps = True
    if args.half_route_feedback:
        from src.mylonas_ijepa_wm import experience_half_square

        goals = list(map(list, FRESH_RIGHT_GOALS))
        reset_region = experience_half_square.RESET_REGION
        collection_maps = True
    if args.grid_starts:
        manifest = make_grid_manifest(goals, rounds=rounds)
    elif rounds > 1:
        manifest = make_repeated_manifest(args.starts_per_goal, rounds=rounds, goals=goals,
                                          extra_goal_count=extra_goal_count, reset_region=reset_region)
        kind = "experience_mpc_more_goals_full_square_v1" if collection_maps else "experience_mpc_repeated_full_square_v1"
    else:
        manifest = make_manifest(args.starts_per_goal)
        kind = "experience_square_v1"
    failure_selector = None
    rank_failures = None
    if args.failure_pairs:
        manifest["failure_pair_cap"] = args.failure_pairs
        failure_selector = partial(sample_failed_frames, cap=args.failure_pairs)
        kind = "experience_mpc_failure_feedback_full_square_v1"
    success_filter = temporal_filter if rounds > 1 else filter_steps
    if args.fresh_loop_curriculum:
        kind = "experience_mpc_loop_failure_cap_full_square_v1"
        manifest.update(failure_draw_cap=.15, loop_radius=.5)
        success_filter = partial(loop_filtered_steps, radius=manifest["loop_radius"])
    if route_feedback:
        from src.mylonas_ijepa_wm import experience_route_feedback

        scope = "half" if args.half_route_feedback else "full"
        kind = (f"experience_mpc_route_feedback_{scope}_grid_starts_v1" if args.grid_starts
                else f"experience_mpc_route_feedback_{scope}_square_v1")
        manifest["loop_radius"] = .5
        domain = _legal_positions
        if args.half_route_feedback:
            manifest["reset_region"] = list(reset_region)
            domain = experience_half_square._right_room
        success_filter = partial(loop_filtered_steps, radius=manifest["loop_radius"], domain=domain)
        failure_selector = partial(experience_route_feedback.sample_frames, goal_ids=manifest["goal_ids"], domain=domain)
        rank_failures = partial(experience_route_feedback.label_failures, domain=domain)
    config = dict(kind=kind, manifest=manifest, fitting=FITTING)
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
            first = (args.band - 1) * rounds
            for index in range(first, first + rounds):
                result = experience.collect_and_fit(
                    runtime, manifest, design, FITTING, None, index,
                    filter_steps=success_filter,
                    summarize_round_episode=summarize_round_episode if rounds > 1 else None,
                    sample_failure=failure_selector,
                    rank_failures=rank_failures,
                )
            return result
        if args.operation == "benchmark":
            return benchmark_band4(runtime, manifest, args.benchmark_run)
        return compare(runtime, manifest, design, band_count=args.band, collection_maps=collection_maps,
                       single_band=fresh)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
