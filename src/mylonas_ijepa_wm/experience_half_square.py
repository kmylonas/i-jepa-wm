"""Success-only collection in square bands across the right room."""
import numpy as np

from src.mylonas_ijepa_wm import experience_curriculum as experience
from src.mylonas_ijepa_wm import geodesic_radial as wall
from src.mylonas_ijepa_wm.curriculum_protocol import PLANNING
from src.mylonas_ijepa_wm.experience_square import BANDS as WIDE_BANDS, _rectangles


HALF_WIDTHS = (*range(6, 43, 4), 48)
BANDS = tuple(zip((0, *HALF_WIDTHS[:-1]), HALF_WIDTHS))
RESET_REGION = (35.01, 56.5, 7.5, 56.5)


def make_manifest(design, starts_per_goal=6, *, bands=BANDS, seed=experience.SEED):
    if starts_per_goal < 1:
        raise ValueError("starts per goal must be positive")
    rng = np.random.default_rng(seed)
    points = np.asarray(design["points"])
    goals = experience._goal_ids(design)
    stages = []
    for index, (inner, outer) in enumerate(bands):
        cases = []
        for g, goal_id in enumerate(goals):
            goal = points[goal_id]
            rectangles = _rectangles(goal, inner, outer, regions=(RESET_REGION,))
            if not len(rectangles):
                continue
            areas = np.prod(rectangles[:, 1] - rectangles[:, 0], axis=1)
            weights = areas / areas.sum()
            for j in range(starts_per_goal):
                for _ in range(10000):
                    lo, hi = rectangles[rng.choice(len(areas), p=weights)]
                    start = rng.uniform(lo, hi)
                    if np.linalg.norm(start - goal) >= 4.5:
                        break
                else:
                    raise ValueError("could not sample a start outside the success radius")
                cases.append(dict(case_id=f"b{index + 1}_g{g:02d}_s{j:02d}", split="train", band=index + 1,
                                  goal_index=g, start=start.tolist(), goal=goal.tolist(), layout=list(wall.LAYOUT),
                                  seed=int(rng.integers(0, 2**31 - 1)), run_seed=experience.SEED))
        stages.append(cases)
    return dict(bands=[list(b) for b in bands], goal_ids=goals, stages=stages, starts_per_goal=starts_per_goal,
                planning=dict(PLANNING), seed=experience.SEED, pair_cap=experience.PAIR_CAP,
                reset_region=list(RESET_REGION))


def make_repeated_manifest(design, starts_per_goal, *, bands, rounds):
    first = make_manifest(design, starts_per_goal, bands=bands)
    later = (make_manifest(design, starts_per_goal, bands=bands, seed=[experience.SEED, number])
             for number in range(1, rounds))
    return experience.repeat_manifest(first, later)


def _right_room(xy):
    lo, hi = wall.BOUNDS
    return ((xy[:, 0] > wall.WALL_RIGHT) & (xy >= lo).all(axis=1) & (xy <= hi).all(axis=1))


def filter_steps(raw, case, design, cap):
    return experience._temporal_filter(raw, case, design, cap, domain=_right_room)
