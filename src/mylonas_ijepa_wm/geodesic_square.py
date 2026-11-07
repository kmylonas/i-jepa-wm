"""Rank hand-positioned observations on an expanding square lattice."""
from collections import defaultdict

from src.mylonas_ijepa_wm import geodesic_radial as radial


SPACING = 2
HALF_WIDTHS = (6, 12, 18, 24, 30, 36, 42, 46)


def make_design():
    goal = radial.GOAL
    bands = list(zip((0, *HALF_WIDTHS[:-1]), HALF_WIDTHS))
    nodes = []
    for band, (inner, outer) in enumerate(bands):
        offsets = range(-outer, outer + 1, SPACING)
        for y in offsets:
            for x in offsets:
                extent = max(abs(x), abs(y))
                point = [goal[0] + x, goal[1] + y]
                if extent > inner and radial.valid(point):
                    nodes.append(dict(point=point, band=band, radius=extent, distance=radial.distance(point)))

    axis = [radial.BOUNDS[0], *range(6, 60, 2), radial.BOUNDS[1]]
    points = [list(goal), *(node["point"] for node in nodes)]
    grid, contours = [], defaultdict(list)
    for y in axis:
        for x in axis:
            if radial.valid((x, y)):
                grid.append(len(points))
                extent = max(abs(x - goal[0]), abs(y - goal[1]))
                contours[extent].append(len(points))
                points.append([x, y])
            else:
                grid.append(None)
    return dict(geometry="square", spacing=SPACING, goal=list(goal), layout=list(radial.LAYOUT),
                bands=[list(b) for b in bands], training=nodes, points=points, axis=axis, grid=grid,
                rings=[dict(extent=extent, ids=ids) for extent, ids in sorted(contours.items())])


def run(output, env, encode, device, signature, *, updates=500):
    """Train a single-goal ranking adapter on eight expanding square grids."""
    return radial._run_design(output, env, encode, device, signature, make_design(), "geodesic_square_v1", updates)


if __name__ == "__main__":
    radial.main(runner=run)
