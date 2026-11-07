"""Shortest-path labels for the fixed-layout radial ranking experiment."""
import math
from itertools import combinations

import numpy as np
import pytest

from src.mylonas_ijepa_wm import geodesic_radial as radial


def test_shortest_path_uses_doorway_and_preserves_clear_segments():
    assert radial.distance((23, 30), (41, 30)) == pytest.approx(18)
    assert radial.distance((23, 12), (15, 15)) == pytest.approx(math.hypot(8, 3))
    goal = (48, 12)
    before = radial.distance((23, 12), goal)
    after = radial.distance((23, 16), goal)
    assert before == pytest.approx(math.hypot(6, 14) + 6 + math.hypot(13, 14))
    assert before - after > .5
    assert radial.distance(goal, (23, 16)) == pytest.approx(after)


def test_cumulative_pairs_include_all_band_and_direction_combinations():
    design = radial.make_design()
    assert design["goal"] == [15., 15.]
    nodes = design["training"]
    assert set(n["band"] for n in nodes) == set(range(len(radial.BANDS)))
    assert any(n["point"][0] > 35 for n in nodes)
    assert all(radial.valid(n["point"]) for n in nodes)
    prior = set()
    for stage in range(len(radial.BANDS)):
        pairs = set(radial.training_pairs(nodes, stage))
        assert prior <= pairs and pairs
        expected = set()
        for i, j in combinations(range(len(nodes)), 2):
            a, b = nodes[i], nodes[j]
            delta = a["distance"] - b["distance"]
            if max(a["band"], b["band"]) <= stage and abs(delta) >= .5:
                expected.add((i, j) if delta > 0 else (j, i))
        assert pairs == expected
        prior = pairs
    same_ring = [(i,j) for i,j in pairs if nodes[i]["radius"] == nodes[j]["radius"]]
    assert same_ring and any(nodes[i]["angle"] != nodes[j]["angle"] for i,j in same_ring)
    for n in nodes:
        assert np.linalg.norm(np.subtract(n["point"], design["goal"])) == pytest.approx(n["radius"])



def test_outer_stages_reach_the_whole_map_and_sample_the_farthest_corner():
    design = radial.make_design()
    assert design["bands"][:4] == [[6, 12], [12, 20], [20, 28], [28, 34]]
    outer_radius = design["bands"][-1][1]
    for i in design["grid"]:
        if i is not None:
            assert math.dist(design["points"][i], design["goal"]) <= outer_radius
    last_band = [n for n in design["training"] if n["band"] == len(design["bands"]) - 1]
    assert len(last_band) >= 2
    assert any(min(n["point"]) > 59 for n in last_band)
