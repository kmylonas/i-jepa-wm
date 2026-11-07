"""The expanding square adds all legal intersections on one fixed lattice."""
from src.mylonas_ijepa_wm import geodesic_radial as radial
from src.mylonas_ijepa_wm import geodesic_square as square


def test_square_covers_corners_and_expands_without_resampling():
    design = square.make_design()
    nodes = design["training"]
    assert design["goal"] == [15., 15.]
    assert len(nodes) == len({tuple(n["point"]) for n in nodes})
    for band, (_, half_width) in enumerate(design["bands"]):
        offsets = range(-half_width, half_width + 1, 2)
        expected = {(15 + x, 15 + y) for x in offsets for y in offsets
                    if (x or y) and radial.valid((15 + x, 15 + y))}
        assert {tuple(n["point"]) for n in nodes if n["band"] <= band} == expected
        assert any(n["band"] == band for n in nodes)
    assert {(9, 9), (9, 21), (21, 9), (21, 21)} <= {
        tuple(n["point"]) for n in nodes if n["band"] == 0
    }
    assert (59, 59) in expected and (5, 5) in expected


def test_square_keeps_geodesic_pair_order_within_a_band():
    nodes = square.make_design()["training"]
    assert any(nodes[a]["band"] == nodes[b]["band"]
               and nodes[a]["distance"] - nodes[b]["distance"] >= .5
               for a, b in radial.training_pairs(nodes, 0))
