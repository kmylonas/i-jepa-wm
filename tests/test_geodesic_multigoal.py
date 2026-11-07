"""The multigoal geodesic run holds out references from every fitting role."""
from src.mylonas_ijepa_wm import geodesic_multigoal as geo


def test_shared_adapter_training_excludes_reserved_reference_and_image_alias():
    design = geo.make_design()
    hashes = [str(i) for i in range(len(design["points"]))]
    training_goal, heldout_goal = design["goal_ids"][0], design["heldout_goal_ids"][0]
    offered = design["samples"][0]
    leaked = offered[0]["id"]
    hashes[leaked] = hashes[heldout_goal]
    previous = set()
    for band in range(len(geo.BANDS)):
        pairs = set(geo.population(design, hashes, band))
        assert previous <= pairs
        assert training_goal in {g for g, _, _ in pairs}
        for g, farther, nearer in pairs:
            assert g in design["goal_ids"] and leaked not in (g, farther, nearer)
            assert heldout_goal not in (g, farther, nearer)
        previous = pairs
    assert set(design["goal_ids"]).isdisjoint(design["heldout_goal_ids"])


def test_goal_specific_labels_can_reverse_the_same_two_positions():
    from src.mylonas_ijepa_wm.geodesic_radial import distance
    a, b = (24, 12), (40, 12)
    assert distance(a, (15, 15)) < distance(b, (15, 15))
    assert distance(a, (49, 15)) > distance(b, (49, 15))
