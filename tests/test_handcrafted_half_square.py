import pytest

from src.mylonas_ijepa_wm import geodesic_radial as wall
from src.mylonas_ijepa_wm import handcrafted_half_square as half
from src.mylonas_ijepa_wm import radial_geometry


@pytest.fixture
def source_and_manifest():
    source = radial_geometry.make_design()
    source["goal_ids"] = [source["goal_id"]]
    return source, dict(goal_ids=source["goal_ids"], bands=[[0, 6]])


def test_legacy_grid_preserves_float32_border_positions(source_and_manifest):
    source, manifest = source_and_manifest
    old_points = [source["points"][i] for i in source["grid"]["point_ids"]]
    assert min(p[1] for p in old_points) < wall.BOUNDS[0]
    assert max(p[0] for p in old_points) > wall.BOUNDS[1]

    design = half.make_design(source, manifest)

    assert [design["points"][i] for i in design["grid"]["point_ids"]] == old_points
    assert all(wall.valid(design["points"][i]) for i in design["evaluation"]["point_ids"])


@pytest.mark.parametrize("point", [[35, 12], [50, 60]])
def test_legacy_grid_rejects_points_outside_right_room(source_and_manifest, point):
    source, manifest = source_and_manifest
    source["points"][source["grid"]["point_ids"][0]] = point

    with pytest.raises(ValueError, match="legal right room"):
        half.make_design(source, manifest)
