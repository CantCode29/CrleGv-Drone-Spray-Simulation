import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import grid
import params


def test_light_at_the_canopy_top_is_almost_full():
    canopy = grid.build_canopy(cell_size=0.01)
    assert canopy.share_of_full_light[:, 0].max() > 0.99


def test_light_falls_with_depth():
    canopy = grid.build_canopy()
    first_column = canopy.share_of_full_light[0, :]
    assert np.all(np.diff(first_column) <= 0)


def test_light_matches_the_beer_lambert_formula():
    leaf_area_index, light_extinction_coefficient, slice_depth = 2.8, 0.575, 3.0
    canopy = grid.build_canopy(leaf_area_index=leaf_area_index,
                               light_extinction_coefficient=light_extinction_coefficient,
                               slice_depth=slice_depth)
    depth = canopy.row_depths[10]
    expected = np.exp(-light_extinction_coefficient * (leaf_area_index / slice_depth) * depth)
    assert abs(canopy.share_of_full_light[0, 10] - expected) < 1e-9


def test_leaves_alone_never_create_deep_shade():
    darkest_canopy_floor = np.exp(-params.LIGHT_EXTINCTION_COEFFICIENT.high * params.LEAF_AREA_INDEX.high)
    assert darkest_canopy_floor > params.DEEP_SHADE_LIGHT_LIMIT.value


def test_fruit_creates_deep_shade():
    canopy = grid.build_canopy()
    assert canopy.deep_shade.any()
