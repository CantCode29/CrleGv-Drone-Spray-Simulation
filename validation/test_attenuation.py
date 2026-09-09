import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import grid
import params as P


def test_intensity_starts_at_unity():
    c = grid.build(dx=0.01)
    assert c.intensity_ratio[:, 0].max() > 0.99


def test_intensity_decreases_with_depth():
    c = grid.build()
    column = c.intensity_ratio[0, :]
    assert np.all(np.diff(column) <= 0)


def test_beer_lambert_matches_closed_form():
    lai, k, depth = 2.8, 0.575, 3.0
    c = grid.build(lai_total=lai, k=k, depth=depth)
    z = c.z_centres[10]
    expected = np.exp(-k * (lai / depth) * z)
    assert abs(c.intensity_ratio[0, 10] - expected) < 1e-9


def test_leaf_attenuation_alone_never_reaches_refuge_threshold():
    """At the sourced citrus LAI range the canopy alone cannot reach 90%
    attenuation, so the refuge classification depends on discrete structure."""
    worst = np.exp(-P.EXTINCTION_COEFF.high * P.LAI_TOTAL.high)
    assert worst > P.REFUGE_THRESHOLD.value


def test_fruit_shading_creates_refuge():
    c = grid.build()
    assert c.refuge.any()
