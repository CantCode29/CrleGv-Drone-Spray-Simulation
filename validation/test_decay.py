import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import survival
import params as P


def test_no_decay_at_time_zero():
    ratios = np.linspace(0.0, 1.0, 11)
    assert np.allclose(survival.viable_fraction(ratios, 0.0), 1.0)


def test_exposed_half_life():
    exposed = np.array([1.0])
    v = survival.viable_fraction(exposed, P.HALF_LIFE_EXPOSED.value)
    assert abs(v[0] - 0.5) < 1e-9


def test_shaded_half_life():
    shaded = np.array([0.0])
    v = survival.viable_fraction(shaded, P.HALF_LIFE_SHADED.value)
    assert abs(v[0] - 0.5) < 1e-9


def test_shade_always_preserves_more_than_sun():
    for h in P.CHECKPOINTS_H:
        assert survival.viable_fraction(np.array([0.0]), h)[0] > \
               survival.viable_fraction(np.array([1.0]), h)[0]


def test_monotonic_in_intensity():
    ratios = np.linspace(0.0, 1.0, 50)
    v = survival.viable_fraction(ratios, 48.0)
    assert np.all(np.diff(v) < 0)
