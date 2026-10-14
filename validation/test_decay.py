import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import params
import survival


def test_all_virus_is_active_at_the_moment_of_spraying():
    light_levels = np.linspace(0.0, 1.0, 11)
    assert np.allclose(survival.share_still_active(light_levels, 0.0), 1.0)


def test_half_remains_after_the_full_sun_half_life():
    full_sun = np.array([1.0])
    remaining = survival.share_still_active(full_sun, params.VIRUS_HALF_LIFE_IN_FULL_SUN.value)
    assert abs(remaining[0] - 0.5) < 1e-9


def test_half_remains_after_the_full_shade_half_life():
    full_shade = np.array([0.0])
    remaining = survival.share_still_active(full_shade, params.VIRUS_HALF_LIFE_IN_FULL_SHADE.value)
    assert abs(remaining[0] - 0.5) < 1e-9


def test_shade_always_keeps_more_virus_active_than_sun():
    for hours in params.CHECKPOINT_HOURS:
        assert survival.share_still_active(np.array([0.0]), hours)[0] > \
               survival.share_still_active(np.array([1.0]), hours)[0]


def test_more_light_always_leaves_less_virus_active():
    light_levels = np.linspace(0.0, 1.0, 50)
    remaining = survival.share_still_active(light_levels, 48.0)
    assert np.all(np.diff(remaining) < 0)
