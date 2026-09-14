import numpy as np

from vjepa.surprise import normalize_over_time


def test_relative_surprise_prefers_a_one_off_event_over_an_always_hard_patch():
    rng = np.random.default_rng(0)
    maps = 1.0 + rng.normal(0, 0.1, size=(50, 14, 14))
    maps[:, 0, 0] += 4.0  # a patch that is always hard to predict (texture, flicker)
    maps[30, 7, 7] += 2.0  # a single unusual moment somewhere else
    assert maps[30].argmax() == 0  # raw error points at the always-hard patch
    assert normalize_over_time(maps)[30].argmax() == 7 * 14 + 7  # relative error points at the event
