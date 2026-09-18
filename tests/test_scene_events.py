"""Synthetic checks for the scene covariates used to rule out the motion (H3) and clutter (H4) confounds."""

import numpy as np
import pytest

from vjepa.eval.scene_events import background, covariates, foreground_masks

H, W, T = 40, 60, 20
BG_LEVEL, FG_LEVEL = 100.0, 200.0


def scene(present: range | list[int], size: int = 10, origin: tuple[int, int] = (5, 5)) -> np.ndarray:
    """A static grey scene with one bright square visible only on the given frames."""
    gray = np.full((T, H, W), BG_LEVEL, np.float32)
    r, c = origin
    for t in present:
        gray[t, r : r + size, c : c + size] = FG_LEVEL
    return gray


def test_background_is_the_scene_without_transients():
    bg = background(scene(range(6)))  # present in 6 of 20 frames, so the median ignores it
    assert np.allclose(bg, BG_LEVEL)


def test_blob_count_and_area_track_the_square():
    cov = covariates(scene(range(6)))
    assert list(cov["blob_count"][:6]) == [1] * 6
    assert list(cov["blob_count"][6:]) == [0] * (T - 6)
    assert cov["foreground_area"][0] == pytest.approx(100 / (H * W))
    assert cov["foreground_area"][6] == 0


def test_two_squares_are_two_blobs():
    gray = scene(range(6))
    gray[range(6), 25:35, 40:50] = FG_LEVEL
    assert list(covariates(gray)["blob_count"][:6]) == [2] * 6


def test_blobs_below_min_area_are_discarded():
    cov = covariates(scene(range(6), size=4))  # 16 px, under the 80 px floor
    assert cov["blob_count"].max() == 0


def test_motion_energy_is_zero_on_a_static_scene():
    assert np.allclose(covariates(scene([])) ["motion_energy"], 0)


def test_motion_energy_fires_only_where_the_frame_changes():
    cov = covariates(scene(range(6)), lag=4)
    # frames 0-3 are edge-padded from the frame-4-vs-0 difference, and 0..5 all contain the square
    assert cov["motion_energy"][4] == pytest.approx(0)
    assert cov["motion_energy"][6] > 0  # frame 6 (empty) against frame 2 (square)
    assert len(cov["motion_energy"]) == T


def test_foreground_masks_yield_one_entry_per_frame():
    assert len(list(foreground_masks(scene(range(6))))) == T
