import numpy as np
import pytest

from vjepa.data.intphys import match_pairs
from vjepa.eval.intphys_eval import TOKEN_AGGREGATES, evaluate, pair_metrics, pair_surprise, token_aggregates


def fake_scene(rng: np.random.Generator, gap: float) -> dict:
    """4 movies (0,1 possible; 2,3 impossible), impossible ones more surprising by `gap` at one window."""
    scores = rng.normal(0.5, 0.01, size=(4, 5, 17))
    scores[2:, :, 8] += gap
    return {
        **{f"scores_{tok}": scores for tok in TOKEN_AGGREGATES},
        "possible": np.array([True, True, False, False]),
        "pairs": np.array([(0, 2), (1, 3)]),
        "contexts": np.array([2, 4, 6, 8, 10]),
    }


def test_match_pairs_pairs_movies_that_diverge_last():
    base = np.zeros((10, 2, 2, 3), dtype=np.uint8)
    clips = [base.copy() for _ in range(4)]
    clips[1][2:] = 1  # diverges from 0 early
    clips[2][7:] = 2  # diverges from 0 late -> partner of 0
    clips[3][2:] = 3
    assert match_pairs(clips) == [(0, 2), (1, 3)]


def test_token_aggregates_localize_a_single_hot_token():
    errs = np.full((1, 7, 14, 14), 0.5)
    errs[0, 3, 5, 5] = 5.0
    agg = token_aggregates(errs)
    assert agg["max"][0] == 5.0 and agg["top5"][0] > agg["mean"][0] > 0.5


def test_clear_signal_gives_perfect_held_out_accuracy_and_noise_gives_chance():
    rng = np.random.default_rng(0)
    clear = {"O1": {f"{i:02d}": fake_scene(rng, gap=0.2) for i in range(1, 31)}}
    report = evaluate(clear, contexts=[2, 4, 6, 8, 10])
    assert report["held_out_pooled"]["relative_accuracy"] == 1.0
    assert report["held_out_pooled"]["pairs"] == 30

    noise = {f"{i:02d}": fake_scene(rng, gap=0.0) for i in range(1, 201)}
    m = pair_metrics(*pair_surprise(noise, None, "max"))
    assert m["relative_accuracy"] == pytest.approx(0.5, abs=0.07)
