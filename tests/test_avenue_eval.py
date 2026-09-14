import numpy as np
import pytest

from vjepa.eval.avenue_eval import frame_scores, postprocess


def fake_video(n_sampled: int = 40, frame_step: int = 4, hot_window: int = 5) -> dict:
    starts = np.arange(0, n_sampled - 16 + 1, 2)
    tokens = np.full((len(starts), 4, 14, 14), 0.5, dtype=np.float16)  # context 8 -> 4 predicted slices
    tokens[hot_window] = 0.9
    return {"starts": starts, "n_sampled": n_sampled, "frame_step": frame_step, "tokens_c8": tokens}


def test_hot_window_marks_the_frames_it_predicts_at_full_frame_rate():
    data = fake_video()
    scores = frame_scores(data, context=8, token_agg="mean", total_frames=160)
    assert scores.shape == (160,)
    start = data["starts"][5]
    predicted = slice((start + 8) * 4, (start + 16) * 4)  # sampled frames s+8..s+15, x4 at full rate
    assert scores[predicted].min() > scores[:predicted.start].max()
    assert np.isfinite(scores).all()


def test_frame_scores_pad_to_the_label_length():
    scores = frame_scores(fake_video(), context=8, token_agg="max", total_frames=163)
    assert len(scores) == 163 and np.isfinite(scores).all()


def test_postprocess_scales_to_unit_range():
    s = postprocess(np.array([3.0, 5.0, 4.0, 9.0]), sigma=0)
    assert s.min() == 0.0 and s.max() == pytest.approx(1.0)
