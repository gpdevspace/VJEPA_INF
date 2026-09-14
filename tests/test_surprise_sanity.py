"""Surprise Meter sanity gates on real clips (skipped without converted weights or clips in data/dev_clips)."""

from pathlib import Path

import pytest
import torch

from vjepa.checkpoint import CHECKPOINT_DIR
from vjepa.device import pick_device
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video

CLIPS = sorted(Path("data/dev_clips").glob("*.mp4"))
pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "vitl16" / "predictor.safetensors").exists() or len(CLIPS) < 2,
    reason="needs converted vitl16 weights and two clips in data/dev_clips",
)


def first_window(path: Path) -> torch.Tensor:
    frames, _ = read_video(path, frame_step=4, max_frames=16)  # pretraining sampled every 4th frame
    return preprocess(frames)


def test_true_context_is_less_surprising_than_swapped_context():
    """Predicting clip A's future from clip B's past must be clearly worse than from A's own past.

    Scale: predicting all zeros for layer-normed targets gives L1 ~0.80, the checkpoint's final training loss is
    0.558, and window-to-window variation within a natural clip is ~0.02.
    """
    x = torch.stack([first_window(p) for p in CLIPS[:2]])
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device), device)
    true, _ = meter.score_windows(x, contexts=(8,))
    swapped, _ = meter.score_windows(x, contexts=(8,), context_x=x.flip(0))
    print(f"\ntrue-context L1 {true.ravel()}  swapped-context L1 {swapped.ravel()}")
    assert ((true > 0.45) & (true < 0.65)).all()  # near the training loss: masks and weights are wired right
    assert (swapped > 1.1 * true).all()
