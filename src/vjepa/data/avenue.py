"""CUHK Avenue: 16 normal training videos and 21 test videos (640x360, 25 fps) with pixel-level anomaly masks.

Download (public): http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip and
ground_truth_demo.zip, both extracted into data/avenue/.
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path("data/avenue")


def test_videos(root: Path = ROOT) -> list[Path]:
    return sorted((root / "Avenue Dataset" / "testing_videos").glob("*.avi"))


def train_videos(root: Path = ROOT) -> list[Path]:
    return sorted((root / "Avenue Dataset" / "training_videos").glob("*.avi"))


def frame_labels(video_stem: str, root: Path = ROOT) -> np.ndarray:
    """Per-frame anomaly flags for a test video: a frame is anomalous if any pixel of its mask is."""
    mat = sio.loadmat(root / "ground_truth_demo" / "testing_label_mask" / f"{int(video_stem)}_label.mat")
    masks = mat["volLabel"]
    return np.array([masks[0, i].any() for i in range(masks.shape[1])])
