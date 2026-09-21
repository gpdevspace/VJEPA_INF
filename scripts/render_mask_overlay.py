"""Side-by-side Avenue clip: original on the left, the same frames with the pixel-level anomaly mask on the right."""

import sys

import numpy as np
import scipy.io as sio

from vjepa.data.avenue import ROOT, test_videos
from vjepa.video import read_video
from vjepa.viz.render import write_mp4

STEM = sys.argv[1] if len(sys.argv) > 1 else "04"
RED = np.array([255, 40, 40], dtype=np.float32)

path = next(p for p in test_videos() if p.stem == STEM)
frames, fps = read_video(path, 1)
masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]


def compose():
    for i, f in enumerate(frames):
        m = masks[0, i].astype(bool)
        right = f.copy()
        right[m] = (0.5 * f[m] + 0.5 * RED).astype(np.uint8)
        yield np.concatenate([f, right], axis=1)


out = write_mp4(compose(), f"outputs/drafts/avenue_{STEM}_mask.mp4", fps=fps)
print(out, len(frames), fps, "anomalous frames:", sum(masks[0, i].any() for i in range(len(frames))))
