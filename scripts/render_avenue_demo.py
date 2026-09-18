"""Render the CUHK Avenue demo video for the first LinkedIn post (video 04, the clearest example).

Requires `uv run vjepa eval-avenue` to have already cached outputs/avenue/vitl16/fs4_stride2/04.npz.
Usage: uv run python scripts/render_avenue_demo.py
"""

import numpy as np

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.eval.avenue_eval import PRIMARY, frame_scores, postprocess
from vjepa.video import read_video
from vjepa.viz.layout import Caption, Scene, Series, Theme, compose
from vjepa.viz.render import write_mp4

TITLE = "V-JEPA Surprise Estimation: Zero-Shot Anomaly Detection"
STEM, (A, B) = "04", (250, 520)  # video stem and the frame span (in original video frame rate) to render

data = dict(np.load(f"outputs/avenue/vitl16/fs4_stride2/{STEM}.npz"))
labels = frame_labels(STEM)
curve = postprocess(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], len(labels)), PRIMARY["sigma"])
frames, fps = read_video(ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi")

y = labels[A:B].astype(int)
edges = np.flatnonzero(np.diff(np.r_[0, y, 0]))
theme = Theme()
scene = Scene(
    title=TITLE,
    subtitle="CUHK Avenue, test video 04 · zero-shot · public V-JEPA ViT-L/16",
    panels=[list(frames[A:B])],
    panel_labels=["V-JEPA watches 1.3 s and predicts the next 1.3 s, over and over"],
    series=[Series(curve[A:B], "surprise", theme.accent)],
    bands=list(zip(edges[::2].tolist(), edges[1::2].tolist())),
    band_label="labeled anomalous",
    captions=[
        Caption(0, 115, "No training on this camera. V-JEPA only predicts what happens next."),
        Caption(115, 215, "Someone sprints through the station, and the surprise climbs."),
        Caption(215, B - A, "Back to normal walking, and it settles."),
    ],
    footer="Surprise: top-5% token error, 8-frame context, 0.5 s smoothing",
)
out = write_mp4(compose(scene), f"outputs/drafts/avenue_{STEM}.mp4", fps=fps)
print(f"wrote {out}")
