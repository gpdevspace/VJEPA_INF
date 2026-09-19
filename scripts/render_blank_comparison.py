"""Render the post-2 opener: the same Surprise Meter on real footage and on a blank grey frame.

Post 1 plotted a min-max normalised curve, which rescales whatever range the video happens to have so that it
always fills the axis from 0 to 1. That made a 1.6% wobble look like the score swinging from empty to full.

This renders the same scoring pipeline -- top-5% token error, 8-frame context, 0.5 s smoothing -- on video 04
and on a clip of solid grey, both on ONE shared axis in absolute units. Both curves on one axis rather than two
side-by-side plots is deliberate: two plots with independent scales is exactly the trick being called out here.

The grey clip has nothing in it to predict, yet it scores 0.617 against video 04's 0.687.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.eval.avenue_eval import PRIMARY, frame_scores
from vjepa.video import read_video
from vjepa.viz.layout import Caption, Scene, Series, Theme, compose
from vjepa.viz.render import write_mp4

STEM, A, B = "04", 250, 520
GREY = 128


def absolute_curve(data: dict, n: int) -> np.ndarray:
    """Smoothed exactly as the eval does, but WITHOUT the per-video min-max that hides the real scale."""
    return gaussian_filter1d(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], n), PRIMARY["sigma"])


avenue = dict(np.load(f"outputs/avenue/vitl16/fs4_stride2/{STEM}.npz"))
blank = dict(np.load("outputs/investigations/blank_scores/grey.npz"))
labels = frame_labels(STEM)

real = absolute_curve(avenue, len(labels))[A:B]
grey = absolute_curve(blank, B - A)
frames, fps = read_video(ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi")
grey_frames = [np.full_like(frames[0], GREY)] * (B - A)

y = labels[A:B].astype(int)
edges = np.flatnonzero(np.diff(np.r_[0, y, 0]))
theme = Theme()
scene = Scene(
    title="V-JEPA scores an empty grey frame at 90% of a real one",
    subtitle="Identical pipeline · absolute scale, no min-max normalisation",
    panels=[list(frames[A:B]), grey_frames],
    panel_labels=["CUHK Avenue test video 04", "Solid grey — nothing to predict"],
    series=[Series(real, f"video 04 · {real.mean():.3f}", theme.accent),
            Series(grey, f"blank grey · {grey.mean():.3f}", theme.calm)],
    bands=list(zip(edges[::2].tolist(), edges[1::2].tolist())),
    band_label="labeled anomalous",
    y_range=(0.0, 0.8),
    y_axis_label="surprise (absolute)",
    captions=[
        Caption(0, 90, "Post 1 plotted this curve min-max normalised, so it always filled the axis."),
        Caption(90, 180, "On an absolute axis, video 04 moves 0.016 — about 2% of the score."),
        Caption(180, B - A, "A blank grey frame scores 0.617. Most of 'surprise' is a floor, not an event."),
    ],
    footer="Surprise: top-5% token error, 8-frame context, 0.5 s smoothing, ViT-L/16, absolute units",
)
out = write_mp4(compose(scene, width=1080, height=1080), "outputs/drafts/blank_comparison.mp4", fps=fps)
print(f"wrote {out}")
print(f"video 04: mean {real.mean():.4f}, span {real.max() - real.min():.4f}")
print(f"blank   : mean {grey.mean():.4f}, span {grey.max() - grey.min():.4f}  ({grey.mean() / real.mean():.1%} of video 04)")
