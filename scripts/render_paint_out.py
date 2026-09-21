"""Side-by-side video for the paint-out test: the original clip against the one with the runner erased.

Both surprise curves are drawn on the SAME absolute axis, which is zoomed to the range the curves actually
occupy (about 0.677-0.697). That zoom is why the bump looks like anything at all: measured against zero it is a
2% wiggle on top of a floor of 0.68, and the axis is labelled with real values so the zoom is visible.

The first anomaly is erased; the second (frames 648-692) is left alone as a control and should be identical.
Usage: uv run python scripts/render_paint_out.py [start] [end]
"""

import sys

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw

from vjepa.data.avenue import ROOT
from vjepa.video import read_video
from vjepa.viz.layout import Theme, font
from vjepa.viz.render import write_mp4
from scripts.paint_out_runner import ERASE, STEM, paint

A, B = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (300, 750)
CTRL = (648, 692)
TH = Theme()
PANEL_H, CURVE_H, HEAD_H = 360, 168, 54


def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, fps = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    painted = paint(frames, masks, ERASE)
    d0 = np.load("outputs/investigations/paint_out_runner.npz")
    c_o, c_p = d0["c_orig"], d0["c_paint"]

    fw = frames.shape[2]
    W, H = fw * 2, HEAD_H + PANEL_H + CURVE_H
    f_small, f_lab, f_tiny = font(17), font(19, bold=True), font(14)
    lo = min(c_o[A:B].min(), c_p[A:B].min()) - 0.002
    hi = max(c_o[A:B].max(), c_p[A:B].max()) + 0.002

    y0, y1 = HEAD_H + PANEL_H + 40, H - 30
    sx = lambda j: 46 + j * (W - 74) / max(B - A - 1, 1)
    sy = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)

    def compose():
        for i in range(A, B):
            canvas = Image.new("RGB", (W, H), TH.bg)
            for k, src in enumerate((frames, painted)):
                canvas.paste(Image.fromarray(src[i]).resize((fw, PANEL_H)), (k * fw, HEAD_H))
            d = ImageDraw.Draw(canvas)
            d.text((14, 12), "Does the runner cause the bump? Erase them and re-score.", TH.fg, font=f_lab)
            d.text((14, 33), f"CUHK Avenue video {STEM} - first anomaly erased, second left as a control",
                   TH.muted, font=f_small)
            for k, t in enumerate(["original", "runner erased (background median)"]):
                d.text((k * fw + 10, HEAD_H + PANEL_H - 24), t, TH.fg, font=f_small)
            d.line([(fw, HEAD_H), (fw, HEAD_H + PANEL_H)], TH.bg, 2)

            for (a, b), col, name in ((ERASE, TH.band, "erased"), (CTRL, (46, 60, 50), "control (untouched)")):
                if b > A and a < B:
                    d.rectangle([sx(max(a, A) - A), y0, sx(min(b, B) - A), y1], fill=col)
                    d.text((sx(max(a, A) - A) + 4, y0 + 2), name, TH.muted, font=f_tiny)
            for v in (lo, (lo + hi) / 2, hi):
                d.line([(46, sy(v)), (W - 28, sy(v))], TH.grid, 1)
                d.text((6, sy(v) - 7), f"{v:.3f}", TH.muted, font=f_tiny)
            for c, col in ((c_o, TH.accent), (c_p, TH.calm)):
                d.line([(sx(j), sy(v)) for j, v in enumerate(c[A:B])], col, 2)
            d.line([(sx(i - A), y0), (sx(i - A), y1)], TH.fg, 1)
            d.text((46, y0 - 24), "original", TH.accent, font=f_small)
            d.text((120, y0 - 24), "runner erased", TH.calm, font=f_small)
            note = "absolute top-5% token error (no per-video rescaling) - note the zoomed axis"
            d.text((W - 28 - d.textlength(note, font=f_small), y0 - 24), note, TH.muted, font=f_small)
            yield np.asarray(canvas)

    out = write_mp4(compose(), f"outputs/drafts/avenue_{STEM}_paintout.mp4", fps=fps)
    print(f"wrote {out}  frames {A}-{B}  y-axis [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
