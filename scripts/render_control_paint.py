"""Side-by-side video for the CONTROL paint-out: the runner's mask replayed at a quiet time.

The range covers both events, so one video shows the contrast directly:

  frames 379-428  the real anomaly. UNTOUCHED in this test, so the runner (red box) is present on both sides
                  and the two curves sit on top of each other.
  frames 500-548  the control. The runner's own mask sequence is replayed here over ordinary pedestrians, and
                  only this stretch differs between the two panels.

Same absolute, zoomed axis as the runner test. Usage: uv run python -m scripts.render_control_paint [start] [end]
"""

import sys

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw

from vjepa.data.avenue import ROOT
from vjepa.video import read_video
from vjepa.viz.layout import Theme, font
from vjepa.viz.render import outline_mask, write_mp4
from scripts.control_paint_out import CONTROL
from scripts.paint_out_runner import ERASE, STEM, background, mask_sequence, paint_masks

A, B = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (340, 600)
TH = Theme()
PANEL_H, CURVE_H, HEAD_H = 360, 168, 54
CYAN = (88, 166, 255)


def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, fps = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    seq = mask_sequence(masks, ERASE)
    painted = paint_masks(frames, seq, CONTROL[0], background(frames))
    d0 = np.load("outputs/investigations/control_paint_out.npz")
    c_o, c_c = d0["c_orig"], d0["c_ctrl"]

    fw = frames.shape[2]
    W, H = fw * 2, HEAD_H + PANEL_H + CURVE_H
    f_small, f_lab, f_tiny = font(17), font(19, bold=True), font(14)
    lo = min(c_o[A:B].min(), c_c[A:B].min()) - 0.002
    hi = max(c_o[A:B].max(), c_c[A:B].max()) + 0.002
    y0, y1 = HEAD_H + PANEL_H + 40, H - 30
    sx = lambda j: 46 + j * (W - 74) / max(B - A - 1, 1)
    sy = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)

    def compose():
        for i in range(A, B):
            canvas = Image.new("RGB", (W, H), TH.bg)
            anom = masks[0, i].astype(bool) if i < masks.shape[1] else np.zeros(frames.shape[1:3], bool)
            ctrl = seq[i - CONTROL[0]] if CONTROL[0] <= i < CONTROL[0] + len(seq) else np.zeros_like(anom)
            for k, src in enumerate((frames, painted)):
                view = outline_mask(outline_mask(src[i], anom), ctrl, CYAN if k == 0 else (70, 96, 130))
                canvas.paste(Image.fromarray(view).resize((fw, PANEL_H)), (k * fw, HEAD_H))
            d = ImageDraw.Draw(canvas)
            d.text((14, 12), "Control: does erasing things create a bump by itself?", TH.fg, font=f_lab)
            d.text((14, 33), "the runner's own mask replayed at a quiet time - the real anomaly is left alone here",
                   TH.muted, font=f_small)
            for k, t in enumerate(["original", "control region erased"]):
                d.text((k * fw + 10, HEAD_H + PANEL_H - 24), t, TH.fg, font=f_small)
            tag = "red = annotated anomaly (untouched)    blue = erased control region"
            d.text((W - 14 - d.textlength(tag, font=f_small), 33), tag, TH.muted, font=f_small)
            d.line([(fw, HEAD_H), (fw, HEAD_H + PANEL_H)], TH.bg, 2)

            for (a, b), col, name in ((ERASE, TH.band, "real anomaly (untouched)"), (CONTROL, (30, 56, 84), "control erasure")):
                if b > A and a < B:
                    d.rectangle([sx(max(a, A) - A), y0, sx(min(b, B) - A), y1], fill=col)
                    d.text((sx(max(a, A) - A) + 4, y0 + 2), name, TH.muted, font=f_tiny)
            for v in (lo, (lo + hi) / 2, hi):
                d.line([(46, sy(v)), (W - 28, sy(v))], TH.grid, 1)
                d.text((6, sy(v) - 7), f"{v:.3f}", TH.muted, font=f_tiny)
            for c, col in ((c_o, TH.accent), (c_c, CYAN)):
                d.line([(sx(j), sy(v)) for j, v in enumerate(c[A:B])], col, 2)
            d.line([(sx(i - A), y0), (sx(i - A), y1)], TH.fg, 1)
            d.text((46, y0 - 24), "original", TH.accent, font=f_small)
            d.text((120, y0 - 24), "control erased", CYAN, font=f_small)
            note = "absolute top-5% token error - zoomed axis"
            d.text((W - 28 - d.textlength(note, font=f_small), y0 - 24), note, TH.muted, font=f_small)
            yield np.asarray(canvas)

    out = write_mp4(compose(), f"outputs/drafts/avenue_{STEM}_control.mp4", fps=fps)
    print(f"wrote {out}  frames {A}-{B}  y-axis [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
