"""Stack the error-map video (where does the error land?) above the paint-out video (does erasing the runner
remove it?) into one clip, sharing a single 300-750 time window.

The naive way to stack two finished mp4s is `ffmpeg vstack`, but both source clips are already H.264 at crf 18,
so vstack-ing them decodes and re-encodes a second time. That compounds compression artifacts, worst on thin
colored lines against a dark background, exactly the surprise curves in both videos. This script instead
reuses each render script's own per-frame drawing code, composites the two panels straight from the source
arrays into one canvas per frame, and calls write_mp4 exactly once, so the output is a single generation of
compression, the same as either source video on its own.

Usage: uv run python scripts/render_stacked_errormap_paintout.py [start] [end]
"""

import sys

import matplotlib
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw
from scipy import ndimage

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.eval.avenue_eval import PRIMARY, frame_scores, postprocess
from vjepa.surprise import normalize_over_time
from vjepa.video import read_video
from vjepa.viz.layout import Theme, font
from vjepa.viz.render import outline_mask, to_uint8, upsample, write_mp4
from scripts.paint_out_runner import ERASE, STEM, paint

A, B = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (300, 750)
CTRL = (648, 692)
WINDOW, TUBELET = 16, 2
SS = 3  # supersample factor for curve panels only: PIL's line/text drawing has no anti-aliasing
TH = Theme()

# top block (error map): matches scripts/render_error_map.py exactly
T_PANEL_H, T_CURVE_H, T_HEAD_H = 360, 150, 54
T_W, T_H = 0, T_HEAD_H + T_PANEL_H + T_CURVE_H  # T_W set once frame width is known

# bottom block (paint-out): matches scripts/render_paint_out.py exactly
B_PANEL_H, B_CURVE_H, B_HEAD_H = 360, 168, 54


def frame_maps(data: dict, n_sampled: int) -> np.ndarray:
    errs = data[f"tokens_c{PRIMARY['context']}"].astype(np.float32)
    acc = np.zeros((n_sampled, *errs.shape[-2:]))
    cnt = np.zeros(n_sampled)
    for s, err in zip(data["starts"], errs):
        for j, sl in enumerate(err):
            f0 = s + PRIMARY["context"] + j * TUBELET
            acc[f0 : f0 + TUBELET] += sl
            cnt[f0 : f0 + TUBELET] += 1
    return acc / np.maximum(cnt, 1)[:, None, None]


def heat(view: np.ndarray, m: np.ndarray, lo: float, hi: float, cmap: str, alpha: float = 0.72) -> np.ndarray:
    s = np.clip((upsample(m, view.shape[:2]) - lo) / (hi - lo + 1e-8), 0, 1)
    rgb = matplotlib.colormaps[cmap](s)[..., :3]
    a = alpha * s[..., None]
    return to_uint8((1 - a) * view / 255.0 + a * rgb)


def outline(view: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return view
    edge = ndimage.binary_dilation(mask, iterations=2) & ~ndimage.binary_erosion(mask, iterations=1)
    out = view.copy()
    out[edge] = (255, 60, 60)
    return out


def main() -> None:
    global T_W

    # --- top block data (scripts/render_error_map.py) ---
    data = dict(np.load(f"outputs/avenue/vitl16/fs4_stride2/{STEM}.npz"))
    labels = frame_labels(STEM)
    frames, fps = read_video(ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi")
    top_masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    step, n_sampled = int(data["frame_step"]), int(data["n_sampled"])

    raw_s = frame_maps(data, n_sampled)
    rel_s = normalize_over_time(raw_s, skip=PRIMARY["context"])
    raw = np.repeat(raw_s, step, axis=0)[: len(frames)]
    rel = np.repeat(rel_s, step, axis=0)[: len(frames)]
    curve = postprocess(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], len(labels)), PRIMARY["sigma"])

    body, zbody = raw[PRIMARY["context"] * step :], rel[PRIMARY["context"] * step :]
    r_lo, r_hi = np.percentile(body, 40), np.percentile(body, 99.5)
    z_lo, z_hi = np.percentile(zbody, 80), np.percentile(zbody, 99.5)

    fw = frames.shape[2]
    T_W = fw * 3
    t_small, t_lab = font(17), font(19, bold=True)
    seg = curve[A:B]

    # --- bottom block data (scripts/render_paint_out.py) ---
    painted = paint(frames, top_masks, ERASE)
    d0 = np.load("outputs/investigations/paint_out_runner.npz")
    c_o, c_p = d0["c_orig"], d0["c_paint"]

    B_W = fw * 2
    b_small, b_lab, b_tiny = font(17), font(19, bold=True), font(14)
    lo = min(c_o[A:B].min(), c_p[A:B].min()) - 0.002
    hi = max(c_o[A:B].max(), c_p[A:B].max()) + 0.002

    # --- combined canvas ---
    W = T_W  # 1920, wider than the bottom block (1280)
    x_off = (W - B_W) // 2  # center the narrower bottom block, true size, no scaling
    H = T_H + (B_HEAD_H + B_PANEL_H + B_CURVE_H)

    def top_canvas(i: int) -> Image.Image:
        view = np.asarray(Image.fromarray(frames[i]).resize((fw, T_PANEL_H)))
        m = top_masks[0, i].astype(bool) if i < top_masks.shape[1] else np.zeros(view.shape[:2], bool)
        panels = [outline(view, m), heat(view, raw[i], r_lo, r_hi, "inferno"),
                  heat(view, rel[i], z_lo, z_hi, "viridis")]
        canvas = Image.new("RGB", (T_W, T_H), TH.bg)
        for k, p in enumerate(panels):
            canvas.paste(Image.fromarray(p), (k * fw, T_HEAD_H))
        d = ImageDraw.Draw(canvas)
        d.text((14, 12), f"CUHK Avenue test video {STEM} - where V-JEPA's prediction error lands", TH.fg, font=t_lab)
        d.text((14, 33), "zero-shot, public ViT-L/16 - a diagnostic, not a localizer", TH.muted, font=t_small)
        for k, t in enumerate(["video (annotated anomaly outlined)", "raw token error",
                               "error vs each patch's own history"]):
            d.text((k * fw + 10, T_HEAD_H + T_PANEL_H - 24), t, TH.fg, font=t_small)
            if k:
                d.line([(k * fw, T_HEAD_H), (k * fw, T_HEAD_H + T_PANEL_H)], TH.bg, 2)

        # the curve is drawn supersampled and downsampled with a high-quality filter: PIL's ImageDraw.line has
        # no anti-aliasing, so a straight draw at native resolution leaves every diagonal segment stair-stepped
        cw, ch = T_W, T_CURVE_H
        cimg = Image.new("RGB", (cw * SS, ch * SS), TH.bg)
        dc = ImageDraw.Draw(cimg)
        cy0, cy1 = 34 * SS, (ch - 26) * SS
        sx = lambda j: (14 + j * (T_W - 28) / max(len(seg) - 1, 1)) * SS
        sy = lambda v: cy1 - v * (cy1 - cy0)
        cf = font(17 * SS)
        e = np.flatnonzero(np.diff(np.r_[0, labels[A:B].astype(int), 0]))
        for a, b in zip(e[::2], e[1::2]):
            dc.rectangle([sx(a), cy0, sx(b), cy1], fill=TH.band)
        dc.line([(14 * SS, cy1), (cw * SS - 14 * SS, cy1)], TH.grid, SS)
        dc.line([(sx(j), sy(v)) for j, v in enumerate(seg)], TH.accent, 2 * SS, joint="curve")
        dc.line([(sx(i - A), cy0), (sx(i - A), cy1)], TH.fg, SS)
        dc.ellipse([sx(i - A) - 4 * SS, sy(seg[i - A]) - 4 * SS, sx(i - A) + 4 * SS, sy(seg[i - A]) + 4 * SS],
                   fill=TH.accent)
        dc.text((14 * SS, 12 * SS), "surprise (top-5% token error, smoothed)", TH.accent, font=cf)
        note = "shaded = labeled anomalous   |   y-axis normalized over the whole video"
        dc.text((cw * SS - 14 * SS - dc.textlength(note, font=cf), 12 * SS), note, TH.muted, font=cf)
        canvas.paste(cimg.resize((cw, ch), Image.LANCZOS), (0, T_HEAD_H + T_PANEL_H))
        return canvas

    def bottom_canvas(i: int) -> Image.Image:
        canvas = Image.new("RGB", (B_W, B_HEAD_H + B_PANEL_H + B_CURVE_H), TH.bg)
        m = top_masks[0, i].astype(bool) if i < top_masks.shape[1] else np.zeros(frames.shape[1:3], bool)
        for k, src in enumerate((frames, painted)):
            view = outline_mask(src[i], m, (255, 60, 60) if k == 0 else (120, 84, 84))
            canvas.paste(Image.fromarray(view).resize((fw, B_PANEL_H)), (k * fw, B_HEAD_H))
        d = ImageDraw.Draw(canvas)
        d.text((14, 12), "Does the runner cause the bump? Erase them and re-score.", TH.fg, font=b_lab)
        d.text((14, 33), f"CUHK Avenue video {STEM} - first anomaly erased, second left as a control",
               TH.muted, font=b_small)
        for k, t in enumerate(["original (red = annotated anomaly)", "runner erased (background median)"]):
            d.text((k * fw + 10, B_HEAD_H + B_PANEL_H - 24), t, TH.fg, font=b_small)
        d.line([(fw, B_HEAD_H), (fw, B_HEAD_H + B_PANEL_H)], TH.bg, 2)

        # same supersample-then-downsample treatment as the top block's curve, for the same reason
        cw, ch = B_W, B_CURVE_H
        cimg = Image.new("RGB", (cw * SS, ch * SS), TH.bg)
        dc = ImageDraw.Draw(cimg)
        cy0, cy1 = 40 * SS, (ch - 30) * SS
        csx = lambda j: (46 + j * (B_W - 74) / max(B - A - 1, 1)) * SS
        csy = lambda v: cy1 - (v - lo) / (hi - lo) * (cy1 - cy0)
        cb_small, cb_tiny = font(17 * SS), font(14 * SS)
        for (a, b), col, name in ((ERASE, TH.band, "erased"), (CTRL, (46, 60, 50), "control (untouched)")):
            if b > A and a < B:
                dc.rectangle([csx(max(a, A) - A), cy0, csx(min(b, B) - A), cy1], fill=col)
                dc.text((csx(max(a, A) - A) + 4 * SS, cy0 + 2 * SS), name, TH.muted, font=cb_tiny)
        for v in (lo, (lo + hi) / 2, hi):
            dc.line([(46 * SS, csy(v)), (cw * SS - 28 * SS, csy(v))], TH.grid, SS)
            dc.text((6 * SS, csy(v) - 7 * SS), f"{v:.3f}", TH.muted, font=cb_tiny)
        for c, col in ((c_o, TH.accent), (c_p, TH.calm)):
            dc.line([(csx(j), csy(v)) for j, v in enumerate(c[A:B])], col, 2 * SS, joint="curve")
        dc.line([(csx(i - A), cy0), (csx(i - A), cy1)], TH.fg, SS)
        dc.text((46 * SS, 16 * SS), "original", TH.accent, font=cb_small)
        dc.text((120 * SS, 16 * SS), "runner erased", TH.calm, font=cb_small)
        note = "absolute top-5% token error (no per-video rescaling) - note the zoomed axis"
        dc.text((cw * SS - 28 * SS - dc.textlength(note, font=cb_small), 16 * SS), note, TH.muted, font=cb_small)
        canvas.paste(cimg.resize((cw, ch), Image.LANCZOS), (0, B_HEAD_H + B_PANEL_H))
        return canvas

    def compose():
        for i in range(A, B):
            combined = Image.new("RGB", (W, H), TH.bg)
            combined.paste(top_canvas(i), (0, 0))
            combined.paste(bottom_canvas(i), (x_off, T_H))
            yield np.asarray(combined)

    out = write_mp4(compose(), f"outputs/drafts/avenue_{STEM}_errormap_paintout_stacked.mp4", fps=fps)
    print(f"wrote {out}  frames {A}-{B}  size {W}x{H}")


if __name__ == "__main__":
    main()
