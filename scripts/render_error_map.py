"""Side-by-side video: the Avenue clip, where V-JEPA's prediction error actually lands, and the same error
measured against each patch's own history.

Three panels, because the raw map on its own is misleading. The README records that raw per-patch error puts
LESS of its top-5% heat inside the annotated anomaly than chance (0.8-6.3% vs 5.2-9.6%): it is drawn to
background texture that is always hard to predict. Normalizing each patch over its own history
(surprise.normalize_over_time) does 2-3x better than chance on videos 03/04/05, though not on 07.

  left    the clip, with the annotated anomaly outlined
  middle  raw token error, the quantity the AUC is computed from
  right   the same error as a z-score against that patch's own history

This is a diagnostic, not a localizer, and it is built from the same non-causal frame mapping as the published
curve: a window's error is spread over every frame it predicts, so heat appears slightly before an event.

Usage: uv run python scripts/render_error_map.py [stem] [start] [end]
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
from vjepa.viz.render import to_uint8, upsample, write_mp4

STEM = sys.argv[1] if len(sys.argv) > 1 else "04"
A, B = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (250, 520)
WINDOW, TUBELET = 16, 2
TH = Theme()
PANEL_H, CURVE_H, HEAD_H = 360, 150, 54


def frame_maps(data: dict, n_sampled: int) -> np.ndarray:
    """Cached token errors (W, S, gh, gw) -> a (n_sampled, gh, gw) map, as SurpriseResult.frame_heatmaps does."""
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
    a = alpha * s[..., None]  # cool regions stay transparent so the scene reads through
    return to_uint8((1 - a) * view / 255.0 + a * rgb)


def outline(view: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return view
    edge = ndimage.binary_dilation(mask, iterations=2) & ~ndimage.binary_erosion(mask, iterations=1)
    out = view.copy()
    out[edge] = (255, 60, 60)
    return out


def main() -> None:
    data = dict(np.load(f"outputs/avenue/vitl16/fs4_stride2/{STEM}.npz"))
    labels = frame_labels(STEM)
    frames, fps = read_video(ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi")
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    step, n_sampled = int(data["frame_step"]), int(data["n_sampled"])

    raw_s = frame_maps(data, n_sampled)
    rel_s = normalize_over_time(raw_s, skip=PRIMARY["context"])
    raw = np.repeat(raw_s, step, axis=0)[: len(frames)]
    rel = np.repeat(rel_s, step, axis=0)[: len(frames)]
    curve = postprocess(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], len(labels)), PRIMARY["sigma"])

    # both colour scales come from the clip's own distribution rather than hand-picked numbers
    body, zbody = raw[PRIMARY["context"] * step :], rel[PRIMARY["context"] * step :]
    r_lo, r_hi = np.percentile(body, 40), np.percentile(body, 99.5)
    z_lo, z_hi = np.percentile(zbody, 80), np.percentile(zbody, 99.5)
    W = frames.shape[2] * 3
    H = HEAD_H + PANEL_H + CURVE_H
    f_small, f_lab = font(17), font(19, bold=True)
    seg = curve[A:B]

    def compose():
        for i in range(A, B):
            view = np.asarray(Image.fromarray(frames[i]).resize((frames.shape[2], PANEL_H)))
            m = masks[0, i].astype(bool) if i < masks.shape[1] else np.zeros(view.shape[:2], bool)
            panels = [outline(view, m), heat(view, raw[i], r_lo, r_hi, "inferno"),
                      heat(view, rel[i], z_lo, z_hi, "viridis")]
            canvas = Image.new("RGB", (W, H), TH.bg)
            for k, p in enumerate(panels):
                canvas.paste(Image.fromarray(p), (k * frames.shape[2], HEAD_H))
            d = ImageDraw.Draw(canvas)
            d.text((14, 12), f"CUHK Avenue test video {STEM} - where V-JEPA's prediction error lands",
                   TH.fg, font=f_lab)
            d.text((14, 33), "zero-shot, public ViT-L/16 - a diagnostic, not a localizer", TH.muted, font=f_small)
            for k, t in enumerate(["video (annotated anomaly outlined)", "raw token error",
                                   "error vs each patch's own history"]):
                d.text((k * frames.shape[2] + 10, HEAD_H + PANEL_H - 24), t, TH.fg, font=f_small)
                if k:
                    d.line([(k * frames.shape[2], HEAD_H), (k * frames.shape[2], HEAD_H + PANEL_H)], TH.bg, 2)

            # surprise curve with the annotated span shaded and a playhead
            y0, y1 = HEAD_H + PANEL_H + 34, H - 26
            sx = lambda j: 14 + j * (W - 28) / max(len(seg) - 1, 1)
            sy = lambda v: y1 - v * (y1 - y0)
            e = np.flatnonzero(np.diff(np.r_[0, labels[A:B].astype(int), 0]))
            for a, b in zip(e[::2], e[1::2]):
                d.rectangle([sx(a), y0, sx(b), y1], fill=TH.band)
            d.line([(14, y1), (W - 14, y1)], TH.grid, 1)
            d.line([(sx(j), sy(v)) for j, v in enumerate(seg)], TH.accent, 2)
            d.line([(sx(i - A), y0), (sx(i - A), y1)], TH.fg, 1)
            d.ellipse([sx(i - A) - 4, sy(seg[i - A]) - 4, sx(i - A) + 4, sy(seg[i - A]) + 4], fill=TH.accent)
            d.text((14, y0 - 22), "surprise (top-5% token error, smoothed)", TH.accent, font=f_small)
            note = "shaded = labeled anomalous   |   y-axis normalized over the whole video"
            d.text((W - 14 - d.textlength(note, font=f_small), y0 - 22), note, TH.muted, font=f_small)
            yield np.asarray(canvas)

    out = write_mp4(compose(), f"outputs/drafts/avenue_{STEM}_errormap.mp4", fps=fps)
    print(f"wrote {out}  frames {A}-{B}  raw scale [{r_lo:.3f}, {r_hi:.3f}]  z scale [{z_lo:.2f}, {z_hi:.2f}]")


if __name__ == "__main__":
    main()
