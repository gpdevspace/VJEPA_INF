"""Does the running person cause the surprise bump, even though their own patches are not hot?

person_vs_background.py found a pedestrian LOWERS their own cell's error (-0.026, in 75% of cells) and that at a
peak the background excess is flat in distance from the nearest person. So the peaks are whole-frame events. That
leaves the question this script answers for one clip: if the runner is removed from the pixels entirely, does the
bump go with them?

Avenue video 04 is the clean case. The runner enters at the right edge at frame 379 and leaves at the left edge
at frame 427, exactly the labeled span, so erasing that span makes them simply never appear -- no pop-in seam.
The video's SECOND anomaly (frames 648-692) is left untouched as an internal control: a working erasure should
drop the first peak and leave the second one standing.

The camera is fixed, so the background is a per-pixel median over the clip. The labeled mask is dilated and
feathered before compositing, to cover the person's outline and shadow.

Both curves are absolute top-5% token error. NO per-video min-max: that is what the published curve does, and it
would rescale the painted video to fill the same 0-1 range and hide the very effect being measured.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.device import DTYPES, pick_device
from vjepa.eval.avenue_eval import PRIMARY, frame_scores
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video

STEM, ERASE = "04", (379, 428)  # the first anomaly; the second (648-692) stays as a control
STEP, STRIDE, SIGMA = 4, 2, 12
DILATE, FEATHER = 7, 4
OUT = Path("outputs/investigations/paint_out_runner.json")
NPZ = Path("outputs/investigations/paint_out_runner.npz")


def background(frames: np.ndarray) -> np.ndarray:
    """Per-pixel median over the clip. The camera never moves, so this is the empty scene."""
    return np.median(frames[::4], axis=0).astype(np.uint8)


def paint(frames: np.ndarray, masks, span: tuple[int, int]) -> np.ndarray:
    bg = background(frames)
    out = frames.copy()
    for f in range(*span):
        m = masks[0, f].astype(bool)
        if not m.any():
            continue
        soft = ndimage.gaussian_filter(ndimage.binary_dilation(m, iterations=DILATE).astype(np.float32), FEATHER)
        a = np.clip(soft / max(soft.max(), 1e-6), 0, 1)[..., None]
        out[f] = (a * bg + (1 - a) * frames[f]).astype(np.uint8)
    return out


def curve(meter, frames: np.ndarray, total: int) -> np.ndarray:
    """Absolute top-5% error per original frame, smoothed but never min-max normalized."""
    sampled = frames[::STEP]
    res = meter.score_clip(preprocess(sampled), [PRIMARY["context"]], STRIDE)
    data = {"starts": res.starts, "n_sampled": len(sampled), "frame_step": STEP,
            f"tokens_c{PRIMARY['context']}": res.token_errors[0]}
    return gaussian_filter1d(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], total), SIGMA)


def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, fps = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    labels = frame_labels(STEM)
    painted = paint(frames, masks, ERASE)
    print(f"{STEM}: {len(frames)} frames, erased {ERASE[0]}-{ERASE[1]}")

    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])
    c_orig = curve(meter, frames, len(labels))
    c_paint = curve(meter, painted, len(labels))

    a, b = ERASE
    ctrl = (648, 692)
    base = np.r_[c_orig[:300], c_orig[750:]].mean()  # quiet stretches away from both anomalies
    res = {
        "erased_span": ERASE, "control_span": ctrl, "baseline_outside": float(base),
        "erased": {"orig_peak": float(c_orig[a:b].max()), "painted_peak": float(c_paint[a:b].max()),
                   "orig_mean": float(c_orig[a:b].mean()), "painted_mean": float(c_paint[a:b].mean())},
        "control": {"orig_peak": float(c_orig[ctrl[0]:ctrl[1]].max()), "painted_peak": float(c_paint[ctrl[0]:ctrl[1]].max())},
        "untouched_frames_max_abs_diff": float(np.abs(c_orig - c_paint)[np.r_[0:a-60, b+60:len(labels)]].max()),
    }
    e = res["erased"]
    rise_o, rise_p = e["orig_peak"] - base, e["painted_peak"] - base
    res["bump_removed_frac"] = float((rise_o - rise_p) / rise_o)

    print(f"\nbaseline away from both anomalies: {base:.4f}")
    print(f"ERASED span {a}-{b}:  original peak {e['orig_peak']:.4f}  painted {e['painted_peak']:.4f}")
    print(f"  rise above baseline: {rise_o:.4f} -> {rise_p:.4f}   ({res['bump_removed_frac']:.0%} of the bump removed)")
    print(f"CONTROL span {ctrl} (untouched): {res['control']['orig_peak']:.4f} -> {res['control']['painted_peak']:.4f}")
    print(f"max drift on untouched frames: {res['untouched_frames_max_abs_diff']:.4f}")

    np.savez(NPZ, c_orig=c_orig, c_paint=c_paint, painted_span=np.array(ERASE))
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT} and {NPZ}")


if __name__ == "__main__":
    main()
