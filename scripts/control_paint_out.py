"""Control for the paint-out test: does erasing things create a bump by itself?

paint_out_runner.py showed that erasing the runner removes 51% of the surprise bump, leaving 49% standing. That
leftover has two possible explanations, and they lead to opposite conclusions:

  something else really is happening at frames 379-428, or
  the inpainting itself perturbs the score -- an artificially static, feathered patch where a motion-blurred
  person used to be is its own kind of unusual input.

The control replays the runner's OWN mask sequence at frames 500-548, a quiet stretch 72 frames after the first
anomaly and 99 before the second. Same mask shapes, same trajectory, same 49-frame duration, same erase-and-
inpaint operation -- but no anomaly. The band sweeps the middle of the frame, so it does erase real pedestrians
walking there, which is the point: it is the same operation applied to ordinary content.

  if the control score DROPS by about the same amount, erasing moving content lowers the score generally and the
      runner is not special
  if it RISES, inpainting manufactures error and the leftover 49% is contaminated
  if it is FLAT, the inpainting is clean and the leftover 49% is real content
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.device import DTYPES, pick_device
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import read_video
from scripts.paint_out_runner import ERASE, STEM, background, curve, mask_sequence, paint_masks

CONTROL = (500, 549)  # quiet: labeled anomalies are 379-428 and 648-692
OUT = Path("outputs/investigations/control_paint_out.json")
NPZ = Path("outputs/investigations/control_paint_out.npz")


def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, _ = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    labels = frame_labels(STEM)

    seq = mask_sequence(masks, ERASE)
    painted = paint_masks(frames, seq, CONTROL[0], background(frames))
    erased_px = float(np.mean([m.sum() for m in seq]))
    print(f"replaying the runner's {len(seq)} masks at frames {CONTROL[0]}-{CONTROL[1]} "
          f"(mean {erased_px:.0f} px/frame)")

    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])
    c_orig = curve(meter, frames, len(labels))
    c_ctrl = curve(meter, painted, len(labels))

    a, b = CONTROL
    base = np.r_[c_orig[:300], c_orig[750:]].mean()
    prev = json.loads(Path("outputs/investigations/paint_out_runner.json").read_text())
    res = {
        "control_span": CONTROL, "erased_px_per_frame": erased_px, "baseline_outside": float(base),
        "orig_mean": float(c_orig[a:b].mean()), "painted_mean": float(c_ctrl[a:b].mean()),
        "orig_peak": float(c_orig[a:b].max()), "painted_peak": float(c_ctrl[a:b].max()),
        "untouched_max_abs_diff": float(np.abs(c_orig - c_ctrl)[np.r_[0:a - 60, b + 60:len(labels)]].max()),
        "runner_test": prev["erased"],
    }
    d_ctrl = res["painted_mean"] - res["orig_mean"]
    d_run = prev["erased"]["painted_mean"] - prev["erased"]["orig_mean"]
    res["control_delta_mean"] = d_ctrl
    res["runner_delta_mean"] = d_run

    print(f"\nbaseline away from anomalies: {base:.4f}")
    print(f"CONTROL {a}-{b}: original mean {res['orig_mean']:.4f} -> erased {res['painted_mean']:.4f}  "
          f"({d_ctrl:+.4f})")
    print(f"RUNNER  {ERASE[0]}-{ERASE[1]}: original mean {prev['erased']['orig_mean']:.4f} -> erased "
          f"{prev['erased']['painted_mean']:.4f}  ({d_run:+.4f})")
    print(f"ratio control/runner: {d_ctrl / d_run:.2f}" if d_run else "")
    print(f"max drift on untouched frames: {res['untouched_max_abs_diff']:.4f}")

    np.savez(NPZ, c_orig=c_orig, c_ctrl=c_ctrl, span=np.array(CONTROL))
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
