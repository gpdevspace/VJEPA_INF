"""Control for half_erase_why.by_distance(): is the frame-wide movement specific to the runner, or would erasing
anything, anywhere, at any time, move distant tokens by a similar amount?

half_erase_why.by_distance() found that erasing the runner moves target tokens elsewhere in the frame by 43% as
much as the tokens the runner actually covers (0.301 vs 0.698). That is only evidence of a real, content-driven
effect if 0.301 is NOT just the background rate at which target embeddings drift for reasons that have nothing
to do with the runner -- ordinary lighting flicker, encoder noise, or the inpainting operation itself.

control_paint_out.py already checked this at the level of the aggregate top-5% score: replaying the runner's own
49 masks at a quiet time (frames 500-549, no labeled anomaly) raised the score by +0.0019, opposite in sign to
the runner erasure (-0.0064) and 3.4x smaller. This repeats that same control -- identical mask shapes, identical
erase-and-inpaint operation, only the moment in time differs -- but at the finer, per-token, inside/outside
resolution that by_distance() uses, which the aggregate score cannot rule out on its own.

If the quiet-time control ALSO shows outside tokens moving by ~43% as much as inside tokens, the frame-wide
effect is not specific to the runner and the original claim does not hold as stated.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image

from vjepa.data.avenue import ROOT
from vjepa.device import DTYPES, autocast, pick_device
from vjepa.eval.avenue_eval import PRIMARY
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter, time_masks
from vjepa.video import preprocess, read_video
from scripts.paint_out_runner import ERASE, STEM, STEP, STRIDE, background, mask_sequence, paint_masks

CONTROL = (500, 549)  # same span control_paint_out.py uses: quiet, no labeled anomaly
OUT = Path("outputs/investigations/control_half_erase.json")
WIN, C = 16, PRIMARY["context"]


@torch.no_grad()
def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, _ = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    seq = mask_sequence(masks, ERASE)  # the runner's own 49 mask shapes
    painted = paint_masks(frames, seq, CONTROL[0], background(frames))  # replayed at the quiet time

    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])
    xo, xp = preprocess(frames[::STEP]), preprocess(painted[::STEP])
    lo, hi = CONTROL[0] // STEP - WIN + 1, CONTROL[1] // STEP - C
    starts = [s for s in range(0, xo.shape[1] - WIN + 1, STRIDE) if lo <= s <= hi]
    print(f"{len(starts)} windows predict the control span (sampled starts {starts[0]}-{starts[-1]})")
    ctx, tgt = (m.to(device) for m in time_masks(C, WIN))

    def grid_mask(orig_frame: int) -> np.ndarray:
        """The REPLAYED mask (runner's shape, shifted to the control's timing) on the 14x14 token grid."""
        idx = orig_frame - CONTROL[0]
        if not (0 <= idx < len(seq)):
            return np.zeros((14, 14), bool)
        m = seq[idx].astype(np.float32)
        return np.asarray(Image.fromarray(m).resize((14, 14), Image.BILINEAR)) > 0.15

    ins, outs = [], []
    for i in range(0, len(starts), 4):
        sl = starts[i : i + 4]
        bo = torch.stack([xo[:, s : s + WIN] for s in sl]).to(device)
        bp = torch.stack([xp[:, s : s + WIN] for s in sl]).to(device)
        n = len(sl)
        cm = ctx.repeat(n, 1)
        with autocast(device, meter.dtype):
            ho = F.layer_norm(h := meter.model.target_encoder(bo), (h.size(-1),))[:, cm.shape[1] :].float()
            hp = F.layer_norm(h := meter.model.target_encoder(bp), (h.size(-1),))[:, cm.shape[1] :].float()
        d = (ho - hp).abs().mean(-1).cpu().numpy().reshape(n, -1, 14, 14)
        for b, s in enumerate(sl):
            for j in range(d.shape[1]):
                gm = grid_mask((s + C + j * 2) * STEP)
                ins.append(d[b, j][gm]); outs.append(d[b, j][~gm])
    inside = np.concatenate([x for x in ins if len(x)])
    outside = np.concatenate([x for x in outs if len(x)])
    ratio = outside.mean() / inside.mean()
    print(f"\ntarget representation movement, quiet-time control (same mask shapes, no real anomaly):")
    print(f"  tokens the replayed mask covers ({len(inside)}): {inside.mean():.4f}")
    print(f"  tokens elsewhere ({len(outside)}):                {outside.mean():.4f}   ({ratio:.0%} as much)")

    runner = json.loads(Path("outputs/investigations/half_erase_why.json").read_text())
    res = {
        "control_span": CONTROL, "d_target_inside_control": float(inside.mean()),
        "d_target_outside_control": float(outside.mean()), "outside_over_inside_control": float(ratio),
        "runner_d_target_inside": runner["d_target_inside_runner"],
        "runner_d_target_outside": runner["d_target_outside_runner"],
        "runner_outside_over_inside": runner["d_target_outside_runner"] / runner["d_target_inside_runner"],
    }
    print(f"\nfor comparison, the real runner: outside/inside = {res['runner_outside_over_inside']:.0%}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
