"""Is the frame-wide movement real, or an artifact of the layer norm?

half_erase_why.py found that erasing the runner moves the target representation by 0.3009 per token even for
tokens nowhere near them. That was measured AFTER `F.layer_norm(h, (h.size(-1),))` in surprise.py, leaving open
whether the normalizer manufactures some of it.

Note what that call normalizes over: `normalized_shape=(D,)` is the LAST dim only, so every token is scaled by
its OWN mean and standard deviation across its 1024 channels. It pools nothing across tokens, so it cannot by
itself carry information from the runner to a distant patch. This script checks that reasoning against numbers
rather than trusting it, by splitting the post-norm movement into its two possible sources:

  direction   the token's raw features genuinely changed
  renorm      the raw features barely moved but the token's own mean/std shifted, rescaling everything

The isolation is exact: re-normalize the ORIGINAL raw token using the PAINTED token's mean and std. Whatever
that moves is attributable to the statistics alone, with the content held fixed.
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
from scripts.paint_out_runner import ERASE, STEM, STEP, STRIDE, paint

OUT = Path("outputs/investigations/prenorm_check.json")
WIN, C = 16, PRIMARY["context"]


def grid_mask(masks, orig_frame: int) -> np.ndarray:
    if not (0 <= orig_frame < masks.shape[1]):
        return np.zeros((14, 14), bool)
    m = masks[0, orig_frame].astype(np.float32)
    return np.asarray(Image.fromarray(m).resize((14, 14), Image.BILINEAR)) > 0.15


@torch.no_grad()
def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, _ = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    painted = paint(frames, masks, ERASE)
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])
    xo, xp = preprocess(frames[::STEP]), preprocess(painted[::STEP])
    lo, hi = ERASE[0] // STEP - WIN + 1, ERASE[1] // STEP - C
    starts = [s for s in range(0, xo.shape[1] - WIN + 1, STRIDE) if lo <= s <= hi]
    ctx, _ = (m.to(device) for m in time_masks(C, WIN))
    print(f"{len(starts)} windows over the erased span")

    keys = ("raw_mag", "d_raw", "d_norm", "d_renorm_only", "dmu", "dsigma")
    acc = {k: {"in": [], "out": []} for k in keys}
    for i in range(0, len(starts), 4):
        sl = starts[i : i + 4]
        bo = torch.stack([xo[:, s : s + WIN] for s in sl]).to(device)
        bp = torch.stack([xp[:, s : s + WIN] for s in sl]).to(device)
        n = len(sl)
        n_ctx = ctx.repeat(n, 1).shape[1]
        with autocast(device, meter.dtype):
            ro = meter.model.target_encoder(bo)[:, n_ctx:].float()
            rp = meter.model.target_encoder(bp)[:, n_ctx:].float()
        mo, so = ro.mean(-1, keepdim=True), ro.std(-1, keepdim=True, unbiased=False)
        mp, sp = rp.mean(-1, keepdim=True), rp.std(-1, keepdim=True, unbiased=False)
        no, np_ = (ro - mo) / (so + 1e-6), (rp - mp) / (sp + 1e-6)
        # the original content re-normalized with the painted token's statistics: statistics alone, content fixed
        renorm_only = (ro - mp) / (sp + 1e-6)
        vals = {
            "raw_mag": ro.abs().mean(-1), "d_raw": (ro - rp).abs().mean(-1),
            "d_norm": (no - np_).abs().mean(-1), "d_renorm_only": (no - renorm_only).abs().mean(-1),
            "dmu": ((mo - mp).abs() / (so + 1e-6)).squeeze(-1), "dsigma": ((so - sp).abs() / (so + 1e-6)).squeeze(-1),
        }
        shaped = {k: v.cpu().numpy().reshape(n, -1, 14, 14) for k, v in vals.items()}
        for b, s in enumerate(sl):
            for j in range(shaped["d_raw"].shape[1]):
                gm = grid_mask(masks, (s + C + j * 2) * STEP)
                for k in keys:
                    acc[k]["in"].append(shaped[k][b, j][gm])
                    acc[k]["out"].append(shaped[k][b, j][~gm])
    m = {k: {w: float(np.concatenate([x for x in v[w] if len(x)]).mean()) for w in ("in", "out")}
         for k, v in acc.items()}

    res = {"means": m,
           "relative_raw_movement_out": m["d_raw"]["out"] / m["raw_mag"]["out"],
           "renorm_share_of_post_norm_movement_out": m["d_renorm_only"]["out"] / m["d_norm"]["out"],
           "renorm_share_of_post_norm_movement_in": m["d_renorm_only"]["in"] / m["d_norm"]["in"]}

    print(f"\n{'':>34}{'runner tokens':>16}{'elsewhere':>12}")
    for k, lab in (("raw_mag", "raw token magnitude"), ("d_raw", "raw movement (pre layer-norm)"),
                   ("d_norm", "movement after layer-norm"),
                   ("d_renorm_only", "  of which: statistics alone"),
                   ("dmu", "mean shift (in units of sigma)"), ("dsigma", "relative sigma change")):
        print(f"{lab:>34}{m[k]['in']:>16.4f}{m[k]['out']:>12.4f}")
    print(f"\nfor tokens ELSEWHERE in the frame:")
    print(f"  raw features moved {res['relative_raw_movement_out']:.1%} of their own magnitude")
    print(f"  the layer norm's own statistics account for {res['renorm_share_of_post_norm_movement_out']:.1%} "
          f"of the post-norm movement")
    verdict = ("frame-wide movement is genuine: the raw features moved before any normalization"
               if res["renorm_share_of_post_norm_movement_out"] < 0.25 else
               "renormalization accounts for a large share; the frame-wide claim needs qualifying")
    res["verdict"] = verdict
    print(f"\n{verdict}")
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
