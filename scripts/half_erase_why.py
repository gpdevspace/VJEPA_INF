"""Why is the half-erasure non-additive? Measure how far each side actually moves when the runner is erased.

half_erase.py found that erasing the runner from BOTH inputs lowers the error by 0.0064, while erasing from
either input alone RAISES it by about 0.0023. That is strongly non-additive, and it means the single-sided
conditions are dominated by a correspondence artifact: a person present in one input and absent from the other.

This measures the two quantities that explain it directly, on the windows that predict the erased span:

  d_target   how far the target representation moves when the runner is erased from the target encoder's input
  d_pred     how far the predictor's OUTPUT moves when the runner is erased from its context

and the full 2x2 of errors, so the curve result can be reproduced from first principles:

             target=original   target=painted
  ctx=orig      baseline          target_only
  ctx=paint     context_only      both

A small d_pred next to a large d_target would say the predictor barely encodes the person at all, which would
fit the floor findings: it closes only 18% of the gap to a perfect prediction and is beaten by a static scene
template on ordinary patches.

(Checked first: this predictor uses learned mask tokens, so the `tgt` argument supplies shape only and no target
values leak into the prediction.)
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

OUT = Path("outputs/investigations/half_erase_why.json")
WIN, C = 16, PRIMARY["context"]


@torch.no_grad()
def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, _ = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    painted = paint(frames, masks, ERASE)
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])

    xo, xp = preprocess(frames[::STEP]), preprocess(painted[::STEP])
    # windows whose predicted frames overlap the erased span
    lo, hi = ERASE[0] // STEP - WIN + 1, ERASE[1] // STEP - C
    starts = [s for s in range(0, xo.shape[1] - WIN + 1, STRIDE) if lo <= s <= hi]
    print(f"{len(starts)} windows predict the erased span (sampled starts {starts[0]}-{starts[-1]})")

    ctx, tgt = (m.to(device) for m in time_masks(C, WIN))
    acc = {k: [] for k in ("d_target", "d_pred", "tgt_mag", "oo", "op", "po", "pp")}
    for i in range(0, len(starts), 4):
        sl = starts[i : i + 4]
        bo = torch.stack([xo[:, s : s + WIN] for s in sl]).to(device)
        bp = torch.stack([xp[:, s : s + WIN] for s in sl]).to(device)
        n = len(sl)
        cm, tm = ctx.repeat(n, 1), tgt.repeat(n, 1)
        with autocast(device, meter.dtype):
            ho = F.layer_norm(h := meter.model.target_encoder(bo), (h.size(-1),))[:, cm.shape[1] :].float()
            hp = F.layer_norm(h := meter.model.target_encoder(bp), (h.size(-1),))[:, cm.shape[1] :].float()
            po = meter.model.predictor(meter.model.encoder(bo, masks=[cm]), ho, [cm], [tm], mask_index=0).float()
            pp = meter.model.predictor(meter.model.encoder(bp, masks=[cm]), hp, [cm], [tm], mask_index=0).float()
        acc["d_target"].append((ho - hp).abs().mean(-1).cpu().numpy())
        acc["d_pred"].append((po - pp).abs().mean(-1).cpu().numpy())
        acc["tgt_mag"].append(ho.abs().mean(-1).cpu().numpy())
        for k, (p, t) in {"oo": (po, ho), "op": (po, hp), "po": (pp, ho), "pp": (pp, hp)}.items():
            acc[k].append((p - t).abs().mean(-1).cpu().numpy())
    a = {k: np.concatenate(v) for k, v in acc.items()}

    def top5(e):  # the aggregate the curve uses
        k = max(1, int(0.05 * e.shape[1]))
        return float(np.sort(e, axis=1)[:, -k:].mean())

    res = {
        "windows": len(starts),
        "d_target_mean": float(a["d_target"].mean()), "d_pred_mean": float(a["d_pred"].mean()),
        "target_magnitude": float(a["tgt_mag"].mean()),
        "ratio_dpred_over_dtarget": float(a["d_pred"].mean() / a["d_target"].mean()),
        "errors_top5": {"baseline_ctxO_tgtO": top5(a["oo"]), "target_only_ctxO_tgtP": top5(a["op"]),
                        "context_only_ctxP_tgtO": top5(a["po"]), "both_ctxP_tgtP": top5(a["pp"])},
        "errors_mean": {k: float(a[k].mean()) for k in ("oo", "op", "po", "pp")},
    }
    e = res["errors_top5"]
    print(f"\nhow far each side moves when the runner is erased (per-token L1, target magnitude "
          f"{res['target_magnitude']:.3f}):")
    print(f"  target representation moves: {res['d_target_mean']:.4f}")
    print(f"  predictor output moves:      {res['d_pred_mean']:.4f}   "
          f"({res['ratio_dpred_over_dtarget']:.0%} as much)")
    print(f"\ntop-5% error, 2x2:")
    print(f"  {'':>16}{'target=orig':>14}{'target=painted':>16}")
    print(f"  {'ctx=orig':>16}{e['baseline_ctxO_tgtO']:>14.4f}{e['target_only_ctxO_tgtP']:>16.4f}")
    print(f"  {'ctx=painted':>16}{e['context_only_ctxP_tgtO']:>14.4f}{e['both_ctxP_tgtP']:>16.4f}")

    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")




@torch.no_grad()
def by_distance() -> None:
    """Is the target's movement local to the runner, or frame-wide?

    d_target above pooled every token. person_vs_background.py found background tokens rise uniformly with
    distance from the nearest pedestrian, so the same split is applied here: how far does the representation
    move for tokens the runner actually covers, versus tokens elsewhere in the frame?
    """
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, _ = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    painted = paint(frames, masks, ERASE)
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])
    xo, xp = preprocess(frames[::STEP]), preprocess(painted[::STEP])
    lo, hi = ERASE[0] // STEP - WIN + 1, ERASE[1] // STEP - C
    starts = [s for s in range(0, xo.shape[1] - WIN + 1, STRIDE) if lo <= s <= hi]
    ctx, tgt = (m.to(device) for m in time_masks(C, WIN))

    def grid_mask(orig_frame: int) -> np.ndarray:
        """The runner's 360x640 mask reduced to the 14x14 token grid (crop='square' maps the whole frame)."""
        if not (0 <= orig_frame < masks.shape[1]):
            return np.zeros((14, 14), bool)
        m = masks[0, orig_frame].astype(np.float32)
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
        d = (ho - hp).abs().mean(-1).cpu().numpy().reshape(n, -1, 14, 14)  # (B, slices, 14, 14)
        for b, s in enumerate(sl):
            for j in range(d.shape[1]):
                gm = grid_mask((s + C + j * 2) * STEP)
                ins.append(d[b, j][gm]); outs.append(d[b, j][~gm])
    inside = np.concatenate([x for x in ins if len(x)])
    outside = np.concatenate([x for x in outs if len(x)])
    print(f"\ntarget representation movement when the runner is erased:")
    print(f"  tokens the runner covers ({len(inside)}): {inside.mean():.4f}")
    print(f"  tokens elsewhere ({len(outside)}):        {outside.mean():.4f}   "
          f"({outside.mean() / inside.mean():.0%} as much)")
    res = json.loads(OUT.read_text())
    res["d_target_inside_runner"] = float(inside.mean())
    res["d_target_outside_runner"] = float(outside.mean())
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    by_distance()
