"""Is V-JEPA's predictor beaten by a static scene template, and why is copying the last frame so bad?

floor_mechanism.py found, on 42 Avenue windows, that a per-(time slice, grid position) mean token scores 0.509
against the predictor's 0.554 -- but that template was fit and evaluated on the same windows. This re-tests it
with the template fit on a disjoint set of videos, and decomposes the target token to explain the other oddity:
copying the last observed frame (0.774) is WORSE than predicting zeros (0.676) and worse than a real token from
an unrelated window (0.719).

The decomposition splits each layer-normed target token into
    global mean + time-slice effect + grid-position effect + residual
and measures how much L1 each part accounts for. If the slice effect is large, the representation carries a
strong "which frame index am I" signal, the predictor gets that part for free, and persistence fails because it
supplies the wrong slice.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vjepa.data.avenue import test_videos
from vjepa.device import autocast, pick_device
from vjepa.eval.avenue_eval import PRIMARY
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter, time_masks
from vjepa.video import preprocess, read_video

OUT = Path("outputs/investigations/floor_template.json")
CACHE = Path("/private/tmp/claude-501/-Users-gpmac-gpbuildspace-VJEPA/c2fbbf21-ae30-423e-b300-e6c206128439/scratchpad/floor_tensors.pt")
C, T, STEP, PER_VIDEO, N_VIDEOS = PRIMARY["context"], 16, 4, 6, 8
GRID = 14 * 14


def agg(err: torch.Tensor) -> dict:
    e = err.numpy()
    k = max(1, int(0.05 * e.shape[1]))
    return {"mean": float(e.mean()), "top5": float(np.sort(e, axis=1)[:, -k:].mean())}


@torch.no_grad()
def collect():
    if CACHE.exists():
        d = torch.load(CACHE)
        return d["tgt"], d["pred"], d["last"], d["vid"]
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device)
    clips, vids = [], []
    for vi, path in enumerate(test_videos()[:N_VIDEOS]):
        frames, _ = read_video(path, STEP)
        if len(frames) < T:
            continue
        for s in np.linspace(0, len(frames) - T, PER_VIDEO, dtype=int):
            clips.append(frames[s : s + T])
            vids.append(vi)
    tgts, preds, lasts = [], [], []
    for i in range(0, len(clips), 4):
        x = torch.stack([preprocess(f) for f in clips[i : i + 4]]).to(device)
        with autocast(device, meter.dtype):
            h = F.layer_norm(hr := meter.model.target_encoder(x), (hr.size(-1),))
            ctx, tgt = (m.to(device).repeat(len(x), 1) for m in time_masks(C, T))
            z = meter.model.encoder(x, masks=[ctx])
            target = h[:, ctx.shape[1] :]
            pred = meter.model.predictor(z, target, [ctx], [tgt], mask_index=0)
        n = ctx.shape[1]
        tgts.append(h[:, n:].float().cpu()); preds.append(pred.float().cpu())
        lasts.append(h[:, n - GRID : n].float().cpu())
    out = (torch.cat(tgts), torch.cat(preds), torch.cat(lasts), torch.tensor(vids))
    torch.save({"tgt": out[0], "pred": out[1], "last": out[2], "vid": out[3]}, CACHE)
    return out


def main() -> None:
    tgt, pred, last, vid = collect()
    B, N, D = tgt.shape
    S = N // GRID
    t4 = tgt.reshape(B, S, GRID, D)
    print(f"windows {B}  slices {S}  positions {GRID}  dim {D}")

    res = {}
    res["predictor"] = agg((tgt - pred).abs().mean(-1))
    res["zeros"] = agg(tgt.abs().mean(-1))

    # --- held-out scene template: fit the per-(slice, position) mean on half the videos, score the other half ---
    uniq = vid.unique()
    fit_v, ev_v = uniq[: len(uniq) // 2], uniq[len(uniq) // 2 :]
    fit_m, ev_m = torch.isin(vid, fit_v), torch.isin(vid, ev_v)
    tmpl = t4[fit_m].mean(0)                                    # (S, GRID, D)
    ev = t4[ev_m]
    res["template_heldout"] = agg((ev - tmpl).abs().mean(-1).reshape(ev_m.sum(), N))
    res["predictor_on_heldout"] = agg((tgt[ev_m] - pred[ev_m]).abs().mean(-1))
    res["template_insample"] = agg((t4 - t4.mean(0)).abs().mean(-1).reshape(B, N))
    print(f"held-out split: fit on videos {fit_v.tolist()} ({int(fit_m.sum())} windows), "
          f"score on {ev_v.tolist()} ({int(ev_m.sum())} windows)")

    # --- why persistence fails: decompose the token into global + slice + position + residual ---
    g = tgt.reshape(-1, D).mean(0)
    slice_eff = t4.mean((0, 2)) - g                              # (S, D)
    pos_eff = t4.mean((0, 1)) - g                                # (GRID, D)
    resid = t4 - (g + slice_eff[None, :, None, :] + pos_eff[None, None, :, :])
    parts = {"global": g.abs().mean(), "slice_effect": slice_eff.abs().mean(),
             "position_effect": pos_eff.abs().mean(), "residual": resid.abs().mean()}
    res["token_parts_L1"] = {k: float(v) for k, v in parts.items()}

    # distance between the mean token of adjacent slices, vs between two random windows at the same slice
    sl_d = float((t4.mean((0, 2))[:-1] - t4.mean((0, 2))[1:]).abs().mean())
    res["adjacent_slice_mean_L1"] = sl_d

    res["persistence"] = agg((tgt - last.repeat(1, S, 1)).abs().mean(-1))
    # persistence corrected: copy the last frame but swap in the target slice's own mean offset
    corr = last.repeat(1, S, 1).reshape(B, S, GRID, D) - slice_eff[3][None, None, None, :] + slice_eff[None, :, None, :]
    res["persistence_slice_corrected"] = agg((t4 - corr).abs().mean(-1).reshape(B, N))

    print(f"\n{'model':>28}{'mean':>9}{'top5':>9}")
    for k in ("zeros", "persistence", "persistence_slice_corrected", "template_insample",
              "template_heldout", "predictor_on_heldout", "predictor"):
        print(f"{k:>28}{res[k]['mean']:>9.4f}{res[k]['top5']:>9.4f}")

    print(f"\ntoken decomposition (mean |.| per dim): " + "  ".join(f"{k}={v:.3f}" for k, v in res['token_parts_L1'].items()))
    print(f"mean L1 between adjacent slice means: {sl_d:.4f}")
    th, ph = res["template_heldout"]["mean"], res["predictor_on_heldout"]["mean"]
    print(f"\nHELD-OUT: template {th:.4f} vs predictor {ph:.4f} -> "
          f"{'template WINS by' if th < ph else 'predictor wins by'} {abs(th-ph):.4f} "
          f"({abs(th-ph)/max(th,ph):.1%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
