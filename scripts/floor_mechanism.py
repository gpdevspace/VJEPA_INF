"""Why is the prediction error ~0.55 and not ~0.05? What would a no-skill predictor score?

floor_decomposition.py established that the floor is not an out-of-distribution artifact: grey scores LOWER
than real content, and a static real frame already scores 98.6% of a moving one. It called the remainder
"ordinary prediction error" and quoted a predict-zeros baseline of 0.676 -- but that number was never produced
by a script in this repo, so it is re-measured here along with the baselines that actually matter.

The mechanism this tests: the target is layer-normalized per token (`F.layer_norm(h, (h.size(-1),))` in
surprise.py). Every target token is forced to zero mean and unit variance across its 1024 dims, whatever the
content. So the units of "surprise" are fixed by the normalizer, not by the video, and the interesting question
is where the predictor sits between trivial baselines measured in those same units:

  zeros            predict 0 for every dim. The error is just E|target|, set by the layer norm.
  global mean      predict the mean target token over the whole sample (one vector for everything).
  position mean    predict the mean token for that grid position (a static scene prior: "this patch is sky").
  persistence      copy the co-located token from the LAST CONTEXT frame ("nothing will change").
  predictor        what V-JEPA actually does.
  shuffled         a real target token from a different window (a valid but wrong answer: the upper end).

If the predictor barely beats persistence, the Surprise Meter is close to a change detector, which bears
directly on the post-2 question of whether surprise tracks anomaly or merely unpredictable motion.
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

OUT = Path("outputs/investigations/floor_mechanism.json")
C, T, STEP, PER_VIDEO, N_VIDEOS = PRIMARY["context"], 16, 4, 6, 8
GRID = 14 * 14


def aggregates(err: np.ndarray) -> dict:
    """err is (B, N_tgt) per-token L1. Report the two aggregates the eval uses."""
    k = max(1, int(0.05 * err.shape[1]))
    return {"mean": float(err.mean()), "top5": float(np.sort(err, axis=1)[:, -k:].mean())}


@torch.no_grad()
def collect(meter, clips):
    """Per window: layer-normed targets, predictor output, last context slice, and pre-LN feature scale."""
    tgts, preds, lasts, raw_scale = [], [], [], []
    for i in range(0, len(clips), 4):
        x = torch.stack([preprocess(f) for f in clips[i : i + 4]]).to(meter.device)
        with autocast(meter.device, meter.dtype):
            h_raw = meter.model.target_encoder(x)
            h = F.layer_norm(h_raw, (h_raw.size(-1),))
            ctx, tgt = (m.to(meter.device).repeat(len(x), 1) for m in time_masks(C, T))
            z = meter.model.encoder(x, masks=[ctx])
            target = h[:, ctx.shape[1] :]
            pred = meter.model.predictor(z, target, [ctx], [tgt], mask_index=0)
        n_ctx = ctx.shape[1]
        tgts.append(h[:, n_ctx:].float().cpu())
        preds.append(pred.float().cpu())
        lasts.append(h[:, n_ctx - GRID : n_ctx].float().cpu())  # last context tubelet slice
        raw_scale.append(h_raw.float().std(-1).mean(-1).cpu())  # per-token std before the layer norm
    return torch.cat(tgts), torch.cat(preds), torch.cat(lasts), torch.cat(raw_scale)


def main() -> None:
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device)
    rng = np.random.default_rng(0)

    clips = []
    for path in test_videos()[:N_VIDEOS]:
        frames, _ = read_video(path, STEP)
        if len(frames) < T:
            continue
        for s in np.linspace(0, len(frames) - T, PER_VIDEO, dtype=int):
            clips.append(frames[s : s + T])
    print(f"windows: {len(clips)}")

    tgt, pred, last, raw_real = collect(meter, clips)
    B, N, D = tgt.shape
    n_slices = N // GRID
    print(f"targets {tuple(tgt.shape)}  ({n_slices} tubelet slices x {GRID} positions)")

    l1 = lambda a, b: (a - b).abs().mean(-1).numpy()

    # --- baselines, all in the same layer-normed units ---
    res = {}
    res["zeros"] = aggregates(l1(tgt, torch.zeros_like(tgt)))
    res["global_mean"] = aggregates(l1(tgt, tgt.reshape(-1, D).mean(0).expand_as(tgt)))

    pos_mean = tgt.reshape(B, n_slices, GRID, D).mean(0, keepdim=True).expand(B, -1, -1, -1).reshape(B, N, D)
    res["position_mean"] = aggregates(l1(tgt, pos_mean))

    persist = last.repeat(1, n_slices, 1)  # copy the last observed frame's tokens forward
    res["persistence"] = aggregates(l1(tgt, persist))

    res["predictor"] = aggregates(l1(tgt, pred))

    perm = rng.permutation(B)
    while (perm == np.arange(B)).any():
        perm = rng.permutation(B)
    res["shuffled_target"] = aggregates(l1(tgt, tgt[perm]))

    # --- how the layer norm sets the scale ---
    shape = clips[0].shape
    grey = [np.full(shape, 128, np.uint8) for _ in range(4)]
    noise = [np.repeat(rng.integers(0, 256, (1, *shape[1:]), dtype=np.uint8), T, 0) for _ in range(4)]
    _, _, _, raw_grey = collect(meter, grey)
    _, _, _, raw_noise = collect(meter, noise)
    scale = {"real": float(raw_real.mean()), "grey": float(raw_grey.mean()), "frozen_noise": float(raw_noise.mean())}

    # how far the per-dim target distribution is from Gaussian (a unit-variance Gaussian would give E|x|=0.798)
    flat = tgt.reshape(-1, D)
    stats = {"E_abs": float(flat.abs().mean()), "gaussian_E_abs": 0.7979,
             "kurtosis": float(((flat - flat.mean()) ** 4).mean() / flat.var() ** 2)}

    print(f"\n{'baseline':>18}{'mean':>9}{'top5':>9}{'% of gap closed':>18}")
    z = res["zeros"]["mean"]
    for name in ("zeros", "global_mean", "position_mean", "persistence", "predictor", "shuffled_target"):
        m = res[name]["mean"]
        print(f"{name:>18}{m:>9.4f}{res[name]['top5']:>9.4f}{(z - m) / z:>17.0%}")

    print(f"\npre-layer-norm feature std per token: " + "  ".join(f"{k}={v:.3f}" for k, v in scale.items()))
    print(f"post-layer-norm E|target| = {stats['E_abs']:.4f} (unit-variance Gaussian would be 0.798, "
          f"kurtosis {stats['kurtosis']:.1f})")

    pv, pe = res["persistence"]["mean"], res["predictor"]["mean"]
    print(f"\npredictor vs persistence: {pe:.4f} vs {pv:.4f} -> predictor is {(pv - pe) / pv:.1%} better than "
          f"copying the last seen frame")
    print(f"predictor vs position prior: {pe:.4f} vs {res['position_mean']['mean']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"baselines": res, "windows": B, "preLN_token_std": scale,
                               "target_dist": stats}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
