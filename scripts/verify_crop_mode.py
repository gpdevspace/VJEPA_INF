"""Does Avenue's crop mode change the conclusions?

The intuitive-physics reference builds its transform with random_resize_aspect_ratio=[1/1, 1/1], i.e. a square
centre crop. Our Avenue eval instead uses crop="square", which squashes the whole 640x360 frame into 224x224.
On IntPhys the two are identical because its frames are already 288x288, so the choice never mattered there --
but Avenue is 16:9, and squashing compresses the horizontal axis 2.9x against the vertical's 1.6x.

That matters for two claims: the centre-to-border error gradient, and the size of the floor. This rescores a
few videos both ways and compares the level, the spatial pattern and the AUC.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.device import pick_device
from vjepa.eval.avenue_eval import PRIMARY, postprocess
from vjepa.eval.intphys_eval import token_aggregates
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video

OUT = Path("outputs/investigations/crop_mode.json")
STEMS = ["01", "04", "06", "16", "05"]
C, T, STEP, STRIDE, GRID = PRIMARY["context"], 16, 4, 4, 14

yy, xx = np.mgrid[0:GRID, 0:GRID]
RING = np.minimum(np.minimum(yy, GRID - 1 - yy), np.minimum(xx, GRID - 1 - xx))


def spread(per_window, starts, n_sampled, total):
    acc, cnt = np.zeros(n_sampled), np.zeros(n_sampled)
    for s, v in zip(starts, per_window):
        acc[s + C : s + T] += v
        cnt[s + C : s + T] += 1
    cov = np.flatnonzero(cnt)
    f = np.repeat(np.interp(np.arange(n_sampled), cov, acc[cov] / cnt[cov]), STEP)[:total]
    return np.pad(f, (0, total - len(f)), mode="edge")


def main() -> None:
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device)
    res = {}

    for crop in ("square", "center"):
        aucs, levels, maps = [], [], []
        for stem in STEMS:
            frames, _ = read_video(ROOT / "Avenue Dataset" / "testing_videos" / f"{stem}.avi", STEP)
            labels = frame_labels(stem)
            clip = preprocess(frames, crop=crop, short_side=224 if crop == "center" else None)
            starts = np.arange(0, clip.shape[1] - T + 1, STRIDE)
            tok = []
            for i in range(0, len(starts), 4):
                batch = torch.stack([clip[:, s : s + T] for s in starts[i : i + 4]])
                _, m = meter.score_windows(batch, contexts=[C])
                tok.append(m[0])
            tok = np.concatenate(tok).astype(np.float32)
            per_window = token_aggregates(tok)["top5"]
            curve = postprocess(spread(per_window, starts, len(frames), len(labels)), PRIMARY["sigma"])
            if 0 < labels.sum() < len(labels):
                aucs.append(roc_auc_score(labels, curve))
            levels.append(tok.mean())
            rel = tok / (tok.mean(axis=(2, 3), keepdims=True) + 1e-8)
            maps.append(rel.mean(axis=(0, 1)))
            print(f"  {crop:>6} {stem}: mean L1 {tok.mean():.4f}  AUC {aucs[-1] if aucs else float('nan'):.3f}")

        m = np.mean(maps, axis=0)
        rings = {f"ring{i}": float(m[RING == i].mean()) for i in range(GRID // 2)}
        res[crop] = {"mean_auc": float(np.mean(aucs)), "aucs": [float(a) for a in aucs],
                     "mean_L1": float(np.mean(levels)), "rings": rings,
                     "outer_inner_ratio": rings["ring0"] / rings["ring6"]}

    print(f"\n{'crop':>8} {'mean L1':>9} {'mean AUC':>9} {'ring0(outer)':>13} {'ring6(inner)':>13} {'ratio':>7}")
    for crop, r in res.items():
        print(f"{crop:>8} {r['mean_L1']:>9.4f} {r['mean_auc']:>9.3f} {r['rings']['ring0']:>13.3f} "
              f"{r['rings']['ring6']:>13.3f} {r['outer_inner_ratio']:>7.2f}")
    print(f"\nper-video AUC  square: {np.round(res['square']['aucs'], 3)}")
    print(f"               center: {np.round(res['center']['aucs'], 3)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
