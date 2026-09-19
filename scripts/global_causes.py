"""What global, scene-wide change drives the surprise peaks?

person_vs_background.py showed the peaks are not local: at a peak, background tokens are elevated uniformly
regardless of how far they sit from the nearest pedestrian (z=+0.21 at distance 1 vs +0.22 at distance 4),
while the pedestrians themselves make their own patch EASIER to predict (-0.026 within-cell, 75% of cells).

Whatever moves the curve therefore acts on the whole frame at once. The cheap candidates for that on fixed
CCTV footage are camera-level, not scene-level: auto-exposure drift, camera shake, and codec keyframe jumps.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from vjepa.data.avenue import frame_labels, test_videos
from vjepa.eval.avenue_eval import CACHE, PRIMARY, frame_scores, postprocess
from vjepa.eval.scene_events import read_gray

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
OUT = Path("outputs/investigations/global_causes.json")
LAG = 4


def global_signals(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Frame-level camera signals, each edge-padded to full length."""
    brightness = gray.mean(axis=(1, 2))
    contrast = gray.std(axis=(1, 2))
    d_bright = np.abs(np.diff(brightness, prepend=brightness[0]))
    d_contrast = np.abs(np.diff(contrast, prepend=contrast[0]))

    # camera shake: best whole-frame integer shift between frames LAG apart, scored on a central crop
    shake = np.zeros(len(gray))
    a = gray[:-LAG, 40:-40, 60:-60]
    for t in range(len(a)):
        ref, cur = gray[t], gray[t + LAG]
        best = min(
            np.abs(cur[40 + dy : -40 + dy or None, 60 + dx : -60 + dx or None] - ref[40:-40, 60:-60]).mean()
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        )
        still = np.abs(cur[40:-40, 60:-60] - ref[40:-40, 60:-60]).mean()
        shake[t + LAG] = still - best  # how much a 1px re-alignment improves the match
    return {"brightness": brightness, "contrast": contrast, "d_brightness": d_bright,
            "d_contrast": d_contrast, "shake": shake}


def main() -> None:
    names = ("brightness", "contrast", "d_brightness", "d_contrast", "shake")
    rhos = {n: [] for n in names}
    for path in test_videos():
        lab = frame_labels(path.stem)
        data = dict(np.load(RUN_DIR / f"{path.stem}.npz"))
        if len(data["starts"]) < 8:
            continue
        curve = postprocess(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], len(lab)), PRIMARY["sigma"])
        sig = global_signals(read_gray(path))
        for n in names:
            rhos[n].append(spearmanr(curve, postprocess(sig[n][: len(lab)], PRIMARY["sigma"])).statistic)
        print(f"  {path.stem}: " + "  ".join(f"{n}={rhos[n][-1]:+.2f}" for n in names))

    print(f"\n{'global signal':>14} {'median rho':>11} {'|rho|>0.3 in':>13}")
    res = {}
    for n in names:
        v = np.array(rhos[n])
        res[n] = {"median_rho": float(np.median(v)), "frac_strong": float((np.abs(v) > 0.3).mean()),
                  "min": float(v.min()), "max": float(v.max())}
        print(f"{n:>14} {res[n]['median_rho']:>+11.3f} {res[n]['frac_strong']:>12.0%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
