"""Is the frame-border error gradient an artifact of the encoder's geometry, or a fact about Avenue?

Earlier finding: on Avenue, raw token error is systematically higher near the frame border (r=+0.71 with
distance from centre) and local texture does not explain it (partial r=-0.14). That pointed at a Vision
Transformer edge effect -- but "points at" is not a test, and every Avenue video shares one camera, so the
scene layout and the frame geometry are perfectly confounded there.

This removes the confound by scoring inputs with no scene at all. A uniform grey clip and a frozen-noise clip
contain nothing that could be intrinsically harder at the edges; a frozen natural frame contains content but no
motion. If the same centre-to-border gradient appears in all of them, it belongs to the model, not the video.
"""

import json
from pathlib import Path

import numpy as np
import torch

from vjepa.data.avenue import test_videos
from vjepa.device import pick_device
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video

GRID, CONTEXT, T = 14, 8, 16
OUT = Path("outputs/investigations/border_artifact.json")


def rings(m: np.ndarray) -> dict[str, float]:
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    r = np.minimum(np.minimum(yy, GRID - 1 - yy), np.minimum(xx, GRID - 1 - xx))
    return {f"ring{i}": round(float(m[r == i].mean()), 3) for i in range(GRID // 2)}


def main() -> None:
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device)
    rng = np.random.default_rng(0)

    avenue = read_video(test_videos()[3], 4)[0]  # video 04, for a real frame and a real moving clip
    clips = {
        "uniform_grey": np.full((T, 360, 640, 3), 128, np.uint8),
        "frozen_noise": np.repeat(rng.integers(0, 256, (1, 360, 640, 3), dtype=np.uint8), T, 0),
        "frozen_avenue_frame": np.repeat(avenue[80:81], T, 0),
        "moving_noise": rng.integers(0, 256, (T, 360, 640, 3), dtype=np.uint8),
        "real_avenue_clip": avenue[80 : 80 + T],
    }

    maps = {}
    for name, frames in clips.items():
        x = preprocess(frames).unsqueeze(0)  # (1, 3, T, 224, 224)
        _, tok = meter.score_windows(x, contexts=[CONTEXT])
        m = tok[0][0].mean(0)  # (14, 14), averaged over target slices
        maps[name] = m / m.mean()  # spatial pattern only
        print(f"\n=== {name} (normalised by its own mean) ===")
        print(np.round(maps[name], 2))
        print("  rings (outer -> inner):", rings(maps[name]))

    yy, xx = np.mgrid[0:GRID, 0:GRID]
    dist = np.sqrt((yy - 6.5) ** 2 + (xx - 6.5) ** 2).ravel()
    print(f"\n{'clip':>22} {'corr(dist, error)':>18} {'outer/inner ratio':>18}")
    summary = {}
    for name, m in maps.items():
        r = float(np.corrcoef(dist, m.ravel())[0, 1])
        ratio = float(rings(m)["ring0"] / rings(m)["ring6"])
        summary[name] = {"corr_distance": r, "outer_inner_ratio": ratio}
        print(f"{name:>22} {r:>18.3f} {ratio:>18.2f}")

    contentless = [summary[k]["corr_distance"] for k in ("uniform_grey", "frozen_noise")]
    verdict = (
        "ARCHITECTURAL: the gradient is present on content-free input, so it belongs to the encoder, not Avenue"
        if min(contentless) > 0.3
        else "NOT ARCHITECTURAL: content-free clips show no border gradient, so Avenue's gradient comes from the scene"
    )
    print(f"\n{verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"maps": {k: v.tolist() for k, v in maps.items()},
                               "rings": {k: rings(v) for k, v in maps.items()},
                               "summary": summary, "verdict": verdict}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
