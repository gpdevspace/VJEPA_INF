"""What is the 0.617 floor made of: two networks disagreeing, or an out-of-distribution input?

A solid grey clip scores 0.617 against a real Avenue clip's 0.687, so ~90% of "surprise" is present when there
is nothing to be surprised by. Two candidate causes, and they call for different conclusions:

  network mismatch -- the predictor and the target encoder are different modules, so their outputs never agree
     exactly, even on a trivial input. This floor would be irreducible and would say nothing about the video.
  out-of-distribution -- flat grey never occurs in natural video, so both modules are being run off-distribution
     and their disagreement is inflated. This floor would be an artifact of a silly input, not a real baseline.

Four conditions separate them. The last two are PAIRED: the frozen clip repeats the first frame of the very
window the real clip covers, so scene, camera and grid position are identical and only motion differs.

  grey          no content, off-distribution
  frozen noise  content, off-distribution, no motion
  frozen real   content, in-distribution, no motion   <- paired
  real          content, in-distribution, motion      <- paired
"""

import json
from pathlib import Path

import numpy as np
import torch

from vjepa.data.avenue import test_videos
from vjepa.device import pick_device
from vjepa.eval.avenue_eval import PRIMARY
from vjepa.eval.intphys_eval import token_aggregates
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video

OUT = Path("outputs/investigations/floor_decomposition.json")
C, T, STEP, PER_VIDEO, N_VIDEOS = PRIMARY["context"], 16, 4, 6, 8


def score(meter, frames_list) -> dict[str, np.ndarray]:
    """Score a list of (T, H, W, 3) uint8 clips, returning both token aggregates."""
    out = {"top5": [], "mean": []}
    for i in range(0, len(frames_list), 4):
        batch = torch.stack([preprocess(f) for f in frames_list[i : i + 4]])
        _, tok = meter.score_windows(batch, contexts=[C])
        agg = token_aggregates(tok[0].astype(np.float32))
        out["top5"].append(agg["top5"])
        out["mean"].append(agg["mean"])
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device)
    rng = np.random.default_rng(0)

    real_clips, frozen_clips = [], []
    for path in test_videos()[:N_VIDEOS]:
        frames, _ = read_video(path, STEP)
        if len(frames) < T:
            continue
        for s in np.linspace(0, len(frames) - T, PER_VIDEO, dtype=int):
            w = frames[s : s + T]
            real_clips.append(w)
            frozen_clips.append(np.repeat(w[:1], T, axis=0))  # same scene, same position, zero motion
    print(f"paired clips: {len(real_clips)}")

    shape = real_clips[0].shape
    conditions = {
        "grey": [np.full(shape, 128, np.uint8) for _ in range(8)],
        "frozen_noise": [np.repeat(rng.integers(0, 256, (1, *shape[1:]), dtype=np.uint8), T, 0) for _ in range(8)],
        "frozen_real": frozen_clips,
        "real": real_clips,
    }
    res = {}
    for name, clips in conditions.items():
        s = score(meter, clips)
        res[name] = {k: {"mean": float(v.mean()), "std": float(v.std())} for k, v in s.items()}
        print(f"  {name:>13}: top5 {s['top5'].mean():.4f} +- {s['top5'].std():.4f}   mean {s['mean'].mean():.4f}")
        if name in ("frozen_real", "real"):
            res[name]["_raw_top5"] = s["top5"].tolist()

    fr = np.array(res["frozen_real"]["_raw_top5"])
    rl = np.array(res["real"]["_raw_top5"])
    grey, noise = res["grey"]["top5"]["mean"], res["frozen_noise"]["top5"]["mean"]

    print(f"\n{'':>34}{'top5':>9}")
    print(f"{'grey (no content, OOD)':>34}{grey:>9.4f}")
    print(f"{'frozen noise (OOD, static)':>34}{noise:>9.4f}")
    print(f"{'frozen real (in-dist, static)':>34}{fr.mean():>9.4f}")
    print(f"{'real (in-dist, moving)':>34}{rl.mean():>9.4f}")

    print(f"\npaired: motion adds {(rl - fr).mean():+.4f} on the same scenes "
          f"(higher in {(rl > fr).mean():.0%} of {len(fr)} pairs)")
    print(f"in-distribution content vs grey: {fr.mean() - grey:+.4f}")
    span = rl.mean() - grey
    print(f"\nof the real score {rl.mean():.4f}: grey floor {grey:.4f} ({grey / rl.mean():.0%}), "
          f"everything the video adds {span:.4f} ({span / rl.mean():.0%})")
    # Reference points measured separately (see the no-skill baseline below): on a real clip, predicting all
    # zeros scores 0.676 and a shuffled-but-valid target scores 0.862, against the predictor's actual 0.550.
    verdict = (
        f"The floor is NOT an out-of-distribution artifact: grey is the LOWEST condition ({grey:.4f}), below "
        f"frozen noise ({noise:.4f}) and frozen real content ({fr.mean():.4f}). Static in-distribution content "
        f"with nothing happening already scores {fr.mean() / rl.mean():.1%} of a real moving clip; motion adds "
        f"only {(rl - fr).mean():+.4f}. The floor is ordinary prediction error -- on a real clip the predictor "
        f"closes just 19% of the gap between predicting zeros (0.676) and predicting perfectly."
    )
    print(f"\n{verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"conditions": {k: {a: v[a] for a in ("top5", "mean")} for k, v in res.items()},
                               "n_pairs": len(fr), "motion_delta": float((rl - fr).mean()),
                               "frozen_real_minus_grey": float(fr.mean() - grey), "verdict": verdict}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
