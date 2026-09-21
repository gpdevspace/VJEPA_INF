"""Which half of the model does the runner act on: the past the predictor reads, or the future it is scored against?

paint_out_runner.py showed the runner causes ~60% of the surprise bump, and control_paint_out.py ruled out the
inpainting as the cause. Open question: the runner's own patches get EASIER to predict while distant background
patches get harder, and we do not know why.

The error has exactly two inputs, and SurpriseMeter.score_windows already lets them be set independently via
`context_x`:

  context encoder   sees only the first 8 frames of the window. Feeds the predictor.
  target encoder    sees all 16 frames. Produces the answer the prediction is scored against.

So the runner can be erased from one input while left in the other:

  both            runner erased everywhere                     (reproduces the earlier test)
  context_only    erased from the predictor's view of the past, left in the target
  target_only     erased from the target, left in the predictor's past

If context_only carries the effect, the runner changes what can be inferred from the past. If target_only
carries it, the runner changes the representation being matched. If neither alone does, the effect needs both,
which would point at a mismatch between the two rather than at either one.

Caveat: the target encoder reads the whole 16-frame window, so "target_only" swaps its entire input, not just
the future half. Feeding it a half-painted clip would make the runner vanish mid-window, which is its own
surprising event, so that variant is deliberately not run.
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import gaussian_filter1d

from vjepa.data.avenue import ROOT, frame_labels
from vjepa.device import DTYPES, pick_device
from vjepa.eval.avenue_eval import PRIMARY, frame_scores
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video
from scripts.paint_out_runner import ERASE, SIGMA, STEM, STEP, STRIDE, paint

OUT = Path("outputs/investigations/half_erase.json")
NPZ = Path("outputs/investigations/half_erase.npz")
WIN, BATCH = 16, 4


def curve_pair(meter, target_frames: np.ndarray, ctx_frames: np.ndarray, total: int) -> np.ndarray:
    """Score with the target encoder reading `target_frames` and the context encoder reading `ctx_frames`."""
    xt, xc = preprocess(target_frames[::STEP]), preprocess(ctx_frames[::STEP])
    starts = list(range(0, xt.shape[1] - WIN + 1, STRIDE))
    maps = []
    for i in range(0, len(starts), BATCH):
        sl = starts[i : i + BATCH]
        bt = torch.stack([xt[:, s : s + WIN] for s in sl])
        bc = torch.stack([xc[:, s : s + WIN] for s in sl])
        _, m = meter.score_windows(bt, [PRIMARY["context"]], context_x=bc)
        maps.append(m[0])
    data = {"starts": np.array(starts), "n_sampled": xt.shape[1], "frame_step": STEP,
            f"tokens_c{PRIMARY['context']}": np.concatenate(maps)}
    return gaussian_filter1d(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], total), SIGMA)


def main() -> None:
    path = ROOT / "Avenue Dataset" / "testing_videos" / f"{STEM}.avi"
    frames, _ = read_video(path)
    masks = sio.loadmat(ROOT / "ground_truth_demo" / "testing_label_mask" / f"{int(STEM)}_label.mat")["volLabel"]
    labels = frame_labels(STEM)
    painted = paint(frames, masks, ERASE)

    device = pick_device()
    meter = SurpriseMeter(load_vjepa("vitl16", device, with_predictor=True), device, DTYPES["fp16"])
    conds = {
        "baseline": (frames, frames),
        "both": (painted, painted),
        "context_only": (frames, painted),   # predictor's past is runner-free, target still has them
        "target_only": (painted, frames),    # target is runner-free, predictor's past still has them
    }
    curves = {}
    for name, (xt, xc) in conds.items():
        curves[name] = curve_pair(meter, xt, xc, len(labels))
        print(f"scored {name}", flush=True)

    a, b = ERASE
    base = np.r_[curves["baseline"][:300], curves["baseline"][750:]].mean()
    prev = json.loads(Path("outputs/investigations/paint_out_runner.json").read_text())
    m = {k: float(v[a:b].mean()) for k, v in curves.items()}
    res = {"erased_span": ERASE, "baseline_outside": float(base), "means": m,
           "deltas_vs_baseline_cond": {k: m[k] - m["baseline"] for k in conds},
           "reproduces_cached_paint_out": abs(m["both"] - prev["erased"]["painted_mean"]) < 5e-4}

    d_both, d_ctx, d_tgt = (res["deltas_vs_baseline_cond"][k] for k in ("both", "context_only", "target_only"))
    res["sum_of_halves"] = d_ctx + d_tgt
    res["additivity_gap"] = d_both - (d_ctx + d_tgt)

    print(f"\nmean surprise over the erased span {a}-{b} (baseline outside = {base:.4f})")
    for k in conds:
        print(f"  {k:>13}: {m[k]:.4f}   delta {res['deltas_vs_baseline_cond'][k]:+.4f}")
    print(f"\ncontext_only {d_ctx:+.4f} + target_only {d_tgt:+.4f} = {d_ctx + d_tgt:+.4f}   vs both {d_both:+.4f}")
    print(f"additivity gap: {res['additivity_gap']:+.4f}")
    print(f"reproduces the cached paint-out run: {res['reproduces_cached_paint_out']}")

    np.savez(NPZ, **curves)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
