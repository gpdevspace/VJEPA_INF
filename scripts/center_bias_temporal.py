"""Test A2: which grid positions drive the score's ups and downs (as opposed to its baseline level)?

Test A measured the average spatial LEVEL of token error and found the frame periphery hottest -- Avenue's
pavement and ceiling are high-texture, and raw error is drawn to texture. But the claim under test is about
SPIKES, which is a statement about variation over time, not level. A peripheral region can hold the baseline
high while a central region does all the moving.

So: per grid position, correlate that position's error over time against the frame score over time, and measure
how much each position varies. Then check whether the top-5% tokens migrate centrally in the windows that spike.
"""

import json
from pathlib import Path

import numpy as np

from vjepa.eval.avenue_eval import CACHE, PRIMARY
from vjepa.data.avenue import test_videos

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
OUT = Path("outputs/investigations/center_bias_temporal.json")
GRID, TOP_FRAC = 14, 0.05


def rings(m: np.ndarray) -> dict[str, float]:
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    r = np.minimum(np.minimum(yy, GRID - 1 - yy), np.minimum(xx, GRID - 1 - xx))
    return {f"ring{i}": float(m[r == i].mean()) for i in range(GRID // 2)}


def main() -> None:
    context = PRIMARY["context"]
    corr_sum, std_sum, n = np.zeros((GRID, GRID)), np.zeros((GRID, GRID)), 0
    hi_hits, lo_hits = np.zeros((GRID, GRID)), np.zeros((GRID, GRID))

    for path in test_videos():
        data = dict(np.load(RUN_DIR / f"{path.stem}.npz"))
        tok = data[f"tokens_c{context}"].astype(np.float32)  # (W, S, 14, 14)
        if len(tok) < 8:
            continue  # 08 and 21 have two windows: no temporal variation to correlate against
        flat = tok.reshape(len(tok), -1)
        k = max(1, int(TOP_FRAC * flat.shape[1]))
        score = np.sort(flat, axis=1)[:, -k:].mean(1)  # the frame score, exactly as the eval computes it

        pos = tok.mean(axis=1)  # (W, 14, 14): each position's error over time
        pz = (pos - pos.mean(0)) / (pos.std(0) + 1e-8)
        sz = (score - score.mean()) / (score.std() + 1e-8)
        corr_sum += (pz * sz[:, None, None]).mean(0)
        std_sum += pos.std(0) / (pos.mean(0) + 1e-8)  # coefficient of variation: how much each position moves
        n += 1

        # do the tokens driving the score move inward when the score is high?
        idx = np.argsort(flat, axis=1)[:, -k:]
        hi, lo = score >= np.percentile(score, 90), score <= np.percentile(score, 50)
        for sel, acc in ((hi, hi_hits), (lo, lo_hits)):
            p = np.unravel_index(idx[sel].ravel(), tok.shape[1:])
            np.add.at(acc, (p[1], p[2]), 1)

    corr, cv = corr_sum / n, std_sum / n
    hi_f, lo_f = hi_hits / hi_hits.sum() * GRID**2, lo_hits / lo_hits.sum() * GRID**2

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for name, m in (
        ("correlation of each position's error with the frame score", corr),
        ("coefficient of variation over time", cv),
        ("top5 position frequency, HIGH-score windows (top 10%)", hi_f),
        ("top5 position frequency, LOW-score windows (bottom 50%)", lo_f),
    ):
        print(f"\n=== {name} ===")
        print(m)
        print("  rings (outer -> inner):", {k: round(v, 3) for k, v in rings(m).items()})

    print("\ninner 8x8 share of top5 selections:  high-score windows "
          f"{hi_hits[3:11, 3:11].sum() / hi_hits.sum():.1%}   low-score windows {lo_hits[3:11, 3:11].sum() / lo_hits.sum():.1%}"
          f"   (uniform would be {8 * 8 / GRID**2:.1%})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "correlation_with_frame_score": corr.tolist(), "coefficient_of_variation": cv.tolist(),
        "top5_freq_high_score_windows": hi_f.tolist(), "top5_freq_low_score_windows": lo_f.tolist(),
        "rings": {n: rings(m) for n, m in (("corr", corr), ("cv", cv), ("hi", hi_f), ("lo", lo_f))},
        "inner8x8_share_high": float(hi_hits[3:11, 3:11].sum() / hi_hits.sum()),
        "inner8x8_share_low": float(lo_hits[3:11, 3:11].sum() / lo_hits.sum()),
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
