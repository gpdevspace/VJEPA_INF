"""Does a patch get harder or EASIER to predict when a person walks into it?

peak_source.py found that at the top-10% surprise peaks the excess error sits on empty background tokens
(z=+0.22) while tokens actually containing a person sit below their own median (z=-0.11). Taken at face value
that says a pedestrian makes a patch MORE predictable, which would invert the whole "surprise spots the person"
story. Two ways it could be an artifact instead:

  - position: people walk through the centre, and the centre is intrinsically low-error. Controlled here by
    comparing each grid cell against ITSELF, occupied vs empty, so position is held fixed by construction.
  - proximity: the "background" excess might just be the person's shadow, motion blur or the background they
    are uncovering, all of which my blob detector misses. Tested by binning background tokens on distance to
    the nearest occupied cell.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter1d

from vjepa.data.avenue import test_videos
from vjepa.eval.avenue_eval import CACHE, PRIMARY

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
GRID_DIR = Path("outputs/investigations/fg_grid")
OUT = Path("outputs/investigations/person_vs_background.json")
GRID, C, WINDOW, TOP_FRAC = 14, PRIMARY["context"], 16, 0.05
FG_OCCUPIED, MIN_SAMPLES = 0.10, 20


def token_occupancy(data, fg):
    starts, step = data["starts"], int(data["frame_step"])
    out = np.zeros((len(starts), (WINDOW - C) // 2, GRID, GRID), np.float32)
    for w, s in enumerate(starts):
        for k in range(out.shape[1]):
            lo = (s + C + 2 * k) * step
            out[w, k] = fg[lo : lo + 2 * step].mean(0) if lo < len(fg) else fg[-1]
    return out


def main() -> None:
    deltas, occupied_lvl, empty_lvl = [], [], []
    dist_bins = {d: [] for d in range(5)}
    peak_dist_bins = {d: [] for d in range(5)}

    for path in test_videos():
        data = dict(np.load(RUN_DIR / f"{path.stem}.npz"))
        tok = data[f"tokens_c{C}"].astype(np.float32)
        if len(tok) < 8:
            continue
        occ = token_occupancy(data, np.load(GRID_DIR / f"{path.stem}.npz")["fg"]) >= FG_OCCUPIED

        # --- within-cell paired test: same cell, occupied vs empty ---
        for gy in range(GRID):
            for gx in range(GRID):
                e, o = tok[:, :, gy, gx].ravel(), occ[:, :, gy, gx].ravel()
                if o.sum() >= MIN_SAMPLES and (~o).sum() >= MIN_SAMPLES:
                    deltas.append(e[o].mean() - e[~o].mean())
                    occupied_lvl.append(e[o].mean())
                    empty_lvl.append(e[~o].mean())

        # --- proximity: is background excess just near the person? ---
        z = (tok - np.median(tok, axis=0, keepdims=True)) / (tok.std(axis=0, keepdims=True) + 1e-6)
        flat = tok.reshape(len(tok), -1)
        k = max(1, int(TOP_FRAC * flat.shape[1]))
        score = np.sort(flat, axis=1)[:, -k:].mean(1)
        is_peak = (score == maximum_filter1d(score, 5)) & (score > np.percentile(score, 90))
        for w in range(len(tok)):
            for s in range(occ.shape[1]):
                m = occ[w, s]
                if not m.any() or m.all():
                    continue
                d = np.clip(distance_transform_edt(~m).astype(int), 0, 4)
                for b in range(1, 5):  # 0 would be the occupied cells themselves
                    v = z[w, s][d == b]
                    if v.size:
                        (peak_dist_bins if is_peak[w] else dist_bins)[b].append(v.mean())

    d = np.array(deltas)
    print(f"within-cell paired test over {len(d)} (video, cell) pairs -- same cell, occupied vs empty:")
    print(f"  mean error when a person is in the cell : {np.mean(occupied_lvl):.4f}")
    print(f"  mean error when the cell is empty       : {np.mean(empty_lvl):.4f}")
    print(f"  mean delta (occupied - empty)           : {d.mean():+.4f}")
    print(f"  cells where a person LOWERS error       : {(d < 0).mean():.1%}")

    print("\nbackground token z-score by distance (in grid cells) from the nearest person:")
    print(f"  {'distance':>9} {'calm windows':>14} {'peak windows':>14}")
    prox = {}
    for b in range(1, 5):
        c_, p_ = float(np.mean(dist_bins[b])), float(np.mean(peak_dist_bins[b]))
        prox[b] = {"calm": c_, "peak": p_}
        print(f"  {b:>9} {c_:>14.3f} {p_:>14.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_pairs": len(d), "mean_occupied": float(np.mean(occupied_lvl)), "mean_empty": float(np.mean(empty_lvl)),
        "mean_delta": float(d.mean()), "frac_person_lowers_error": float((d < 0).mean()),
        "z_by_distance_from_person": prox}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
