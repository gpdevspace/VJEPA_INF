"""Where does the VARIATION in the surprise curve actually come from?

Established so far: a blank grey clip already scores 0.46 mean token error against a real Avenue clip's 0.55,
so most of the score is a floor that has nothing to do with the video, and the raw Avenue curve only moves 4.4%
around its mean. The border gradient is positional rather than textural, but it is a near-constant offset --
constant things cannot produce peaks.

So this asks the remaining question directly: of the part that MOVES, how much sits on the pedestrians?

  1. Score each video three ways -- all tokens, centre tokens only, border tokens only -- and compare AUC.
     If the centre-only score is the better detector, the useful signal is on the people and the border is
     ballast dragging it down.
  2. At each peak, z-score every token against its own history and ask where the excess sits: on foreground
     (a person) or on empty background.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.metrics import roc_auc_score

from vjepa.data.avenue import frame_labels, test_videos
from vjepa.eval.avenue_eval import CACHE, PRIMARY, postprocess

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
GRID_DIR = Path("outputs/investigations/fg_grid")
OUT = Path("outputs/investigations/peak_source.json")
GRID, C, WINDOW, TOP_FRAC = 14, PRIMARY["context"], 16, 0.05
FG_OCCUPIED = 0.10  # a grid cell counts as holding a person if >=10% of its pixels are foreground

yy, xx = np.mgrid[0:GRID, 0:GRID]
RING = np.minimum(np.minimum(yy, GRID - 1 - yy), np.minimum(xx, GRID - 1 - xx))
CENTRE = RING >= 3  # inner 8x8
BORDER = RING <= 1  # outer two rings


def top5(vals: np.ndarray) -> np.ndarray:
    k = max(1, int(TOP_FRAC * vals.shape[1]))
    return np.sort(vals, axis=1)[:, -k:].mean(1)


def spread(per_window, data, n):
    ns, st = int(data["n_sampled"]), int(data["frame_step"])
    acc, cnt = np.zeros(ns), np.zeros(ns)
    for s, v in zip(data["starts"], per_window):
        acc[s + C : s + WINDOW] += v
        cnt[s + C : s + WINDOW] += 1
    cov = np.flatnonzero(cnt)
    f = np.repeat(np.interp(np.arange(ns), cov, acc[cov] / cnt[cov]), st)[:n]
    return np.pad(f, (0, n - len(f)), mode="edge")


def token_occupancy(data, fg):
    starts, step = data["starts"], int(data["frame_step"])
    slices = (WINDOW - C) // 2
    out = np.zeros((len(starts), slices, GRID, GRID), np.float32)
    for w, s in enumerate(starts):
        for k in range(slices):
            lo = (s + C + 2 * k) * step
            out[w, k] = fg[lo : lo + 2 * step].mean(0) if lo < len(fg) else fg[-1]
    return out


def main() -> None:
    variants = {"all_tokens": None, "centre_only": CENTRE, "border_only": BORDER}
    acc = {k: {"s": [], "l": [], "v": []} for k in variants}
    peak_fg, peak_bg, ctrl_fg, ctrl_bg, n_peaks = [], [], [], [], 0

    for path in test_videos():
        stem = path.stem
        lab = frame_labels(stem)
        data = dict(np.load(RUN_DIR / f"{stem}.npz"))
        tok = data[f"tokens_c{C}"].astype(np.float32)
        if len(tok) < 8:
            continue

        for name, mask in variants.items():
            sel = tok.reshape(len(tok), -1) if mask is None else tok[:, :, mask].reshape(len(tok), -1)
            curve = postprocess(spread(top5(sel), data, len(lab)), PRIMARY["sigma"])
            acc[name]["s"].append(curve)
            acc[name]["l"].append(lab)
            if 0 < lab.sum() < len(lab):
                acc[name]["v"].append(roc_auc_score(lab, curve))

        # where does the excess at a peak sit -- on people, or on empty background?
        occ = token_occupancy(data, np.load(GRID_DIR / f"{stem}.npz")["fg"])
        z = (tok - np.median(tok, axis=0, keepdims=True)) / (tok.std(axis=0, keepdims=True) + 1e-6)
        win_score = top5(tok.reshape(len(tok), -1))
        is_peak = (win_score == maximum_filter1d(win_score, 5)) & (win_score > np.percentile(win_score, 90))
        has_fg = occ >= FG_OCCUPIED
        for w in np.flatnonzero(is_peak):
            if has_fg[w].any() and (~has_fg[w]).any():
                peak_fg.append(z[w][has_fg[w]].mean())
                peak_bg.append(z[w][~has_fg[w]].mean())
                n_peaks += 1
        for w in np.flatnonzero(win_score <= np.percentile(win_score, 50)):
            if has_fg[w].any() and (~has_fg[w]).any():
                ctrl_fg.append(z[w][has_fg[w]].mean())
                ctrl_bg.append(z[w][~has_fg[w]].mean())

    print(f"{'scoring region':>16} {'micro':>7} {'macro':>7}")
    res = {}
    for k, r in acc.items():
        res[k] = {"micro_auc": float(roc_auc_score(np.concatenate(r["l"]), np.concatenate(r["s"]))),
                  "macro_auc": float(np.mean(r["v"]))}
        print(f"{k:>16} {res[k]['micro_auc']:>7.3f} {res[k]['macro_auc']:>7.3f}")

    pf, pb, cf, cb = (float(np.mean(x)) for x in (peak_fg, peak_bg, ctrl_fg, ctrl_bg))
    print(f"\nat the top-10% peaks (n={n_peaks}), mean z-score of tokens vs their own history:")
    print(f"  tokens holding a person : {pf:+.3f}        (calm windows: {cf:+.3f})")
    print(f"  empty background tokens : {pb:+.3f}        (calm windows: {cb:+.3f})")
    print(f"  peak-minus-calm lift    : person {pf - cf:+.3f}   background {pb - cb:+.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"auc_by_region": res, "n_peaks": n_peaks,
        "peak_z_foreground": pf, "peak_z_background": pb,
        "calm_z_foreground": cf, "calm_z_background": cb}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
