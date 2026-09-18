"""Stage A: does surprise spike on ordinary, non-anomalous moments as much as on labeled anomalies?

Hypothesis (user, 2026-09-15): V-JEPA's predictor makes one deterministic guess at a genuinely multimodal
future (a new pedestrian could enter from either side, or not; an existing one could keep walking, stop, turn).
Whatever actually happens will look like a "miss" relative to that hedge almost regardless of whether it was
behaviorally anomalous — so surprise may really be tracking "was the next moment hard to predict," which
correlates with anomalies but isn't the same thing.

This script checks it across all 21 Avenue test videos (not just the one used in the first post): for every
video, find every local peak in the raw (unsmoothed, un-normalized) surprise curve, and compare the magnitude
of peaks that land on labeled-anomalous frames vs. peaks that don't.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from vjepa.data.avenue import frame_labels
from vjepa.eval.avenue_eval import CACHE, PRIMARY, frame_scores

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
PEAK_WINDOW = 15  # frames on each side considered for local-maximum detection (~0.6s at 25fps)
PEAK_PERCENTILE = 70  # ignore tiny wobble; only count peaks above this percentile of that video's own scores


def find_peaks(curve: np.ndarray) -> np.ndarray:
    is_peak = (curve == maximum_filter1d(curve, size=PEAK_WINDOW)) & (curve > np.percentile(curve, PEAK_PERCENTILE))
    idx = np.flatnonzero(is_peak)
    kept: list[int] = []
    for i in sorted(idx, key=lambda i: -curve[i]):  # de-dup peaks within PEAK_WINDOW of a stronger one
        if all(abs(i - k) > PEAK_WINDOW for k in kept):
            kept.append(i)
    return np.array(sorted(kept))


def main() -> None:
    rows = []
    for path in sorted(RUN_DIR.glob("*.npz")):
        if path.stem in ("report",):
            continue
        data = dict(np.load(path))
        labels = frame_labels(path.stem)
        if not (0 < labels.sum() < len(labels)):
            continue  # skip videos that are all-anomalous or all-normal (no contrast to measure)
        curve = frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], len(labels))
        peaks = find_peaks(curve)
        if len(peaks) == 0:
            continue
        peak_labels = labels[peaks].astype(bool)
        rows.append(
            {
                "video": path.stem,
                "n_peaks": len(peaks),
                "n_labeled_peaks": int(peak_labels.sum()),
                "n_unlabeled_peaks": int((~peak_labels).sum()),
                "median_labeled_peak": float(np.median(curve[peaks][peak_labels])) if peak_labels.any() else np.nan,
                "median_unlabeled_peak": float(np.median(curve[peaks][~peak_labels])) if (~peak_labels).any() else np.nan,
                "video_median": float(np.median(curve)),
                "video_p95": float(np.percentile(curve, 95)),
            }
        )

    print(f"{'video':>6} {'peaks':>6} {'labeled':>8} {'unlabeled':>10} {'med.lbl':>9} {'med.unlbl':>10} {'vid.med':>8} {'vid.p95':>8}")
    for r in rows:
        print(
            f"{r['video']:>6} {r['n_peaks']:>6} {r['n_labeled_peaks']:>8} {r['n_unlabeled_peaks']:>10} "
            f"{r['median_labeled_peak']:>9.4f} {r['median_unlabeled_peak']:>10.4f} {r['video_median']:>8.4f} {r['video_p95']:>8.4f}"
        )

    # Dataset-wide: for videos with BOTH labeled and unlabeled peaks, how much higher is a labeled peak, typically?
    both = [r for r in rows if not np.isnan(r["median_labeled_peak"]) and not np.isnan(r["median_unlabeled_peak"])]
    gaps = np.array([r["median_labeled_peak"] - r["median_unlabeled_peak"] for r in both])
    unlabeled_above_video_p95 = np.mean(
        [r["median_unlabeled_peak"] > r["video_p95"] for r in both]
    )  # do unlabeled peaks even clear the video's own top-5% threshold?
    print(f"\n{len(both)}/{len(rows)} videos have both labeled and unlabeled peaks to compare.")
    print(f"labeled-peak minus unlabeled-peak median gap: mean {gaps.mean():.4f}, median {np.median(gaps):.4f}, "
          f"range [{gaps.min():.4f}, {gaps.max():.4f}]")
    print(f"median unlabeled peak exceeds that video's own 95th-percentile score in {unlabeled_above_video_p95:.0%} "
          f"of videos (compare with each video's median to see how far above baseline unlabeled peaks sit)")
    print(f"unlabeled peaks HIGHER than the median labeled peak in {(gaps < 0).mean():.0%} of videos")

    Path("outputs/investigations").mkdir(parents=True, exist_ok=True)
    np.savez("outputs/investigations/entrant_confound_stage_a.npz", rows=rows, gaps=gaps)


if __name__ == "__main__":
    main()
