"""Detection delay of the Surprise Meter on CUHK Avenue at a fixed false-alarm rate.

Question (from a reader of post 1): how early does the surprise signal rise relative to the annotated event onset,
and what is the detection delay at a fixed false-alarm rate, alongside the AUC?

The curve behind the published 0.734 AUC cannot answer this. `frame_scores` spreads each window's score over frames
up to 2.4 s BEFORE the window ends, then applies a symmetric Gaussian and a per-video min-max, so it uses the
future. A delay measured on it would be negative by construction. This script builds a causal score instead.

Protocol, fixed before any result was computed:
  score      cached tokens, C=8, top-5% (the PRIMARY config). Each window's score is stamped at its LAST frame, the
             moment that frame is decoded, and held until the next window. No smoothing and no per-video
             normalization in the primary. Exploratory variant: mean of the last 3 windows (~1 s).
  threshold  calibrated on the 16 Avenue TRAINING videos (10.2 min of normal footage, never used before): the
             lowest threshold whose alarm rate on them is at most the target. Alarm = a window above threshold with
             no other above-threshold window in the previous 2 s, so a lingering excursion is one alarm.
             Targets 0.5 / 1 / 2 false alarms per minute; 1 per minute is the headline.
  events     annotated segments closer than 2 s are merged into one event (74 raw segments are heavily fragmented).
             Events starting before frame 100 are dropped: there is no onset to detect when the video starts mid-event.
  detection  an event is detected if any window in [onset - 2 s, event end] is above threshold; delay is the first
             such window's stamp minus the onset. Negative delay = the alarm precedes the annotated onset.
  null       the same procedure on pseudo-events of the same length at random positions on normal footage
             (>= 2 s from any annotation), 200 draws per event. This is what chance detection looks like at this FAR.
  CIs        bootstrap over videos (events within a video are not independent).
  check      realised false-alarm rate on the normal frames of the test videos is reported next to the target.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vjepa.data.avenue import frame_labels, test_videos, train_videos
from vjepa.eval.avenue_eval import CACHE, PRIMARY, frame_scores, postprocess, video_scores
from vjepa.eval.intphys_eval import token_aggregates
from vjepa.video import read_video

FPS = 25
WINDOW = 16
REFRACT = 2 * FPS
PRE = 2 * FPS
MERGE_GAP = 2 * FPS
MIN_ONSET = 100
MIN_FRAMES = 100  # videos 08 and 21 are shorter than the window at the usual frame step
TARGETS = (0.5, 1.0, 2.0)  # false alarms per minute
HEADLINE = 1.0
DRAWS = 200
BOOT = 2000
TEST_DIR = CACHE / "vitl16" / "fs4_stride2"
TRAIN_DIR = CACHE / "vitl16" / "train_fs4_stride2"
OUT = Path("outputs/investigations/detection_delay.json")
rng = np.random.default_rng(0)


def extract_train() -> None:
    from vjepa.device import DTYPES, pick_device
    from vjepa.models import load_vjepa
    from vjepa.surprise import SurpriseMeter

    todo = [p for p in train_videos() if not (TRAIN_DIR / f"{p.stem}.npz").exists()]
    if not todo:
        return
    device = pick_device(None)
    meter = SurpriseMeter(load_vjepa("vitl16", device), device, DTYPES["fp16"])
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for p in todo:
        np.savez(TRAIN_DIR / f"{p.stem}.npz", **video_scores(meter, p, 4, (PRIMARY["context"],), 2))
        print(f"extracted train {p.stem}", flush=True)


def window_series(data: dict, smooth: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(stamp in raw frames, causal score) per window; the stamp is the window's last frame."""
    c, step = PRIMARY["context"], int(data["frame_step"])
    score = token_aggregates(data[f"tokens_c{c}"].astype(np.float32))[PRIMARY["token_agg"]]
    if smooth > 1:
        pad = np.r_[np.full(smooth - 1, score[0]), score]
        score = np.convolve(pad, np.ones(smooth) / smooth, mode="valid")
    return (data["starts"] + WINDOW - 1) * step, score


def alarms(t: np.ndarray, s: np.ndarray, thr: float) -> int:
    above = t[s > thr]
    return 0 if not len(above) else 1 + int((np.diff(above) > REFRACT).sum())


def lowest_passing(candidates_desc, rate) -> float:
    """Lowest threshold whose alarm rate is within target, scanning down from the top and stopping at the first miss.

    Merging alarms makes the rate non-monotone at very low thresholds (one endless excursion counts as one alarm),
    so an upward scan would wrongly accept a near-zero threshold.
    """
    best = None
    for thr in candidates_desc:
        if not rate(thr):
            break
        best = float(thr)
    if best is None:
        raise RuntimeError("no threshold reaches the target")
    return best


def calibrate(series: list[tuple[np.ndarray, np.ndarray]], per_min: float) -> float:
    minutes = sum(t[-1] - t[0] for t, _ in series) / FPS / 60
    cands = np.unique(np.concatenate([s for _, s in series]))[::-1]
    return lowest_passing(cands, lambda thr: sum(alarms(t, s, thr) for t, s in series) / minutes <= per_min)


def events_of(labels: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, labels.astype(int), 0])
    segs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    merged: list[list[int]] = []
    for a, b in segs:
        if merged and a - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = int(b)
        else:
            merged.append([int(a), int(b)])
    return [(a, b) for a, b in merged if a >= MIN_ONSET]


def detect(t: np.ndarray, s: np.ndarray, thr: float, onset: int, end: int) -> float | None:
    hit = np.flatnonzero((t >= onset - PRE) & (t <= end) & (s > thr))
    return None if not len(hit) else float(t[hit[0]] - onset) / FPS


def null_delays(t: np.ndarray, s: np.ndarray, thr: float, labels: np.ndarray, length: int, n: int) -> list:
    near = np.convolve(labels.astype(float), np.ones(2 * PRE + 1), mode="same") > 0
    ok = [o for o in range(MIN_ONSET, len(labels) - length)
          if not near[o - PRE : o + length + PRE].any() and o - PRE >= t[0]]
    if not ok:
        return []
    return [detect(t, s, thr, o, o + length) for o in rng.choice(ok, size=n)]


def summarize(delays: list) -> dict:
    d = np.array([x for x in delays if x is not None])
    return {
        "events": len(delays),
        "detected": float(len(d) / len(delays)),
        "median_delay_s": float(np.median(d)) if len(d) else None,
        "detected_within_1s": float(np.mean([x is not None and x <= 1 for x in delays])),
        "detected_within_2s": float(np.mean([x is not None and x <= 2 for x in delays])),
        "detected_before_onset": float(np.mean([x is not None and x < 0 for x in delays])),
    }


def run(smooth: int, label: str) -> dict:
    train = [window_series(dict(np.load(TRAIN_DIR / f"{p.stem}.npz")), smooth) for p in train_videos()]
    test = {}
    for p in test_videos():
        lab = frame_labels(p.stem)
        if len(lab) >= MIN_FRAMES:
            test[p.stem] = (*window_series(dict(np.load(TEST_DIR / f"{p.stem}.npz")), smooth), lab)

    # causal AUC of the very same score: each frame takes the latest window stamped at or before it
    ys, ss = [], []
    for t, s, lab in test.values():
        idx = np.searchsorted(t, np.arange(len(lab)), side="right") - 1
        keep = idx >= 0
        ys.append(lab[keep]); ss.append(s[idx[keep]])
    result = {"label": label, "causal_micro_auc": float(roc_auc_score(np.concatenate(ys), np.concatenate(ss)))}

    for target in TARGETS:
        thr = calibrate(train, target)
        real, null, per_video = [], [], {}
        fa_n, fa_min = 0, 0.0
        for stem, (t, s, lab) in test.items():
            evs = events_of(lab)
            per_video[stem] = [detect(t, s, thr, a, b) for a, b in evs]
            for a, b in evs:
                null += null_delays(t, s, thr, lab, b - a, DRAWS)
            # realised FA rate on normal stretches (>= 2 s from any anomaly)
            near = np.convolve(lab.astype(float), np.ones(2 * PRE + 1), mode="same") > 0
            for seg in np.split(np.arange(len(lab)), np.flatnonzero(np.diff(near.astype(int))) + 1):
                if len(seg) and not near[seg[0]]:
                    m = (t >= seg[0]) & (t <= seg[-1])
                    if m.sum() > 1:
                        fa_n += alarms(t[m], s[m], thr)
                        fa_min += (t[m][-1] - t[m][0]) / FPS / 60
        stems = [k for k, v in per_video.items() if v]
        real = [d for k in stems for d in per_video[k]]
        boots = []
        for _ in range(BOOT):
            pick = rng.choice(stems, size=len(stems))
            boots.append(summarize([d for k in pick for d in per_video[k]]))
        ci = {k: [float(np.nanpercentile([b[k] if b[k] is not None else np.nan for b in boots], q)) for q in (2.5, 97.5)]
              for k in ("detected", "median_delay_s", "detected_within_1s", "detected_within_2s")}
        result[f"far_{target}"] = {
            "threshold": thr,
            "realised_far_per_min_on_test_normals": fa_n / fa_min,
            "real": summarize(real),
            "real_ci95_bootstrap_over_videos": ci,
            "null": summarize(null),
            "delays_s": sorted(round(x, 2) for x in real if x is not None),
        }
    return result


def noncausal_illustration() -> dict:
    """The published curve (centered windows, sigma=12, per-video min-max), same FAR calibration: what NOT to report."""
    def curve(p, data, n):
        return postprocess(frame_scores(data, PRIMARY["context"], PRIMARY["token_agg"], n), PRIMARY["sigma"])

    train = []
    for p in train_videos():
        n = len(read_video(p, 1)[0])
        train.append(curve(p, dict(np.load(TRAIN_DIR / f"{p.stem}.npz")), n))
    out = {}
    for target in TARGETS:
        minutes = sum(len(c) for c in train) / FPS / 60
        thr = lowest_passing(
            np.linspace(1, 0, 2001),
            lambda th: sum(alarms(np.arange(len(c)), c, th) for c in train) / minutes <= target,
        )
        delays = []
        for p in test_videos():
            lab = frame_labels(p.stem)
            if len(lab) < MIN_FRAMES:
                continue
            c = curve(p, dict(np.load(TEST_DIR / f"{p.stem}.npz")), len(lab))
            delays += [detect(np.arange(len(c)), c, thr, a, b) for a, b in events_of(lab)]
        out[f"far_{target}"] = summarize(delays)
    return out


def main() -> None:
    extract_train()
    res = {"primary": run(1, "causal, no smoothing"), "smoothed3": run(3, "causal, trailing mean of 3 windows")}
    res["noncausal_published_curve"] = noncausal_illustration()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    for name in ("primary", "smoothed3"):
        r = res[name]
        print(f"\n== {r['label']}  (causal frame AUC {r['causal_micro_auc']:.3f})")
        for target in TARGETS:
            x = r[f"far_{target}"]
            print(f" FAR {target}/min (test realised {x['realised_far_per_min_on_test_normals']:.2f}): "
                  f"real {x['real']}\n{'':30}null {x['null']}")
    print("\n== non-causal published curve", json.dumps(res["noncausal_published_curve"], indent=1))


if __name__ == "__main__":
    main()
