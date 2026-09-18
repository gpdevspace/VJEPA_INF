"""Stage 2: can dumb low-level covariates match V-JEPA's Avenue AUC?

Two confounds have to fall before "surprise tracks unpredictable new information" can mean anything:

  H3 (motion energy) -- fast motion and blur are hard to predict whatever causes them. If plain frame
     differencing scores as well as V-JEPA, post 1's "zero-shot anomaly detection" framing needs qualifying.
  H4 (clutter)       -- more people means more hard tokens, which lifts a top-5% aggregate on its own.

Each covariate is pushed through exactly the postprocessing V-JEPA's scores get (Gaussian smoothing, then
per-video min-max) and scored with the same micro/macro AUC, so the comparison is like for like.

Beating V-JEPA and *explaining* V-JEPA are different claims, so this also reports the rank correlation between
each covariate and surprise on normal frames, and the AUC of the two averaged. A covariate that scores higher
while staying uncorrelated is a better detector, not an account of what surprise measures.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from vjepa.data.avenue import frame_labels, test_videos
from vjepa.eval.avenue_eval import CACHE, PRIMARY, frame_scores, postprocess
from vjepa.eval.scene_events import covariates, read_gray

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
COV_DIR = Path("outputs/investigations/covariates")
OUT = Path("outputs/investigations/motion_baseline.json")
COVARIATES = ("motion_energy", "foreground_area", "blob_count")
SIGMAS = (0, 6, 12, 24)
# 08 (36 frames) and 21 (76 frames) are shorter than ~2 V-JEPA windows, so their surprise curve has 3 distinct
# values and is near chance by construction. Reported both with and without them rather than quietly dropped.
DEGENERATE = {"08", "21"}


def video_covariates(path: Path) -> dict[str, np.ndarray]:
    """Covariates for one video, cached: decoding and blob-labelling every frame takes ~6 s."""
    cache = COV_DIR / f"{path.stem}.npz"
    if cache.exists():
        return dict(np.load(cache))
    cov = covariates(read_gray(path))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **cov)
    return cov


def load() -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    out = {}
    for path in test_videos():
        stem = path.stem
        lab = frame_labels(stem)
        cov = video_covariates(path)
        vjepa = frame_scores(dict(np.load(RUN_DIR / f"{stem}.npz")), PRIMARY["context"], PRIMARY["token_agg"], len(lab))
        series = {k: cov[k][: len(lab)] for k in COVARIATES} | {"vjepa": vjepa}
        assert all(len(v) == len(lab) for v in series.values()), f"{stem}: length mismatch"
        out[stem] = (lab, series)
        print(f"  scored {stem} ({len(lab)} frames)")
    return out


def auc(data, name: str, sigma: float, videos: set[str]) -> dict[str, float]:
    pooled_s, pooled_l, per_video = [], [], []
    for stem in videos:
        lab, series = data[stem]
        s = postprocess(series[name], sigma) if name != "vjepa+motion" else (
            postprocess(series["vjepa"], sigma) + postprocess(series["motion_energy"], sigma)
        ) / 2
        pooled_s.append(s)
        pooled_l.append(lab)
        if 0 < lab.sum() < len(lab):
            per_video.append(roc_auc_score(lab, s))
    return {
        "micro_auc": float(roc_auc_score(np.concatenate(pooled_l), np.concatenate(pooled_s))),
        "macro_auc": float(np.mean(per_video)),
    }


def main() -> None:
    data = load()
    signals = (*COVARIATES, "vjepa", "vjepa+motion")
    all_videos = set(data)
    kept = all_videos - DEGENERATE

    headline = {n: auc(data, n, PRIMARY["sigma"], all_videos) for n in signals}
    without_degenerate = {n: auc(data, n, PRIMARY["sigma"], kept) for n in signals}
    sensitivity = {f"sigma={s}": {n: auc(data, n, s, all_videos) for n in signals} for s in SIGMAS}

    # How much of V-JEPA's curve is just this covariate? Measured on normal frames only, so labeled anomalies
    # (where both signals are expected to rise together) cannot manufacture the correlation.
    corr = {}
    for name in COVARIATES:
        rhos = [
            spearmanr(series["vjepa"][~lab], series[name][~lab]).statistic
            for lab, series in data.values()
            if (~lab).sum() > 10
        ]
        corr[name] = {"median_rho": float(np.median(rhos)), "min": float(np.min(rhos)), "max": float(np.max(rhos))}

    print(f"\n{'signal':>16} {'micro':>7} {'macro':>7}   {'rho vs V-JEPA (normal frames)':>32}")
    for n in signals:
        c = corr.get(n)
        tail = f"{c['median_rho']:+.3f} [{c['min']:+.2f}, {c['max']:+.2f}]" if c else ""
        print(f"{n:>16} {headline[n]['micro_auc']:>7.3f} {headline[n]['macro_auc']:>7.3f}   {tail:>32}")

    print(f"\nexcluding {sorted(DEGENERATE)} (too short for a meaningful surprise curve):")
    for n in signals:
        print(f"{n:>16} {without_degenerate[n]['micro_auc']:>7.3f} {without_degenerate[n]['macro_auc']:>7.3f}")

    print("\nsmoothing sensitivity (micro/macro):")
    for s in SIGMAS:
        r = sensitivity[f"sigma={s}"]
        print(f"  sigma={s:>2}  " + "  ".join(f"{n}={r[n]['micro_auc']:.3f}/{r[n]['macro_auc']:.3f}" for n in signals))

    best = max(COVARIATES, key=lambda k: headline[k]["micro_auc"])
    beats = headline[best]["micro_auc"] >= headline["vjepa"]["micro_auc"]
    verdict = {
        "h3_beats_vjepa_as_detector": bool(beats),
        "best_covariate": best,
        "h3_explains_surprise": bool(abs(corr["motion_energy"]["median_rho"]) > 0.5),
        "h4_clutter_explains_surprise": bool(abs(corr["blob_count"]["median_rho"]) > 0.5),
        "summary": (
            f"{best} reaches micro AUC {headline[best]['micro_auc']:.3f} vs V-JEPA's {headline['vjepa']['micro_auc']:.3f}, "
            f"but correlates with surprise at only rho={corr['motion_energy']['median_rho']:+.3f} on normal frames, and "
            f"averaging the two gives {headline['vjepa+motion']['micro_auc']:.3f} micro / "
            f"{headline['vjepa+motion']['macro_auc']:.3f} macro -- so motion is a better Avenue detector than "
            f"surprise without being an account of what surprise measures."
        ),
    }
    print(f"\n{verdict['summary']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "auc": headline,
                "auc_excluding_degenerate": without_degenerate,
                "smoothing_sensitivity": sensitivity,
                "spearman_vs_vjepa_normal_frames": corr,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
