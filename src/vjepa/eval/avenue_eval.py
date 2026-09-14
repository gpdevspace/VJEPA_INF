"""Zero-shot frame-level anomaly detection on CUHK Avenue with the Surprise Meter.

Nothing is trained on Avenue: each test video is scored by V-JEPA's prediction error on its own future. Scores
are mapped back to every frame, then (standard in video anomaly detection) optionally Gaussian-smoothed and
min-max normalized per video. Frame-level AUC is reported micro (all frames pooled) and macro (mean per video).

PRIMARY was fixed before looking at any Avenue result; every other configuration is exploratory.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

from vjepa.data.avenue import frame_labels, test_videos
from vjepa.eval.intphys_eval import TOKEN_AGGREGATES, token_aggregates
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess, read_video

CACHE = Path("outputs/avenue")
PRIMARY = {"context": 8, "token_agg": "top5", "sigma": 12}
SIGMAS = (0, 12)  # frames at 25 fps: raw, and ~0.5 s smoothing


def video_scores(meter: SurpriseMeter, path: Path, frame_step: int, contexts: Sequence[int], stride: int) -> dict:
    window = meter.model.spec.num_frames
    frames, _ = read_video(path, frame_step)
    while len(frames) < window and frame_step > 1:  # a few test videos are only ~1.5 s long
        frame_step //= 2
        frames, _ = read_video(path, frame_step)
    if len(frames) < window:
        frames = np.concatenate([frames, np.repeat(frames[-1:], window - len(frames), axis=0)])
    result = meter.score_clip(preprocess(frames), contexts, stride)
    return {
        "starts": result.starts,
        "contexts": np.array(contexts),
        "n_sampled": len(frames),
        "frame_step": frame_step,
        # per-token errors (W, S, gh, gw) per context: enough to re-aggregate or render heatmaps without rerunning
        **{f"tokens_c{c}": errs.astype(np.float16) for c, errs in zip(contexts, result.token_errors)},
    }


def extract(
    meter: SurpriseMeter,
    frame_step: int,
    contexts: Sequence[int],
    stride: int,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    out = {}
    for path in test_videos():
        cache = cache_dir / f"{path.stem}.npz"
        data = dict(np.load(cache)) if cache.exists() else None
        if data is None or not np.array_equal(data["contexts"], contexts):
            data = video_scores(meter, path, frame_step, contexts, stride)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **data)
        out[path.stem] = data
        if progress:
            progress(path.stem)
    return out


def frame_scores(data: dict, context: int, token_agg: str, total_frames: int, window: int = 16) -> np.ndarray:
    """One score per original frame: window scores spread over the frames they predict, then upsampled in time."""
    per_window = token_aggregates(data[f"tokens_c{context}"].astype(np.float32))[token_agg]
    n, step = int(data["n_sampled"]), int(data["frame_step"])
    acc, cnt = np.zeros(n), np.zeros(n)
    for s, v in zip(data["starts"], per_window):
        acc[s + context : s + window] += v
        cnt[s + context : s + window] += 1
    covered = np.flatnonzero(cnt)
    curve = np.interp(np.arange(n), covered, acc[covered] / cnt[covered])  # edges take the nearest covered value
    full = np.repeat(curve, step)[:total_frames]
    return np.pad(full, (0, total_frames - len(full)), mode="edge")


def postprocess(scores: np.ndarray, sigma: float) -> np.ndarray:
    s = gaussian_filter1d(scores, sigma) if sigma else scores
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


def config_auc(videos: dict[str, dict], context: int, token_agg: str, sigma: float) -> dict:
    all_scores, all_labels, per_video = [], [], []
    for stem, data in videos.items():
        labels = frame_labels(stem)
        scores = postprocess(frame_scores(data, context, token_agg, len(labels)), sigma)
        all_scores.append(scores)
        all_labels.append(labels)
        if 0 < labels.sum() < len(labels):
            per_video.append(roc_auc_score(labels, scores))
    return {
        "micro_auc": roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores)),
        "macro_auc": float(np.mean(per_video)),
        "videos": len(videos),
    }


def evaluate(videos: dict[str, dict], contexts: Sequence[int]) -> dict:
    grid = {
        f"C={c}/{tok}/sigma={sigma}": config_auc(videos, c, tok, sigma)
        for tok in TOKEN_AGGREGATES
        for c in contexts
        for sigma in SIGMAS
    }
    primary = f"C={PRIMARY['context']}/{PRIMARY['token_agg']}/sigma={PRIMARY['sigma']}"
    return {"primary": primary, "primary_result": grid[primary], "exploratory": grid}


def format_report(report: dict) -> str:
    p = report["primary_result"]
    lines = [
        f"Primary (fixed in advance) {report['primary']}: micro AUC {p['micro_auc']:.3f}, macro AUC {p['macro_auc']:.3f}",
        "",
        "| config (exploratory) | micro AUC | macro AUC |",
        "|---|---|---|",
    ]
    for name, r in sorted(report["exploratory"].items(), key=lambda kv: -kv[1]["micro_auc"]):
        lines.append(f"| {name} | {r['micro_auc']:.3f} | {r['macro_auc']:.3f} |")
    return "\n".join(lines)
