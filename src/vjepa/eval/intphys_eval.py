"""IntPhys dev evaluation of the Surprise Meter.

Metrics follow jepa-intuitive-physics `compute_metrics`: within each matched (possible, impossible) pair, is the
possible movie less surprising? Surprise per movie is the max or mean over windows, per context length, or
"filtered" (min over context lengths first). Because the dev set is also the only labeled split, each block's
scenes are split in half: the configuration is chosen on one half and reported on the other.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from vjepa.data.intphys import list_scenes, load_frames, match_pairs
from vjepa.surprise import SurpriseMeter
from vjepa.video import preprocess

CACHE = Path("outputs/intphys")
AGGREGATIONS = ("max", "mean")  # over a movie's windows
TOKEN_AGGREGATES = ("mean", "top5", "max")  # over a window's predicted tokens; "mean" is the reference


def token_aggregates(token_errors: np.ndarray) -> dict[str, np.ndarray]:
    """Per-token errors `(W, S, gh, gw)` -> per-window scores.

    The reference averages over every predicted token, which dilutes a small object's violation; "top5" (mean of
    the 5% largest errors) and "max" keep spatially localized surprise.
    """
    flat = token_errors.reshape(len(token_errors), -1)
    k = max(1, int(0.05 * flat.shape[1]))
    return {"mean": flat.mean(1), "top5": np.sort(flat, axis=1)[:, -k:].mean(1), "max": flat.max(1)}


def scene_scores(meter: SurpriseMeter, movies, frame_step: int, contexts: Sequence[int], stride: int) -> dict:
    """Window scores `(4, len(contexts), W)` per token aggregate for a scene's movies, plus labels and pairs."""
    clips = [load_frames(m, frame_step) for m in movies]
    per_movie = []
    for clip in clips:
        result = meter.score_clip(preprocess(clip), contexts, stride)
        per_ctx = [token_aggregates(errs) for errs in result.token_errors]
        per_movie.append({tok: np.stack([a[tok] for a in per_ctx]) for tok in TOKEN_AGGREGATES})
    return {
        **{f"scores_{tok}": np.stack([m[tok] for m in per_movie]) for tok in TOKEN_AGGREGATES},
        "possible": np.array([m.possible for m in movies]),
        "pairs": np.array(match_pairs(clips)),
        "contexts": np.array(contexts),
    }


def extract_block(
    meter: SurpriseMeter,
    block: str,
    frame_step: int,
    contexts: Sequence[int],
    stride: int,
    cache_dir: Path,
    progress: Callable[[str], None] | None = None,
    max_scenes: int | None = None,
) -> dict[str, dict]:
    """Scores for every scene of a block, cached per scene so sweeps and reruns are free."""
    out = {}
    for scene, movies in list(list_scenes(block).items())[:max_scenes]:
        path = cache_dir / block / f"{scene}.npz"
        data = dict(np.load(path)) if path.exists() else None
        if data is None or "scores_top5" not in data or not np.array_equal(data["contexts"], contexts):
            data = scene_scores(meter, movies, frame_step, contexts, stride)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **data)
        out[scene] = data
        if progress:
            progress(f"{block}/{scene}")
    return out


def pair_surprise(
    scenes: dict[str, dict], context_index: int | None, agg: str, token_agg: str = "mean"
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregated surprise of the possible and impossible movie of every matched pair.

    `context_index=None` is the reference's "filtered" variant: min over context lengths per window.
    `token_agg` picks how each window's token errors were reduced ("mean" is the reference).
    """
    possible, impossible = [], []
    for data in scenes.values():
        scores = data[f"scores_{token_agg}"]
        s = scores.min(1) if context_index is None else scores[:, context_index]
        per_movie = s.max(-1) if agg == "max" else s.mean(-1)
        for a, b in data["pairs"]:
            if data["possible"][a] == data["possible"][b]:
                raise ValueError("matched pair does not contain one possible and one impossible movie")
            p, i = (a, b) if data["possible"][a] else (b, a)
            possible.append(per_movie[p])
            impossible.append(per_movie[i])
    return np.array(possible), np.array(impossible)


def pair_metrics(possible: np.ndarray, impossible: np.ndarray) -> dict:
    wins, n = int((possible < impossible).sum()), len(possible)
    is_impossible = np.r_[np.zeros(n), np.ones(n)]
    return {
        "pairs": n,
        "relative_accuracy": wins / n,
        "p_value": binomtest(wins, n, 0.5, alternative="greater").pvalue,  # vs chance
        "auroc": roc_auc_score(is_impossible, np.r_[possible, impossible]),  # single-video detection
    }


def config_name(contexts: Sequence[int], context_index: int | None, agg: str, token_agg: str = "mean") -> str:
    ctx = "min-over-C" if context_index is None else f"C={contexts[context_index]}"
    return f"{ctx}/{agg}-over-time/{token_agg}-over-tokens"


def evaluate(block_scenes: dict[str, dict[str, dict]], contexts: Sequence[int], seed: int = 0) -> dict:
    """Per block: pick the config with the best relative accuracy on half the scenes, report it on the other half."""
    configs = [
        (ci, agg, tok) for tok in TOKEN_AGGREGATES for ci in [*range(len(contexts)), None] for agg in AGGREGATIONS
    ]
    rng = np.random.default_rng(seed)
    report = {"blocks": {}, "contexts": list(contexts), "seed": seed}
    pooled_pos, pooled_imp = [], []
    for block, scenes in block_scenes.items():
        names = sorted(scenes, key=int)
        order = rng.permutation(len(names))
        tune = {names[k]: scenes[names[k]] for k in order[: len(names) // 2]}
        test = {names[k]: scenes[names[k]] for k in order[len(names) // 2 :]}

        def score(cfg, subset):
            return pair_metrics(*pair_surprise(subset, *cfg))

        best = max(configs, key=lambda cfg: (score(cfg, tune)["relative_accuracy"], score(cfg, tune)["auroc"]))
        pos, imp = pair_surprise(test, *best)
        pooled_pos.append(pos)
        pooled_imp.append(imp)
        report["blocks"][block] = {
            "chosen_on_tune_half": config_name(contexts, *best),
            "tune": score(best, tune),
            "held_out": pair_metrics(pos, imp),
            "all_scenes_every_config": {config_name(contexts, *cfg): score(cfg, scenes) for cfg in configs},
        }
    report["held_out_pooled"] = pair_metrics(np.concatenate(pooled_pos), np.concatenate(pooled_imp))
    return report


def format_report(report: dict, blocks_desc: dict[str, str]) -> str:
    lines = [
        "| block | property | config (chosen on tune half) | held-out rel. acc | pairs | p vs chance | AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for block, r in report["blocks"].items():
        h = r["held_out"]
        lines.append(
            f"| {block} | {blocks_desc[block]} | {r['chosen_on_tune_half']} | {h['relative_accuracy']:.1%} | "
            f"{h['pairs']} | {h['p_value']:.2g} | {h['auroc']:.3f} |"
        )
    p = report["held_out_pooled"]
    lines.append(
        f"| all | pooled | per-block | {p['relative_accuracy']:.1%} | {p['pairs']} | {p['p_value']:.2g} | {p['auroc']:.3f} |"
    )
    return "\n".join(lines)
