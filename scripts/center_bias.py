"""Tests A and C: does surprise depend on WHERE in the frame something happens (H5), not just what?

Observation (user, 2026-09-19): surprise seems to spike whenever anything moves in the central ~70-80% of the
frame, and in clip 04 it stayed elevated across the whole sprint rather than spiking at the entry and decaying.
The second point argues against an entry/new-information account; the first suggests a spatial prior.

The trap: in Avenue people walk through the middle, so "error is high in the centre" is exactly what content
alone predicts. Everything here is built to separate position from content.

  Test A -- the spatial error map, restricted to tokens with NO foreground in them. Empty background tokens
            have no pedestrian content to explain a centre-edge gap, so any gap left is positional.
  Test C -- where the top-5% tokens that drive the score actually sit. If they cluster centrally even when
            people do not, the centre bias lives in my aggregate rather than in V-JEPA.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from vjepa.data.avenue import frame_labels, test_videos
from vjepa.eval.avenue_eval import CACHE, PRIMARY
from vjepa.eval.scene_events import background, foreground_masks, read_gray

RUN_DIR = CACHE / "vitl16" / "fs4_stride2"
GRID_DIR = Path("outputs/investigations/fg_grid")
OUT = Path("outputs/investigations/center_bias.json")
GRID = 14
WINDOW = 16
TOP_FRAC = 0.05
EMPTY = 0.02  # a grid cell counts as background if under 2% of its pixels are foreground


def fg_grid(path: Path) -> np.ndarray:
    """(T, 14, 14) foreground fraction per grid cell, matching the token grid's geometry.

    crop="square" squashes the whole 640x360 frame into the square the model sees, so a grid cell is the whole
    frame divided 14 ways on each axis -- no cropping to undo.
    """
    cache = GRID_DIR / f"{path.stem}.npz"
    if cache.exists():
        return np.load(cache)["fg"]
    gray = read_gray(path)
    bg = background(gray)
    out = np.stack(
        [
            np.asarray(Image.fromarray(mask.astype(np.float32)).resize((GRID, GRID), Image.BOX))
            for mask, _, _ in foreground_masks(gray, bg)
        ]
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fg=out)
    return out


def token_occupancy(data: dict, fg: np.ndarray, context: int) -> np.ndarray:
    """(W, S, 14, 14) foreground fraction for the frames each predicted token actually covers."""
    starts, step = data["starts"], int(data["frame_step"])
    slices = (WINDOW - context) // 2
    out = np.zeros((len(starts), slices, GRID, GRID), np.float32)
    for w, s in enumerate(starts):
        for k in range(slices):
            lo = (s + context + 2 * k) * step  # absolute tubelet (context/2 + k), two sampled frames wide
            out[w, k] = fg[lo : lo + 2 * step].mean(0) if lo < len(fg) else fg[-1]
    return out


def radial_profile(m: np.ndarray) -> dict[str, float]:
    """Mean of a 14x14 map by ring, from the outer border inwards."""
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    ring = np.minimum(np.minimum(yy, GRID - 1 - yy), np.minimum(xx, GRID - 1 - xx))
    return {f"ring{r}": float(m[ring == r].mean()) for r in range(GRID // 2)}


def main() -> None:
    context = PRIMARY["context"]
    err_all = np.zeros((GRID, GRID))
    err_bg = np.zeros((GRID, GRID))
    n_all = np.zeros((GRID, GRID))
    n_bg = np.zeros((GRID, GRID))
    top_hits = np.zeros((GRID, GRID))
    top_hits_bg = np.zeros((GRID, GRID))
    n_top = 0

    for path in test_videos():
        stem = path.stem
        data = dict(np.load(RUN_DIR / f"{stem}.npz"))
        tok = data[f"tokens_c{context}"].astype(np.float32)  # (W, S, 14, 14)
        fg = fg_grid(path)
        occ = token_occupancy(data, fg, context)

        # Spatial pattern only: divide each window-slice by its own mean so a globally hard window cannot
        # masquerade as a spatial effect.
        rel = tok / (tok.mean(axis=(2, 3), keepdims=True) + 1e-8)
        err_all += rel.sum(axis=(0, 1))
        n_all += rel.shape[0] * rel.shape[1]
        bg_mask = occ < EMPTY
        err_bg += (rel * bg_mask).sum(axis=(0, 1))
        n_bg += bg_mask.sum(axis=(0, 1))

        flat = tok.reshape(len(tok), -1)
        k = max(1, int(TOP_FRAC * flat.shape[1]))
        idx = np.argsort(flat, axis=1)[:, -k:]  # the tokens the top5 aggregate actually averages
        pos = np.unravel_index(idx.ravel(), tok.shape[1:])
        np.add.at(top_hits, (pos[1], pos[2]), 1)
        sel_bg = bg_mask.reshape(len(tok), -1)[np.arange(len(tok))[:, None], idx].ravel()
        np.add.at(top_hits_bg, (pos[1][sel_bg], pos[2][sel_bg]), 1)
        n_top += idx.size
        print(f"  {stem}: {len(tok)} windows, {bg_mask.mean():.0%} of tokens are empty background")

    maps = {
        "error_all_tokens": err_all / n_all,
        "error_background_tokens_only": err_bg / np.maximum(n_bg, 1),
        "top5_selection_frequency": top_hits / n_top * (GRID * GRID),  # 1.0 == uniform
        "top5_selection_frequency_background_only": top_hits_bg / max(top_hits_bg.sum(), 1) * (GRID * GRID),
    }

    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for name, m in maps.items():
        print(f"\n=== {name} (14x14, row 0 = top of frame) ===")
        print(m)
        print("  by ring (outer -> inner):", {k: round(v, 3) for k, v in radial_profile(m).items()})

    bgmap = maps["error_background_tokens_only"]
    border, inner = bgmap[0].mean() + bgmap[-1].mean(), bgmap[3:11, 3:11].mean()
    print(f"\nbackground-only error: border rows {border / 2:.3f} vs inner 8x8 {inner:.3f}  ratio {inner / (border / 2):.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {k: v.tolist() for k, v in maps.items()}
            | {"rings": {k: radial_profile(v) for k, v in maps.items()}, "empty_token_threshold": EMPTY},
            indent=2,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
