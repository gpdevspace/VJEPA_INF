"""Patch-similarity tracking: pick one patch, see where V-JEPA finds the same thing across the whole video."""

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

from vjepa.viz.pca import frame_maps
from vjepa.viz.render import to_uint8, upsample


def token_position(starts: np.ndarray, frame: int, window: int = 16, tubelet_size: int = 2) -> tuple[int, int]:
    """(window, time slice) whose tokens represent `frame`; later windows win, as in `frame_maps`."""
    w = max(i for i, s in enumerate(starts) if s <= frame < s + window)
    return w, (frame - starts[w]) // tubelet_size


def similarity_maps(
    tokens: np.ndarray, starts: np.ndarray, query_frame: int, query_yx: tuple[float, float], total_frames: int
) -> np.ndarray:
    """Cosine similarity of one query token to every token, laid out per frame: `(total_frames, gh, gw)`.

    `tokens` is `(W, slices, gh, gw, D)` from `extract_tokens`; `query_yx` is in [0, 1] of the model's square view.
    """
    gh, gw = tokens.shape[2:4]
    w, j = token_position(starts, query_frame)
    r, c = min(int(query_yx[0] * gh), gh - 1), min(int(query_yx[1] * gw), gw - 1)
    t = tokens.astype(np.float32)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    sims = t @ t[w, j, r, c]
    return frame_maps(starts, sims[..., None], total_frames)[..., 0]


def heat_overlay(
    view: np.ndarray, sim: np.ndarray, lo: float, hi: float, alpha: float = 0.75, cmap: str = "inferno"
) -> np.ndarray:
    """Blend a similarity map over a uint8 frame; tokens below `lo` stay transparent so the scene shows through."""
    s = np.clip((upsample(sim, view.shape[:2]) - lo) / (hi - lo + 1e-8), 0, 1)
    heat = matplotlib.colormaps[cmap](s)[..., :3]
    a = alpha * s[..., None]
    return to_uint8((1 - a) * view / 255.0 + a * heat)


def mark(view: np.ndarray, yx: tuple[float, float], radius: int = 10) -> np.ndarray:
    im = Image.fromarray(view)
    y, x = yx[0] * view.shape[0], yx[1] * view.shape[1]
    ImageDraw.Draw(im).ellipse([x - radius, y - radius, x + radius, y + radius], outline=(0, 255, 255), width=3)
    return np.asarray(im)
