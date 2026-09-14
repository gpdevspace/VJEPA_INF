"""PCA of V-JEPA patch tokens to RGB: the "what does the model see" visualization."""

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(tokens: np.ndarray, n_components: int = 3, max_samples: int = 50_000, seed: int = 0) -> PCA:
    """Fit PCA on (a random subset of) all tokens of a video, so colors are consistent across time."""
    flat = tokens.reshape(-1, tokens.shape[-1]).astype(np.float32)
    if len(flat) > max_samples:
        flat = flat[np.random.default_rng(seed).choice(len(flat), max_samples, replace=False)]
    return PCA(n_components, random_state=seed).fit(flat)


def tokens_to_rgb(
    tokens: np.ndarray, pca: PCA, bounds: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Project `(..., D)` tokens onto 3 components, scaled to [0, 1] per channel by 1st/99th percentiles."""
    proj = pca.transform(tokens.reshape(-1, tokens.shape[-1]).astype(np.float32))
    if bounds is None:
        bounds = (np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0))
    lo, hi = bounds
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(*tokens.shape[:-1], 3), bounds


def frame_maps(starts: np.ndarray, window_maps: np.ndarray, total_frames: int, tubelet_size: int = 2) -> np.ndarray:
    """Spread per-window `(W, slices, gh, gw, C)` maps onto frames: `(total_frames, gh, gw, C)`; later windows win."""
    out = np.zeros((total_frames, *window_maps.shape[2:]), dtype=np.float32)
    for s, window in zip(starts, window_maps):
        for j, slice_map in enumerate(window):
            out[s + j * tubelet_size : s + (j + 1) * tubelet_size] = slice_map
    return out
