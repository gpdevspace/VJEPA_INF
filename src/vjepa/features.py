"""Frozen V-JEPA features: target-encoder tokens for 16-frame windows of a clip."""

import numpy as np
import torch

from vjepa.device import autocast
from vjepa.models import VJEPA


def window_starts(total_frames: int, window: int, stride: int, cover_end: bool = False) -> list[int]:
    starts = list(range(0, total_frames - window + 1, stride))
    if not starts:
        raise ValueError(f"need at least {window} frames, got {total_frames}")
    if cover_end and starts[-1] + window < total_frames:
        starts.append(total_frames - window)
    return starts


@torch.no_grad()
def extract_tokens(
    model: VJEPA,
    clip: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    stride: int | None = None,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Target-encoder tokens for a preprocessed `(3, T, H, W)` clip.

    Windows are non-overlapping by default, plus one end-aligned window so every frame is covered.
    Returns `(starts, tokens)` with tokens shaped `(W, slices, grid_h, grid_w, D)` in float16.
    """
    spec = model.spec
    n = spec.num_frames
    starts = window_starts(clip.shape[1], n, stride or n, cover_end=True)
    gh, gw = clip.shape[2] // spec.patch_size, clip.shape[3] // spec.patch_size
    out = []
    for i in range(0, len(starts), batch_size):
        batch = torch.stack([clip[:, s : s + n] for s in starts[i : i + batch_size]]).to(device)
        with autocast(device, dtype):
            h = model.target_encoder(batch)
        out.append(h.float().view(len(batch), n // spec.tubelet_size, gh, gw, -1).cpu().numpy().astype(np.float16))
    return np.array(starts), np.concatenate(out)
