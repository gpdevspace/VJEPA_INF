"""Low-level scene covariates for CUHK Avenue: motion energy, foreground area and blob count.

These are the confounds the surprise investigation has to rule out before any claim about "unpredictable new
information" can stand: fast motion is hard to predict whatever causes it (H3), and more people simply means
more hard tokens feeding a top-5% aggregate (H4).

Pure numpy + scipy. The camera is static, so a per-pixel temporal median over the whole video is a good enough
background model; nothing here needs OpenCV.
"""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
from scipy import ndimage

SCALE = 2  # 640x360 -> 320x180: enough to see a person, small enough to hold a whole video in memory
FG_THRESHOLD = 12.0  # grey levels above the background median before a pixel counts as foreground
MIN_BLOB_AREA = 80  # px at 320x180; below this is sensor noise and compression mosquito artifacts
MOTION_LAG = 4  # frames; matches the frame_step V-JEPA was run at, so both see the same temporal spacing


def read_gray(path: str | Path, scale: int = SCALE) -> np.ndarray:
    """Decode a whole video to `(T, H//scale, W//scale)` float32 grey, downsampling with a block mean.

    Frames are reduced during decode rather than after: a full-resolution RGB Avenue video is ~1 GB in memory.
    """
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            g = frame.to_ndarray(format="gray").astype(np.float32)
            h, w = (s // scale * scale for s in g.shape)
            frames.append(g[:h, :w].reshape(h // scale, scale, w // scale, scale).mean((1, 3)))
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames)


def background(gray: np.ndarray) -> np.ndarray:
    """Per-pixel temporal median: the static scene with every transient pedestrian removed."""
    return np.median(gray, axis=0)


def foreground_masks(
    gray: np.ndarray,
    bg: np.ndarray | None = None,
    threshold: float = FG_THRESHOLD,
    min_area: int = MIN_BLOB_AREA,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield `(mask, labels, n_blobs)` per frame, keeping only connected components above `min_area`.

    Generated rather than returned as one array so that callers never hold T full-size boolean masks at once.
    """
    bg = background(gray) if bg is None else bg
    structure = np.ones((3, 3), bool)
    for frame in gray:
        raw = np.abs(frame - bg) > threshold
        opened = ndimage.binary_opening(raw, structure=structure)
        labels, n = ndimage.label(opened, structure=structure)
        if n:
            areas = np.bincount(labels.ravel())
            areas[0] = 0
            keep = areas >= min_area
            labels = np.where(keep[labels], labels, 0)
            n = int(keep.sum())
        yield labels > 0, labels, n


def covariates(
    gray: np.ndarray,
    threshold: float = FG_THRESHOLD,
    min_area: int = MIN_BLOB_AREA,
    lag: int = MOTION_LAG,
) -> dict[str, np.ndarray]:
    """Per-frame scene covariates, each of length `len(gray)`.

    - `motion_energy`: mean absolute difference against the frame `lag` back, edge-padded at the start.
    - `foreground_area`: fraction of pixels that differ from the background model.
    - `blob_count`: number of foreground components large enough to be a person or object.
    """
    diff = np.abs(gray[lag:] - gray[:-lag]).mean(axis=(1, 2))
    motion = np.concatenate([np.repeat(diff[:1], lag), diff])  # edge-pad the frames with no predecessor
    bg = background(gray)
    area, blobs = np.empty(len(gray)), np.empty(len(gray))
    for i, (mask, _, n) in enumerate(foreground_masks(gray, bg, threshold, min_area)):
        area[i], blobs[i] = mask.mean(), n
    return {"motion_energy": motion, "foreground_area": area, "blob_count": blobs}
