"""Frame compositing helpers and H.264 MP4 encoding through ffmpeg."""

import itertools
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def upsample(small: np.ndarray, size: tuple[int, int], smooth: bool = True) -> np.ndarray:
    """Resize a `(gh, gw)` or `(gh, gw, C)` float map to `(h, w[, C])` for overlaying on frames."""
    t = torch.from_numpy(np.ascontiguousarray(small, dtype=np.float32))
    chw = t.permute(2, 0, 1)[None] if t.ndim == 3 else t[None, None]
    out = F.interpolate(chw, size=size, mode="bicubic" if smooth else "nearest")[0]
    return (out.permute(1, 2, 0) if t.ndim == 3 else out[0]).numpy()


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def outline_mask(view: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (255, 60, 60)) -> np.ndarray:
    """Draw the boundary of a boolean mask onto a uint8 frame, leaving the interior untouched."""
    from scipy import ndimage

    if not mask.any():
        return view
    edge = ndimage.binary_dilation(mask, iterations=2) & ~ndimage.binary_erosion(mask, iterations=1)
    out = view.copy()
    out[edge] = colour
    return out


def write_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: float, crf: int = 18) -> Path:
    """Encode same-sized RGB uint8 frames as H.264/yuv420p MP4 (plays on LinkedIn, QuickTime, browsers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[0] // 2 * 2, first.shape[1] // 2 * 2  # yuv420p needs even dimensions
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]  # fmt: skip
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for frame in itertools.chain([first], frames):
            proc.stdin.write(np.ascontiguousarray(frame[:h, :w]).tobytes())
        proc.stdin.close()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg exited with {proc.returncode} while writing {path}")
    return path
