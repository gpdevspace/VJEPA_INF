"""Video decoding and preprocessing that matches V-JEPA's evaluation transforms."""

import re
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def read_video(path: str | Path, frame_step: int = 1, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    """Decode every `frame_step`-th frame as RGB uint8 `(T, H, W, 3)`; also returns the source fps."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i % frame_step:
                continue
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), fps


def _natural_key(path: Path) -> list:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path.name)]


def read_frame_dir(path: str | Path, frame_step: int = 1) -> np.ndarray:
    """Read an image-sequence folder (e.g. IntPhys `scene/`) in natural order as `(T, H, W, 3)` uint8."""
    files = sorted((p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
    return np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files[::frame_step]])


def display_frame(frame: np.ndarray, size: int, crop: str = "square") -> np.ndarray:
    """The square region the model sees (same geometry as `preprocess`), resized to `size` px for rendering."""
    if crop == "center":
        h, w = frame.shape[:2]
        s = min(h, w)
        top, left = (h - s) // 2, (w - s) // 2
        frame = frame[top : top + s, left : left + s]
    return np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.LANCZOS))


def preprocess(
    frames: np.ndarray, size: int = 224, crop: str = "square", short_side: int | None = None, chunk: int = 64
) -> torch.Tensor:
    """uint8 `(T, H, W, 3)` -> ImageNet-normalized float `(3, T, size, size)`.

    crop="square" resizes the whole frame (nothing cut off, aspect squashed), which is what the intuitive-physics
    evals do. crop="center" resizes the short side to `short_side` (default `size`) and center-crops; Meta's
    video-classification eval uses short_side=256 for 224 crops.
    """
    out = []
    for i in range(0, len(frames), chunk):  # chunked so long HD videos don't blow up memory
        batch = np.ascontiguousarray(frames[i : i + chunk])  # also accepts reversed (negative-stride) arrays
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        h, w = x.shape[-2:]
        if crop == "center":
            scale = (short_side or size) / min(h, w)
            x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="bilinear", antialias=True)
            top, left = (x.shape[-2] - size) // 2, (x.shape[-1] - size) // 2
            x = x[..., top : top + size, left : left + size]
        elif crop == "square":
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True)
        else:
            raise ValueError(f"unknown crop mode {crop!r}")
        out.append(x)
    x = torch.cat(out).permute(1, 0, 2, 3)  # (3, T, H, W)
    return (x - IMAGENET_MEAN) / IMAGENET_STD
