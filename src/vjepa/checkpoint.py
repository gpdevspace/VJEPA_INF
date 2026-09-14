"""Download Meta's released V-JEPA 1 checkpoints and split them into slim safetensors files."""

import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

CHECKPOINT_DIR = Path("checkpoints")
BASE_URL = "https://dl.fbaipublicfiles.com/jepa"
PARTS = ("encoder", "target_encoder", "predictor")


def pretrain_url(model: str) -> str:
    return f"{BASE_URL}/{model}/{model}.pth.tar"


def probe_url(model: str, task: str) -> str:
    return f"{BASE_URL}/{model}/{task}-probe.pth.tar"


def download(url: str, dest: Path) -> Path:
    """Download `url` to `dest`, resuming a partial file if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have >= total:
        return dest
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(request) as r:
        resumed = r.status == 206  # the server may ignore Range; then start over
        with (
            open(dest, "ab" if resumed else "wb") as f,
            tqdm(total=total, initial=have if resumed else 0, unit="B", unit_scale=True, desc=dest.name) as bar,
        ):
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def convert(model: str, root: Path = CHECKPOINT_DIR) -> dict:
    """Split `<model>.pth.tar` into encoder / target_encoder / predictor safetensors.

    Drops the optimizer state (5.1 GB -> ~2.5 GB for ViT-L) and keeps fp32 weights so outputs match the original
    bit-for-bit; cast at load time instead. Returns the training metadata stored alongside the weights.
    """
    ckpt = torch.load(root / model / f"{model}.pth.tar", map_location="cpu", mmap=True, weights_only=True)
    for part in PARTS:
        state = {k.removeprefix("module.backbone."): v.contiguous() for k, v in ckpt[part].items()}
        save_file(state, root / model / f"{part}.safetensors")
    return {k: ckpt[k] for k in ("epoch", "loss", "batch_size", "lr") if k in ckpt}


def load_part(model: str, part: str, root: Path = CHECKPOINT_DIR) -> dict[str, torch.Tensor]:
    path = root / model / f"{part}.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `uv run vjepa download {model}` and `uv run vjepa convert {model}`")
    return load_file(path)


def load_probe_state(model: str, task: str, root: Path = CHECKPOINT_DIR) -> dict[str, torch.Tensor]:
    """Attentive-probe weights (`AttentiveClassifier` keys) from a released `<task>-probe.pth.tar`."""
    ckpt = torch.load(root / model / f"{task}-probe.pth.tar", map_location="cpu", weights_only=True)
    return {k.removeprefix("module."): v for k, v in ckpt["classifier"].items()}
