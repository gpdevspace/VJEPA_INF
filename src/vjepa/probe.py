"""Meta's released attentive probes (SSv2, K400) on top of the frozen V-JEPA target encoder.

Matches `evals/video_classification_frozen` in facebookresearch/jepa: 16-frame segments at frame_step 4, one per
equal partition of the video, tokens of all segments concatenated before the attentive pooler
(`attend_across_segments: true`), short side resized to 256 and center-cropped to 224.

Known issue (checked 2026-09-13): the released K400 probe does not work with the released vitl16 target encoder.
On clips of known classes the true label ranks ~100-170 of 400 at every frame step, on still frames too, while the
released IN1K probe on the same encoder and preprocessing is right (Grace Hopper -> military uniform, a Samoyed
photo -> Samoyed, a soccer clip -> soccer ball). Treat the SSv2 probe as unverified and train your own probe
for video tasks.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from vjepa.checkpoint import load_probe_state
from vjepa.device import autocast
from vjepa.models import VJEPA
from vjepa.third_party.jepa.models.attentive_pooler import AttentiveClassifier
from vjepa.video import preprocess

LABELS_DIR = Path("data/labels")


@dataclass(frozen=True)
class ProbeSpec:
    num_classes: int
    num_segments: int
    labels_file: str  # alphabetical class names (mmaction2 label maps), i.e. the official label-id order
    frame_step: int = 4


PROBES = {
    "ssv2": ProbeSpec(174, 2, "ssv2_labels.txt"),
    "k400": ProbeSpec(400, 8, "k400_labels.txt"),
}


def load_labels(task: str) -> list[str]:
    labels = (LABELS_DIR / PROBES[task].labels_file).read_text().splitlines()
    if len(labels) != PROBES[task].num_classes:
        raise ValueError(f"{task}: expected {PROBES[task].num_classes} labels, found {len(labels)}")
    return labels


def load_probe(model: VJEPA, task: str, device: torch.device) -> AttentiveClassifier:
    enc = model.target_encoder
    probe = AttentiveClassifier(
        embed_dim=enc.embed_dim, num_heads=enc.num_heads, depth=1, num_classes=PROBES[task].num_classes
    )
    probe.load_state_dict(load_probe_state(model.name, task), strict=True)
    return probe.eval().to(device)


def segment_indices(total: int, num_segments: int, frames_per_clip: int = 16, frame_step: int = 4) -> list[np.ndarray]:
    """One clip centered in each equal partition of the video; indices past either end are clamped (padding)."""
    clip_len = frames_per_clip * frame_step
    out = []
    for i in range(num_segments):
        start = int(round((i + 0.5) * total / num_segments - clip_len / 2))
        start = max(0, min(start, total - clip_len))
        out.append(np.clip(start + np.arange(frames_per_clip) * frame_step, 0, total - 1))
    return out


@torch.no_grad()
def classify(
    model: VJEPA,
    probe: AttentiveClassifier,
    frames: np.ndarray,
    task: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    topk: int = 5,
) -> list[tuple[str, float]]:
    """Top-k (label, probability) for a video given as all decoded frames `(T, H, W, 3)` at native fps."""
    spec = PROBES[task]
    tokens = []
    for idx in segment_indices(len(frames), spec.num_segments, model.spec.num_frames, spec.frame_step):
        clip = preprocess(frames[idx], crop="center", short_side=256)[None].to(device)
        with autocast(device, dtype):
            tokens.append(model.target_encoder(clip))
    with autocast(device, dtype):
        logits = probe(torch.cat(tokens, dim=1))
    probs = logits.float().softmax(-1)[0].cpu()
    top = probs.topk(topk)
    labels = load_labels(task)
    return [(labels[i], p.item()) for i, p in zip(top.indices, top.values)]
