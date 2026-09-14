"""Surprise Meter: V-JEPA's prediction error on the future, following Garrido et al. (2025).

For each 16-frame window, the context encoder sees only the first C frames' tokens, the predictor fills in the
remaining frames in representation space, and surprise is the L1 distance to what the target encoder computed
for those frames from the real video. Mirrors `evals/intuitive_physics/eval.py` in
facebookresearch/jepa-intuitive-physics (target layer-norm, time masks, single-mask predictor call).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from vjepa.device import autocast
from vjepa.models import VJEPA
from vjepa.third_party.jepa.masks.utils import apply_masks

DEFAULT_CONTEXTS = (2, 4, 6, 8, 10)


def time_masks(
    context_frames: int, num_frames: int = 16, tubelet_size: int = 2, grid_h: int = 14, grid_w: int = 14
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token indices of the first `context_frames` frames (context) and of the remaining frames (target).

    Tokens are ordered (time slice, row, col), so splitting in time is splitting the index range.
    """
    if context_frames % tubelet_size or not 0 < context_frames < num_frames:
        raise ValueError(f"context_frames must be a multiple of {tubelet_size} in (0, {num_frames})")
    per_slice = grid_h * grid_w
    n_ctx = per_slice * context_frames // tubelet_size
    idx = torch.arange(per_slice * num_frames // tubelet_size)
    return idx[:n_ctx], idx[n_ctx:]


def normalize_over_time(maps: np.ndarray, skip: int = 0) -> np.ndarray:
    """Per-location z-score over time: how unusual each patch's error is relative to that patch's own history.

    Textured or flickering regions are always hard to predict; this suppresses them so events stand out.
    The first `skip` frames (not yet predicted) are left out of the statistics.
    """
    ref = maps[skip:]
    return (maps - np.median(ref, axis=0, keepdims=True)) / (ref.std(axis=0, keepdims=True) + 1e-6)


@dataclass
class SurpriseResult:
    starts: np.ndarray  # (W,) first frame of each window, in sampled-frame units
    contexts: tuple[int, ...]
    scores: np.ndarray  # (len(contexts), W) mean L1 error per window
    token_errors: list[np.ndarray]  # per context: (W, target_slices, grid_h, grid_w)
    num_frames: int = 16
    tubelet_size: int = 2

    def frame_curve(self, context_index: int, total_frames: int) -> np.ndarray:
        """Per-frame surprise: mean score of the windows whose predicted span covers each frame (NaN if none)."""
        c = self.contexts[context_index]
        acc, cnt = np.zeros(total_frames), np.zeros(total_frames)
        for s, v in zip(self.starts, self.scores[context_index]):
            acc[s + c : s + self.num_frames] += v
            cnt[s + c : s + self.num_frames] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return acc / cnt

    def frame_heatmaps(self, context_index: int, total_frames: int) -> np.ndarray:
        """`(total_frames, grid_h, grid_w)` token-error maps averaged over the windows that predict each frame."""
        c = self.contexts[context_index]
        errs = self.token_errors[context_index]
        acc, cnt = np.zeros((total_frames, *errs.shape[-2:])), np.zeros(total_frames)
        for s, err in zip(self.starts, errs):
            for j, err_slice in enumerate(err):
                f0 = s + c + j * self.tubelet_size
                acc[f0 : f0 + self.tubelet_size] += err_slice
                cnt[f0 : f0 + self.tubelet_size] += 1
        return acc / np.maximum(cnt, 1)[:, None, None]


class SurpriseMeter:
    def __init__(self, model: VJEPA, device: torch.device, dtype: torch.dtype = torch.float16, mask_index: int = 0):
        if model.encoder is None or model.predictor is None:
            raise ValueError("SurpriseMeter needs the context encoder and predictor (load_vjepa(with_predictor=True))")
        self.model, self.device, self.dtype = model, device, dtype
        # The reference wraps the predictor in PredictorMultiMaskWrapper, which passes mask_index=0 for a single mask.
        self.mask_index = mask_index

    @torch.no_grad()
    def score_windows(
        self, x: torch.Tensor, contexts: Sequence[int] = DEFAULT_CONTEXTS, context_x: torch.Tensor | None = None
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Score a batch of `(B, 3, T, H, W)` windows.

        Returns per-window scores `(len(contexts), B)` and, per context length, token errors `(B, S, grid_h, grid_w)`.
        `context_x`, if given, supplies the context from different clips (a sanity check: it should be more surprising).
        """
        spec = self.model.spec
        B, _, T, H, W = x.shape
        gh, gw = H // spec.patch_size, W // spec.patch_size
        x = x.to(self.device)
        cx = x if context_x is None else context_x.to(self.device)
        scores, maps = [], []
        with autocast(self.device, self.dtype):
            h = self.model.target_encoder(x)
            h = F.layer_norm(h, (h.size(-1),))
            for c in contexts:
                ctx, tgt = (m.to(self.device).repeat(B, 1) for m in time_masks(c, T, spec.tubelet_size, gh, gw))
                z = self.model.encoder(cx, masks=[ctx])
                target = apply_masks(h, [tgt])
                pred = self.model.predictor(z, target, [ctx], [tgt], mask_index=self.mask_index)
                err = (pred.float() - target.float()).abs().mean(-1)  # (B, N_tgt)
                scores.append(err.mean(-1))
                maps.append(err.view(B, -1, gh, gw))
        # One device sync per batch rather than per context length.
        return torch.stack(scores).cpu().numpy(), [m.cpu().numpy() for m in maps]

    def score_clip(
        self,
        clip: torch.Tensor,
        contexts: Sequence[int] = DEFAULT_CONTEXTS,
        stride: int = 2,
        batch_size: int = 4,
        progress: Callable[[int, int], None] | None = None,
    ) -> SurpriseResult:
        """Slide a window over a preprocessed `(3, T, H, W)` clip and score every window."""
        n = self.model.spec.num_frames
        starts = list(range(0, clip.shape[1] - n + 1, stride))
        if not starts:
            raise ValueError(f"need at least {n} frames, got {clip.shape[1]}")
        all_scores, all_maps = [], [[] for _ in contexts]
        for i in range(0, len(starts), batch_size):
            batch = torch.stack([clip[:, s : s + n] for s in starts[i : i + batch_size]])
            scores, maps = self.score_windows(batch, contexts)
            all_scores.append(scores)
            for k, m in enumerate(maps):
                all_maps[k].append(m)
            if progress:
                progress(min(i + batch_size, len(starts)), len(starts))
        return SurpriseResult(
            starts=np.array(starts),
            contexts=tuple(contexts),
            scores=np.concatenate(all_scores, axis=1),
            token_errors=[np.concatenate(m) for m in all_maps],
            num_frames=n,
            tubelet_size=self.model.spec.tubelet_size,
        )
