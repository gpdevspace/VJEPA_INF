"""Build V-JEPA 1 networks from the vendored Meta code and load the released weights."""

from dataclasses import dataclass

import torch
from torch import nn

from vjepa.checkpoint import load_part
from vjepa.third_party.jepa.models import vision_transformer as vit
from vjepa.third_party.jepa.models.predictor import vit_predictor


@dataclass(frozen=True)
class ModelSpec:
    arch: str
    img_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    tubelet_size: int = 2
    pred_depth: int = 12
    pred_embed_dim: int = 384
    uniform_power: bool = True
    num_mask_tokens: int = 2  # one per pretraining mask type (short- and long-range blocks)


# From facebookresearch/jepa configs/pretrain/<model>.yaml
SPECS = {
    "vitl16": ModelSpec("vit_large"),
    "vith16": ModelSpec("vit_huge"),
    "vith16-384": ModelSpec("vit_huge", img_size=384),
}


def build_encoder(spec: ModelSpec) -> nn.Module:
    return vit.__dict__[spec.arch](
        img_size=spec.img_size,
        patch_size=spec.patch_size,
        num_frames=spec.num_frames,
        tubelet_size=spec.tubelet_size,
        uniform_power=spec.uniform_power,
    )


def build_predictor(spec: ModelSpec, embed_dim: int, num_heads: int) -> nn.Module:
    return vit_predictor(
        img_size=spec.img_size,
        patch_size=spec.patch_size,
        num_frames=spec.num_frames,
        tubelet_size=spec.tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=spec.pred_embed_dim,
        depth=spec.pred_depth,
        num_heads=num_heads,
        uniform_power=spec.uniform_power,
        use_mask_tokens=True,
        num_mask_tokens=spec.num_mask_tokens,
        zero_init_mask_tokens=True,
    )


@dataclass
class VJEPA:
    name: str
    spec: ModelSpec
    encoder: nn.Module | None  # context encoder (sees only unmasked tokens)
    target_encoder: nn.Module  # EMA encoder; Meta's frozen evals use this one for features
    predictor: nn.Module | None


def _frozen(module: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> nn.Module:
    module.load_state_dict(state, strict=True)
    module.requires_grad_(False)
    return module.eval().to(device)


def load_vjepa(model: str = "vitl16", device: torch.device | str = "cpu", with_predictor: bool = True) -> VJEPA:
    """Load the frozen target encoder, plus the context encoder and predictor when `with_predictor`."""
    spec = SPECS[model]
    device = torch.device(device)
    target = _frozen(build_encoder(spec), load_part(model, "target_encoder"), device)
    encoder = predictor = None
    if with_predictor:
        encoder = _frozen(build_encoder(spec), load_part(model, "encoder"), device)
        predictor = build_predictor(spec, embed_dim=target.embed_dim, num_heads=target.num_heads)
        predictor = _frozen(predictor, load_part(model, "predictor"), device)
    return VJEPA(model, spec, encoder, target, predictor)
