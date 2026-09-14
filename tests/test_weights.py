"""Checks against the real ViT-L/16 weights (skipped until `vjepa download` + `vjepa convert` have run)."""

import pytest
import torch
import torch.nn.functional as F

from vjepa.checkpoint import CHECKPOINT_DIR
from vjepa.models import load_vjepa
from vjepa.surprise import SurpriseMeter, time_masks

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "vitl16" / "predictor.safetensors").exists(), reason="converted vitl16 weights missing"
)


def smooth_clip(batch: int = 1) -> torch.Tensor:
    """A deterministic moving-gradient clip: closer to natural video statistics than white noise."""
    t = torch.arange(16).view(1, 1, 16, 1, 1).float()
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 224), torch.linspace(-1, 1, 224), indexing="ij")
    phase = torch.tensor([0.0, 2.1, 4.2]).view(1, 3, 1, 1, 1)
    return torch.sin(3 * xx + 2 * yy + 0.3 * t + phase).repeat(batch, 1, 1, 1, 1)


@pytest.fixture(scope="module")
def cpu_model():
    return load_vjepa("vitl16", "cpu")


def test_predictor_output_covers_every_target_token(cpu_model):
    meter = SurpriseMeter(cpu_model, torch.device("cpu"), torch.float32)
    scores, maps = meter.score_windows(smooth_clip(2), contexts=(4,))
    _, tgt = time_masks(4)
    assert scores.shape == (1, 2)
    assert maps[0].shape == (2, len(tgt) // 196, 14, 14)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple Silicon")
def test_mps_matches_cpu_fp32(cpu_model):
    """fp32 on MPS must equal CPU; fp16 (what the reference evals use) may only perturb a few tokens slightly."""
    x = smooth_clip()
    mps_model = load_vjepa("vitl16", "mps", with_predictor=False)
    with torch.no_grad():
        ref = cpu_model.target_encoder(x)
        fp32 = mps_model.target_encoder(x.to("mps")).cpu()
        with torch.autocast("mps", dtype=torch.float16):
            fp16 = mps_model.target_encoder(x.to("mps")).float().cpu()
    assert F.cosine_similarity(ref, fp32, dim=-1).min() > 0.9999
    cos16 = F.cosine_similarity(ref, fp16, dim=-1)
    assert cos16.mean() > 0.999 and cos16.quantile(0.01) > 0.99
