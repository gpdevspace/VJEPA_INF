"""The vendored model code must match facebookresearch/jepa exactly (set JEPA_SRC to a clone to run)."""

import contextlib
import importlib
import os

import pytest
import torch

from vjepa.third_party.jepa.models import predictor as vendored_pred
from vjepa.third_party.jepa.models import vision_transformer as vendored_vit

JEPA_SRC = os.environ.get("JEPA_SRC")
pytestmark = pytest.mark.skipif(not JEPA_SRC, reason="set JEPA_SRC to a facebookresearch/jepa clone")

KW = dict(img_size=64, patch_size=16, num_frames=4, tubelet_size=2, uniform_power=True)


@pytest.fixture
def original(monkeypatch):
    monkeypatch.syspath_prepend(JEPA_SRC)
    # Newer torch may drop this CUDA-only context manager; it is a no-op on CPU anyway.
    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", lambda *a, **k: contextlib.nullcontext(), raising=False)
    return importlib.import_module("src.models.vision_transformer"), importlib.import_module("src.models.predictor")


def test_encoder_and_predictor_match_original(original):
    orig_vit, orig_pred = original
    torch.manual_seed(0)
    enc_a = orig_vit.vit_small(**KW).eval()
    enc_b = vendored_vit.vit_small(**KW).eval()
    enc_b.load_state_dict(enc_a.state_dict())

    pred_kw = dict(
        **KW, embed_dim=384, predictor_embed_dim=96, depth=2, num_heads=6, use_mask_tokens=True, num_mask_tokens=2
    )
    pred_a = orig_pred.vit_predictor(**pred_kw).eval()
    torch.nn.init.normal_(pred_a.mask_tokens[0])  # zero-init tokens would hide a mask_index mix-up
    pred_b = vendored_pred.vit_predictor(**pred_kw).eval()
    pred_b.load_state_dict(pred_a.state_dict())

    x = torch.randn(2, 3, 4, 64, 64)
    ctx, tgt = torch.arange(16).repeat(2, 1), torch.arange(16, 32).repeat(2, 1)  # slice 0 predicts slice 1
    with torch.no_grad():
        assert torch.allclose(enc_a(x), enc_b(x), atol=1e-5)
        za, zb = enc_a(x, masks=[ctx]), enc_b(x, masks=[ctx])
        assert torch.allclose(za, zb, atol=1e-5)
        pa = pred_a(za, None, [ctx], [tgt], mask_index=0)
        pb = pred_b(zb, None, [ctx], [tgt], mask_index=0)
        assert torch.allclose(pa, pb, atol=1e-5)
