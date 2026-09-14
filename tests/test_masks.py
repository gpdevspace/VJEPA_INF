import pytest
import torch

from vjepa.surprise import time_masks
from vjepa.third_party.jepa.models.utils.patch_embed import PatchEmbed3D


def test_time_masks_split_index_range():
    ctx, tgt = time_masks(4)
    assert len(ctx) == 2 * 196 and len(tgt) == 6 * 196
    assert torch.equal(torch.cat([ctx, tgt]), torch.arange(8 * 196))


def test_time_masks_rejects_odd_or_out_of_range_context():
    for bad in (0, 3, 16):
        with pytest.raises(ValueError):
            time_masks(bad)


def test_token_order_is_time_then_row_then_col():
    torch.manual_seed(0)
    embed = PatchEmbed3D(patch_size=16, tubelet_size=2, in_chans=3, embed_dim=8)
    torch.nn.init.zeros_(embed.proj.bias)
    x = torch.zeros(1, 3, 16, 224, 224)
    x[:, :, 4:6, 32:48, 64:80] = 1.0  # tubelet slice 2, patch row 2, patch col 4
    active = embed(x)[0].abs().sum(-1).nonzero().flatten().tolist()
    assert active == [2 * 196 + 2 * 14 + 4]
