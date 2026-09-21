"""Where does the predictor actually beat a scene template, and why is copying the previous frame so bad?

floor_template.py left two things unexplained. Held-out, a per-(slice, position) mean token scores 0.534 on the
mean aggregate against the predictor's 0.554 -- the template wins -- yet on the top-5% aggregate the predictor
wins (0.682 vs 0.721). And persistence (0.774) is worse than predicting zeros (0.676), which a slice-offset
correction did not fix.

Two measurements, on the cached tensors:

1. similarity structure -- cosine between the same grid position at different tubelet slices of the SAME window,
   against the same slice in a DIFFERENT window. If adjacent slices are no more similar than unrelated windows,
   the target encoder's tokens are not a frame-by-frame picture, and copying one forward is meaningless.

2. where the predictor's edge lives -- rank tokens by how badly the scene template does on them, and compare the
   predictor with the template within each decile. The Surprise Meter aggregates the top 5%, so an edge that is
   concentrated in the worst tokens is the part that the published AUC actually uses.
"""

import json
from pathlib import Path

import numpy as np
import torch

CACHE = Path("/private/tmp/claude-501/-Users-gpmac-gpbuildspace-VJEPA/c2fbbf21-ae30-423e-b300-e6c206128439/scratchpad/floor_tensors.pt")
OUT = Path("outputs/investigations/floor_structure.json")
GRID = 14 * 14


def main() -> None:
    d = torch.load(CACHE)
    tgt, pred, last, vid = d["tgt"], d["pred"], d["last"], d["vid"]
    B, N, D = tgt.shape
    S = N // GRID
    t4 = tgt.reshape(B, S, GRID, D)
    res = {}

    # --- 1. how similar are tokens across slices vs across windows? ---
    cos = torch.nn.functional.cosine_similarity
    same_win = {}
    for lag in range(1, S):
        a, b = t4[:, : S - lag], t4[:, lag:]
        same_win[f"lag{lag}"] = float(cos(a, b, dim=-1).mean())
    # last context slice -> each target slice (what persistence relies on)
    to_last = {f"slice{s}": float(cos(t4[:, s], last, dim=-1).mean()) for s in range(S)}
    perm = torch.randperm(B)
    while (perm == torch.arange(B)).any():
        perm = torch.randperm(B)
    diff_win = float(cos(t4, t4[perm], dim=-1).mean())          # same slice+position, different window
    diff_pos = float(cos(t4, t4[:, :, torch.randperm(GRID)], dim=-1).mean())  # same window+slice, other position
    res["cosine"] = {"same_window_across_slices": same_win, "last_context_to_target": to_last,
                     "different_window_same_slice_pos": diff_win, "same_window_shuffled_position": diff_pos}

    print("cosine similarity of layer-normed target tokens")
    print(f"  same window, adjacent slices:  " + "  ".join(f"{k}={v:+.3f}" for k, v in same_win.items()))
    print(f"  last context slice -> target:  " + "  ".join(f"{k}={v:+.3f}" for k, v in to_last.items()))
    print(f"  different window, same slice+position: {diff_win:+.3f}")
    print(f"  same window+slice, shuffled position:  {diff_pos:+.3f}")

    # --- 2. where does the predictor beat the template? held-out split, as before ---
    uniq = vid.unique()
    fit_m = torch.isin(vid, uniq[: len(uniq) // 2])
    ev_m = ~fit_m
    tmpl = t4[fit_m].mean(0)
    e_t = (t4[ev_m] - tmpl).abs().mean(-1).reshape(-1)                  # template error per token
    e_p = (tgt[ev_m] - pred[ev_m]).abs().mean(-1).reshape(-1)           # predictor error per token
    order = torch.argsort(e_t)
    deciles = []
    print(f"\n{'decile by template error':>26}{'template':>10}{'predictor':>11}{'edge':>8}")
    for i in range(10):
        idx = order[i * len(order) // 10 : (i + 1) * len(order) // 10]
        t_, p_ = float(e_t[idx].mean()), float(e_p[idx].mean())
        deciles.append({"decile": i + 1, "template": t_, "predictor": p_, "edge": t_ - p_})
        print(f"{f'{i+1} ({"easiest" if i==0 else "hardest" if i==9 else ""})':>26}{t_:>10.4f}{p_:>11.4f}{t_-p_:>+8.4f}")
    res["deciles_heldout"] = deciles

    k = max(1, int(0.05 * len(e_t)))
    top_t = torch.topk(e_t, k).values.mean()
    top_p = torch.topk(e_p, k).values.mean()
    res["top5pct_pooled"] = {"template": float(top_t), "predictor": float(top_p)}
    res["overall_heldout"] = {"template": float(e_t.mean()), "predictor": float(e_p.mean())}

    print(f"\nheld-out overall: template {e_t.mean():.4f}  predictor {e_p.mean():.4f}")
    print(f"held-out top 5% of tokens: template {top_t:.4f}  predictor {top_p:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()


def outlier_dims() -> None:
    """Is the L1 error dominated by a few consistently large feature dimensions?

    Kurtosis 27.7 on the layer-normed targets (floor_mechanism.py) suggests a handful of outlier dims, which are
    well documented in large transformers. If they are the same dims every time, the "floor" is mostly the
    predictor failing to match a few channels, not a diffuse failure across the representation.
    """
    d = torch.load(CACHE)
    tgt, pred = d["tgt"], d["pred"]
    err = (tgt - pred).abs()                      # (B, N, D)
    per_dim_err = err.mean((0, 1))                # (D,)
    per_dim_mag = tgt.abs().mean((0, 1))
    order = torch.argsort(per_dim_err, descending=True)
    D = tgt.shape[-1]
    shares = {f"top{n}": float(per_dim_err[order[:n]].sum() / per_dim_err.sum()) for n in (1, 5, 10, 50, 100)}
    print("\nshare of total L1 error carried by the worst dimensions "
          f"(of {D}): " + "  ".join(f"{k}={v:.1%}" for k, v in shares.items()))
    print(f"  worst dim: error {per_dim_err[order[0]]:.3f}, target magnitude {per_dim_mag[order[0]]:.3f}  "
          f"(median dim: error {per_dim_err.median():.3f}, magnitude {per_dim_mag.median():.3f})")
    # are the big-error dims the same as the big-magnitude dims?
    rho = float(torch.corrcoef(torch.stack([per_dim_err, per_dim_mag]))[0, 1])
    print(f"  correlation between a dim's mean |target| and its mean error: {rho:+.3f}")
    frac_uniform = 1 / D
    print(f"  uniform share would be top10={10*frac_uniform:.1%}, top100={100*frac_uniform:.1%}")
    res = json.loads(OUT.read_text())
    res["outlier_dims"] = {"error_share": shares, "corr_magnitude_vs_error": rho}
    OUT.write_text(json.dumps(res, indent=2))


outlier_dims()
