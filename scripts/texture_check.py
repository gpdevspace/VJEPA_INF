"""Does border error track local TEXTURE, or just distance from the frame edge regardless of content?

Challenged by the user (2026-09-19): "textured regions are hard to predict" was asserted, not tested, and it
sits awkwardly with JEPA's actual design goal -- predicting in a representation space that's supposed to
discard exactly this kind of irrelevant fine-grained detail (Garrido et al. 2025). If border tokens are hard
regardless of what's actually there, that's not "expected texture cost", it could be something else: a Vision
Transformer edge effect (border patches have less surrounding context), independent of content.

Test: build a texture-energy map of each video's static background (high-pass filtered, pooled onto the same
14x14 grid as the tokens), and check whether it predicts the background-only error map from center_bias.py
better than radial distance from the frame centre does.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

from vjepa.data.avenue import test_videos
from vjepa.eval.scene_events import background, read_gray

GRID = 14
OUT = Path("outputs/investigations/texture_check.json")


def texture_energy(bg: np.ndarray) -> np.ndarray:
    """High-pass residual (bg minus a heavy blur of itself), pooled to 14x14: how much fine detail sits here."""
    hf = np.abs(bg - gaussian_filter(bg, sigma=4))
    return np.asarray(Image.fromarray(hf).resize((GRID, GRID), Image.BOX))


def radial_distance() -> np.ndarray:
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    return np.sqrt((yy - (GRID - 1) / 2) ** 2 + (xx - (GRID - 1) / 2) ** 2)


def main() -> None:
    maps = []
    for path in test_videos():
        bg = background(read_gray(path))
        maps.append(texture_energy(bg))
        print(f"  {path.stem}: texture range [{maps[-1].min():.1f}, {maps[-1].max():.1f}]")
    texture = np.mean(maps, axis=0)

    err = np.array(json.load(open("outputs/investigations/center_bias.json"))["error_background_tokens_only"])
    dist = radial_distance()

    t, e, d = texture.ravel(), err.ravel(), dist.ravel()
    r_texture, _ = pearsonr(t, e)
    rho_texture, _ = spearmanr(t, e)
    r_dist, _ = pearsonr(d, e)

    # partial correlation: does texture explain error left over after removing the radial trend?
    A = np.vstack([d, np.ones_like(d)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid_e = e - A @ coef
    coef_t, *_ = np.linalg.lstsq(A, t, rcond=None)
    resid_t = t - A @ coef_t
    r_partial, _ = pearsonr(resid_t, resid_e)

    print(f"\ncorrelation(texture, background error)      : Pearson r={r_texture:+.3f}  Spearman rho={rho_texture:+.3f}")
    print(f"correlation(radial distance, background error): Pearson r={r_dist:+.3f}")
    print(f"partial correlation(texture, error | distance removed): r={r_partial:+.3f}")

    verdict = (
        "TEXTURE EXPLAINS IT: still correlated once distance-from-centre is regressed out"
        if abs(r_partial) > 0.3
        else "TEXTURE DOES NOT EXPLAIN IT: the border effect is not about local visual detail -- "
        "likely a positional/architectural effect (e.g. a ViT edge artifact), not texture"
    )
    print(f"\n{verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "texture_map": texture.tolist(), "error_map": err.tolist(), "radial_distance_map": dist.tolist(),
        "pearson_r_texture_vs_error": float(r_texture), "spearman_rho_texture_vs_error": float(rho_texture),
        "pearson_r_distance_vs_error": float(r_dist), "partial_r_texture_given_distance": float(r_partial),
        "verdict": verdict,
    }, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
