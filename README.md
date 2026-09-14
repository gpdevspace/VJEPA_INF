# V-JEPA 1, hands-on

Experiments with Meta's [V-JEPA 1](https://arxiv.org/abs/2404.08471) (ViT-L/16) on an Apple M5 laptop. The goal is
to build a **Surprise Meter**: V-JEPA's predictor guesses what the next frames look like *in representation
space*, and the error of that guess is used as a zero-shot signal for impossible or anomalous events.

Status: work in progress. Everything below was measured in this repo; open questions are marked as such.

## Setup

```bash
uv sync
uv run vjepa download vitl16 --probe ssv2 k400   # 5.1 GB checkpoint + attentive probes (resumable)
uv run vjepa convert vitl16                      # -> checkpoints/vitl16/{encoder,target_encoder,predictor}.safetensors
uv run pytest -q                                 # set JEPA_SRC=<facebookresearch/jepa clone> to also run the port test
```

## Commands

| command | what it does |
|---|---|
| `vjepa bench` | encoder and surprise throughput per precision and batch size |
| `vjepa surprise VIDEO` | surprise curve (5 context lengths) for a video file or frame folder |
| `vjepa track VIDEO --x --y --at-frame` | pick a patch, highlight where V-JEPA's tokens are most similar over time |
| `vjepa pca VIDEO` | PCA-of-features video (kept for reference; see findings) |
| `vjepa classify VIDEO --task k400` | Meta's released attentive probes (see findings before trusting it) |
| `vjepa eval-intphys` | IntPhys 2019 dev set: relative accuracy with a tune/held-out scene split |
| `vjepa eval-avenue` | CUHK Avenue: zero-shot frame-level anomaly AUC |

## What's verified

- **Port.** The encoder, predictor and probe code are vendored from `facebookresearch/jepa`, with one patch (the
  CUDA-only `sdp_kernel` context removed). Outputs match the original code to 1e-5, and MPS fp32 matches CPU fp32 exactly.
- **Precision.** fp16 on MPS matches fp32 closely (per-token cosine: mean 0.9998+, 1st percentile ≥ 0.998). bf16 on
  MPS is noticeably worse (some tokens drop to cosine 0.42), so everything here runs in fp16.
- **Speed (M5, fp16).** 7.3 clips/s for the ViT-L encoder (16 frames at 224 px); one surprise window with 5
  context lengths takes ~1.2 s.
- **Predictor wiring.** Predicting a clip's future from its own past gives L1 0.563–0.569, next to the
  checkpoint's final training loss of 0.558. Taking the past from a different clip raises it to 0.659–0.667.

## Findings so far

- **Meta's released K400 video probe doesn't work with the released `vitl16` weights.** On clips of known classes,
  the true label ranks 100–173 of 400 at every frame step, and on still frames too. The released ImageNet probe on
  the same encoder and preprocessing is correct: Grace Hopper → military uniform (62%), a Samoyed photo → Samoyed (96%),
  a soccer video → soccer ball (74%). So the pipeline is fine and the video probe checkpoints are the problem.
- **PCA-of-features videos are illegible for V-JEPA 1** (tried layers 12/18/24, z-scoring, and removing per-position
  and per-frame means). Patch-similarity maps are more readable, but they follow appearance ("white things") more
  than object identity.
- **IntPhys (in progress):** early scenes are near chance with the public ViT-L, whichever way the surprise is
  aggregated. This matches a public report for ViT-H
  ([jepa-intuitive-physics #3](https://github.com/facebookresearch/jepa-intuitive-physics/issues/3)). The paper's
  98% came from Meta's own retrained ViT-H with RoPE.
- **CUHK Avenue (zero-shot, nothing trained on Avenue):** frame-level AUC over all 21 test videos is **0.734 micro,
  0.774 macro**. That's with the configuration fixed before any results: 8-frame context, top-5% token error,
  0.5 s smoothing.
  - Per video: median 0.835, range 0.43–0.99 (4 videos ≥ 0.9, 3 below 0.6).
  - The best of 30 exploratory configurations reaches 0.757 / 0.816, but it was chosen after seeing results.
  - For context, Liu et al. (2018) report 85.1% micro AUC with a model *trained* on Avenue's normal videos.
- **Surprise says *when*, not reliably *where*.** Checked against Avenue's pixel-level anomaly masks (videos 03,
  04, 05, 07):
  - Raw per-patch error puts *less* of its top-5% heat inside the anomaly than chance would (0.8–6.3% vs 5.2–9.6%),
    because it is drawn to background texture that is always hard to predict.
  - Relative error (each patch compared with its own history) does 2–3× better than chance on three videos
    (13.6–16.6%), but not on video 07 (3.2%).

  So demos show the video plus the surprise curve, not a heatmap that claims precise localization.

## License

`src/vjepa/third_party/jepa/` is Meta's code under CC BY-NC 4.0 (see its `LICENSE` and `NOTICE`), and the released
weights share that license, so this project is for non-commercial use.
