"""`uv run vjepa <command>`: download, convert, benchmark, and run the Surprise Meter."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vjepa import checkpoint
from vjepa.device import DTYPES, allocated_memory_gb, autocast, pick_device, synchronize


def cmd_download(args: argparse.Namespace) -> None:
    root = checkpoint.CHECKPOINT_DIR / args.model
    checkpoint.download(checkpoint.pretrain_url(args.model), root / f"{args.model}.pth.tar")
    for task in args.probe:
        checkpoint.download(checkpoint.probe_url(args.model, task), root / f"{task}-probe.pth.tar")


def cmd_convert(args: argparse.Namespace) -> None:
    meta = checkpoint.convert(args.model)
    print(f"wrote {checkpoint.CHECKPOINT_DIR / args.model}/{{{','.join(checkpoint.PARTS)}}}.safetensors; {meta}")


def cmd_bench(args: argparse.Namespace) -> None:
    from vjepa.models import load_vjepa
    from vjepa.surprise import SurpriseMeter

    device = pick_device(args.device)
    model = load_vjepa(args.model, device)
    spec = model.spec
    print(f"device={device} model={args.model}")
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        meter = SurpriseMeter(model, device, dtype)
        for bs in args.batch:
            x = torch.randn(bs, 3, spec.num_frames, spec.img_size, spec.img_size)

            def encode():
                with torch.no_grad(), autocast(device, dtype):
                    model.target_encoder(x.to(device))

            def surprise():
                meter.score_windows(x)

            for name, fn in (("encode", encode), ("surprise x5 ctx", surprise)):
                fn()  # warm-up
                synchronize(device)
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fn()
                synchronize(device)
                dt = (time.perf_counter() - t0) / args.iters
                print(
                    f"{dtype_name:>4} bs={bs} {name:<16} {dt * 1000:8.1f} ms/batch  "
                    f"{bs / dt:6.2f} clips/s  mem={allocated_memory_gb(device):.1f} GB"
                )


def cmd_pca(args: argparse.Namespace) -> None:
    from vjepa.features import extract_tokens
    from vjepa.models import load_vjepa
    from vjepa.video import display_frame, preprocess, read_video
    from vjepa.viz.pca import fit_pca, frame_maps, tokens_to_rgb
    from vjepa.viz.render import to_uint8, upsample, write_mp4

    device = pick_device(args.device)
    frames, fps = read_video(args.video, args.frame_step, args.max_frames)
    model = load_vjepa(args.model, device, with_predictor=False)
    starts, tokens = extract_tokens(model, preprocess(frames, crop=args.crop), device, DTYPES[args.dtype])
    rgb, _ = tokens_to_rgb(tokens, fit_pca(tokens))
    maps = frame_maps(starts, rgb, len(frames), model.spec.tubelet_size)

    size = args.size

    def side_by_side():
        for frame, m in zip(frames, maps):
            view = display_frame(frame, size, args.crop)
            yield np.concatenate([view, to_uint8(upsample(m, (size, size), smooth=not args.blocky))], axis=1)

    out = Path(args.out or f"outputs/pca/{Path(args.video).stem}.mp4")
    write_mp4(side_by_side(), out, fps / args.frame_step)
    print(f"{len(frames)} frames, {len(starts)} windows -> {out}")


def cmd_track(args: argparse.Namespace) -> None:
    from vjepa.features import extract_tokens
    from vjepa.models import load_vjepa
    from vjepa.video import display_frame, preprocess, read_video
    from vjepa.viz.render import write_mp4
    from vjepa.viz.track import heat_overlay, mark, similarity_maps

    device = pick_device(args.device)
    frames, fps = read_video(args.video, args.frame_step, args.max_frames)
    model = load_vjepa(args.model, device, with_predictor=False)
    starts, tokens = extract_tokens(model, preprocess(frames, crop=args.crop), device, DTYPES[args.dtype])
    query_frame, query_yx = min(args.at_frame, len(frames) - 1), (args.y, args.x)
    sims = similarity_maps(tokens, starts, query_frame, query_yx, len(frames))
    lo, hi = np.percentile(sims, 100 - args.show_top), np.percentile(sims, 99.9)  # global: consistent over time

    def side_by_side():
        for i, (frame, sim) in enumerate(zip(frames, sims)):
            view = display_frame(frame, args.size, args.crop)
            if abs(i - query_frame) <= 8:  # show the query point around the moment it was picked
                view = mark(view, query_yx)
            yield np.concatenate([view, heat_overlay(view, sim, lo, hi)], axis=1)

    out = Path(args.out or f"outputs/track/{Path(args.video).stem}.mp4")
    write_mp4(side_by_side(), out, fps / args.frame_step)
    print(f"query token at frame {query_frame}, (y, x)={query_yx} -> {out}")


def cmd_surprise(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vjepa.models import load_vjepa
    from vjepa.surprise import SurpriseMeter
    from vjepa.video import preprocess, read_frame_dir, read_video

    device = pick_device(args.device)
    src = Path(args.video)
    if src.is_dir():
        frames, fps = read_frame_dir(src, args.frame_step), None
    else:
        frames, fps = read_video(src, args.frame_step)
    clip = preprocess(frames, crop=args.crop)
    meter = SurpriseMeter(load_vjepa(args.model, device), device, DTYPES[args.dtype])
    result = meter.score_clip(clip, args.contexts, args.stride, args.batch_size)

    out = Path(args.out or f"outputs/surprise/{src.stem if src.is_file() else src.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    curves = np.stack([result.frame_curve(i, len(frames)) for i in range(len(result.contexts))])
    np.savez(
        out.with_suffix(".npz"),
        starts=result.starts,
        contexts=result.contexts,
        scores=result.scores,
        curves=curves,
        frame_step=args.frame_step,
        fps=fps or np.nan,
    )
    fig, ax = plt.subplots(figsize=(9, 3))
    for c, curve in zip(result.contexts, curves):
        ax.plot(curve, label=f"C={c}")
    ax.set(xlabel=f"frame (every {args.frame_step})", ylabel="surprise (L1)", title=src.name)
    ax.legend(ncol=len(result.contexts), fontsize=8)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    per_ctx = ", ".join(f"C={c}: max {s.max():.4f} mean {s.mean():.4f}" for c, s in zip(result.contexts, result.scores))
    print(f"{src.name}: {len(result.starts)} windows | {per_ctx}\nwrote {out}.npz / .png")


def cmd_classify(args: argparse.Namespace) -> None:
    from vjepa.models import load_vjepa
    from vjepa.probe import classify, load_probe
    from vjepa.video import read_video

    device = pick_device(args.device)
    model = load_vjepa(args.model, device, with_predictor=False)
    probe = load_probe(model, args.task, device)
    frames, _ = read_video(args.video)
    runs = [("forward", frames)] + ([("reversed", frames[::-1])] if args.reverse else [])
    for name, clip in runs:
        print(f"{Path(args.video).name} [{args.task}, {name}]")
        for label, p in classify(model, probe, clip, args.task, device, DTYPES[args.dtype], args.topk):
            print(f"  {p:6.1%}  {label}")


def cmd_eval_intphys(args: argparse.Namespace) -> None:
    import json

    from vjepa.data.intphys import BLOCKS
    from vjepa.eval.intphys_eval import CACHE, evaluate, extract_block, format_report
    from vjepa.models import load_vjepa
    from vjepa.surprise import SurpriseMeter

    device = pick_device(args.device)
    meter = SurpriseMeter(load_vjepa(args.model, device), device, DTYPES[args.dtype])
    run_dir = CACHE / args.model / f"fs{args.frame_step}_stride{args.stride}"
    t0, done = time.perf_counter(), 0

    def progress(name: str) -> None:
        nonlocal done
        done += 1
        print(f"[{time.perf_counter() - t0:6.0f}s] {name} ({done} scenes)", flush=True)

    block_scenes = {
        block: extract_block(
            meter, block, args.frame_step, args.contexts, args.stride, run_dir, progress, args.max_scenes
        )
        for block in args.blocks
    }
    report = evaluate(block_scenes, args.contexts, args.seed)
    table = format_report(report, BLOCKS)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (run_dir / "report.md").write_text(table + "\n")
    print(table)


def cmd_eval_avenue(args: argparse.Namespace) -> None:
    import json

    from vjepa.eval.avenue_eval import CACHE, evaluate, extract, format_report
    from vjepa.models import load_vjepa
    from vjepa.surprise import SurpriseMeter

    device = pick_device(args.device)
    meter = SurpriseMeter(load_vjepa(args.model, device), device, DTYPES[args.dtype])
    run_dir = CACHE / args.model / f"fs{args.frame_step}_stride{args.stride}"
    t0 = time.perf_counter()

    def progress(name: str) -> None:
        print(f"[{time.perf_counter() - t0:6.0f}s] video {name}", flush=True)

    videos = extract(meter, args.frame_step, args.contexts, args.stride, run_dir, progress)
    report = evaluate(videos, args.contexts)
    table = format_report(report)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    (run_dir / "report.md").write_text(table + "\n")
    print(table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vjepa", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download a released checkpoint (resumable)")
    p.add_argument("model", nargs="?", default="vitl16")
    p.add_argument("--probe", nargs="*", default=[], help="also fetch attentive probes, e.g. ssv2 k400")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("convert", help="split the training checkpoint into safetensors")
    p.add_argument("model", nargs="?", default="vitl16")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("bench", help="measure encoder and surprise throughput")
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtypes", nargs="+", default=["fp32", "fp16", "bf16"], choices=list(DTYPES))
    p.add_argument("--batch", nargs="+", type=int, default=[1, 4])
    p.add_argument("--iters", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("pca", help="side-by-side video of the frames and their V-JEPA features (PCA -> RGB)")
    p.add_argument("video")
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--crop", default="center", choices=["square", "center"])
    p.add_argument("--size", type=int, default=540, help="pixel size of each square panel")
    p.add_argument("--blocky", action="store_true", help="show raw 16x16 patches instead of smooth upsampling")
    p.add_argument("--out", help="output .mp4 (default outputs/pca/<name>.mp4)")
    p.set_defaults(func=cmd_pca)

    p = sub.add_parser("track", help="click-a-patch video: where does V-JEPA see the same thing over time?")
    p.add_argument("video")
    p.add_argument("--x", type=float, required=True, help="query x in [0, 1] of the model's square view")
    p.add_argument("--y", type=float, required=True, help="query y in [0, 1] of the model's square view")
    p.add_argument("--at-frame", type=int, default=0, help="query frame index (in sampled frames)")
    p.add_argument("--show-top", type=float, default=5.0, help="percent of most similar tokens to highlight")
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--crop", default="center", choices=["square", "center"])
    p.add_argument("--size", type=int, default=540)
    p.add_argument("--out", help="output .mp4 (default outputs/track/<name>.mp4)")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("surprise", help="surprise curve for a video file or frame folder")
    p.add_argument("video")
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--contexts", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--crop", default="square", choices=["square", "center"])
    p.add_argument("--out", help="output path stem (default outputs/surprise/<name>)")
    p.set_defaults(func=cmd_surprise)

    p = sub.add_parser("classify", help="action labels from Meta's released attentive probes (ssv2 / k400)")
    p.add_argument("video")
    p.add_argument("--task", default="ssv2", choices=["ssv2", "k400"])
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--reverse", action="store_true", help="also classify the video played backwards")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("eval-intphys", help="Surprise Meter on the IntPhys dev set (held-out relative accuracy)")
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--contexts", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    p.add_argument("--blocks", nargs="+", default=["O1", "O2", "O3"])
    p.add_argument("--max-scenes", type=int, help="limit scenes per block (smoke test)")
    p.add_argument("--seed", type=int, default=0, help="seed for the tune/held-out scene split")
    p.set_defaults(func=cmd_eval_intphys)

    p = sub.add_parser("eval-avenue", help="zero-shot frame-level anomaly detection on CUHK Avenue")
    p.add_argument("--model", default="vitl16")
    p.add_argument("--device")
    p.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    p.add_argument("--frame-step", type=int, default=4)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--contexts", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    p.set_defaults(func=cmd_eval_avenue)

    args = parser.parse_args(argv)
    args.func(args)
