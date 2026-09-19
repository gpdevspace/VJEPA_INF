"""LinkedIn-ready 4:5 compositor: title band, video panels, a surprise curve with playhead, burned-in captions.

Feed autoplay is muted, so every word the viewer needs is on screen. Everything is drawn with PIL per frame.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    ("/System/Library/Fonts/Helvetica.ttc", 1, 0),  # (path, bold index, regular index)
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0, 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, 0),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path, bold_index, regular_index in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=bold_index if bold else regular_index)
    return ImageFont.load_default(size)


@dataclass
class Theme:
    bg: tuple[int, int, int] = (13, 17, 23)
    fg: tuple[int, int, int] = (236, 239, 244)
    muted: tuple[int, int, int] = (139, 148, 158)
    accent: tuple[int, int, int] = (255, 122, 69)  # surprise
    calm: tuple[int, int, int] = (88, 166, 255)  # comparison series
    grid: tuple[int, int, int] = (40, 46, 54)
    band: tuple[int, int, int] = (52, 62, 76)  # shaded ground-truth spans


@dataclass
class Series:
    """One curve on the surprise plot: a value per output frame (NaN where undefined)."""

    values: np.ndarray
    label: str
    color: tuple[int, int, int]


@dataclass
class Caption:
    start: int  # output frame index (inclusive)
    end: int  # exclusive
    text: str


@dataclass
class Scene:
    title: str
    subtitle: str
    panels: Sequence[Sequence[np.ndarray]]  # per panel, one uint8 RGB frame per output frame
    panel_labels: Sequence[str]
    series: Sequence[Series] = ()
    captions: Sequence[Caption] = ()
    markers: Sequence[tuple[int, str]] = field(default_factory=list)  # (frame, label) vertical lines on the plot
    bands: Sequence[tuple[int, int]] = field(default_factory=list)  # [start, end) frame spans shaded on the plot
    band_label: str = ""
    footer: str = ""
    y_range: tuple[float, float] | None = None  # fix the plot's y-axis instead of auto-scaling, and label it
    y_axis_label: str = ""


def _fit(img: np.ndarray, w: int, h: int) -> Image.Image:
    im = Image.fromarray(img)
    scale = min(w / im.width, h / im.height)
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)


def _fit_font(text: str, size: int, max_width: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Largest font (starting at `size`) that fits `text` on one line."""
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fnt = font(size, bold)
    while size > 24 and measure.textlength(text, font=fnt) > max_width:
        size -= 2
        fnt = font(size, bold)
    return fnt


def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def compose(scene: Scene, width: int = 1080, height: int = 1350, theme: Theme = Theme()) -> Iterator[np.ndarray]:
    """Yield composed RGB frames for `scene`, one per panel frame."""
    pad = 48
    sub_f, label_f, cap_f, small_f = font(30), font(26, True), font(38, True), font(24)
    title_f = _fit_font(scene.title, 52, width - 2 * pad)
    n_frames = len(scene.panels[0])
    n_panels = len(scene.panels)
    gap = 24
    panel_w = (width - 2 * pad - gap * (n_panels - 1)) // n_panels
    frame_h, frame_w = scene.panels[0][0].shape[:2]
    panel_h = min(round(panel_w * frame_h / frame_w), 600 if n_panels > 1 else 620)  # follow the video's aspect
    top = 190
    plot_top = top + panel_h + 120  # leaves room for panel labels, then the legend row
    plot_h = 260 if scene.series else 0
    plot_left = pad + (86 if scene.y_range else 10)  # room for tick labels when the axis is absolute
    plot_right = width - pad - 10

    if scene.y_range is not None:
        lo, hi = scene.y_range
    else:
        finite = [s.values[np.isfinite(s.values)] for s in scene.series]
        lo = min((v.min() for v in finite if len(v)), default=0.0)
        hi = max((v.max() for v in finite if len(v)), default=1.0)
        margin = 0.08 * (hi - lo or 1.0)
        lo, hi = lo - margin, hi + margin

    def xy(i: int, v: float) -> tuple[float, float]:
        x = plot_left + (plot_right - plot_left) * i / max(1, n_frames - 1)
        return x, plot_top + plot_h - (v - lo) / (hi - lo) * plot_h

    for t in range(n_frames):
        canvas = Image.new("RGB", (width, height), theme.bg)
        draw = ImageDraw.Draw(canvas)
        draw.text((pad, 44), scene.title, font=title_f, fill=theme.fg)
        draw.text((pad, 112), scene.subtitle, font=sub_f, fill=theme.muted)

        for k, (frames, label) in enumerate(zip(scene.panels, scene.panel_labels)):
            x0 = pad + k * (panel_w + gap)
            im = _fit(frames[t], panel_w, panel_h)
            canvas.paste(im, (x0 + (panel_w - im.width) // 2, top + (panel_h - im.height) // 2))
            draw.text((x0, top + panel_h + 14), label, font=label_f, fill=theme.muted)

        if scene.series:
            for start, end in scene.bands:
                (x0, _), (x1, _) = xy(start, lo), xy(end - 1, lo)
                draw.rectangle([x0, plot_top, max(x1, x0 + 2), plot_top + plot_h], fill=theme.band)
            if scene.y_range is not None:
                for v in np.linspace(lo, hi, 5):
                    _, gy = xy(0, v)
                    draw.line([(plot_left, gy), (plot_right, gy)], fill=theme.grid, width=1)
                    draw.text((plot_left - 12, gy), f"{v:.2f}", font=small_f, fill=theme.muted, anchor="rm")
                if scene.y_axis_label:
                    draw.text((pad, plot_top - 40), scene.y_axis_label, font=small_f, fill=theme.muted)
            draw.line([(plot_left, plot_top + plot_h), (plot_right, plot_top + plot_h)], fill=theme.grid, width=2)
            for frame, text in scene.markers:
                x, _ = xy(frame, lo)
                draw.line([(x, plot_top), (x, plot_top + plot_h)], fill=theme.grid, width=2)
                draw.text((x + 6, plot_top), text, font=small_f, fill=theme.muted)
            legend_x = plot_left if not scene.y_axis_label else plot_left + 150
            for s in scene.series:
                pts = [xy(i, v) for i, v in enumerate(s.values[: t + 1]) if np.isfinite(v)]
                if len(pts) > 1:
                    draw.line(pts, fill=s.color, width=5, joint="curve")
                if pts:
                    x, y = pts[-1]
                    draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=s.color)
                draw.text((legend_x, plot_top - 40), s.label, font=small_f, fill=s.color)
                legend_x += draw.textlength(s.label, font=small_f) + 36
            if scene.bands and scene.band_label:
                draw.rectangle([legend_x, plot_top - 34, legend_x + 22, plot_top - 14], fill=theme.band)
                draw.text((legend_x + 30, plot_top - 40), scene.band_label, font=small_f, fill=theme.muted)
            if not scene.y_axis_label:  # a labelled absolute axis already says which way is up
                axis_note = "higher = more surprised"
                draw.text(
                    (plot_right - draw.textlength(axis_note, font=small_f), plot_top - 40),
                    axis_note,
                    font=small_f,
                    fill=theme.muted,
                )
            x, _ = xy(t, lo)
            draw.line([(x, plot_top), (x, plot_top + plot_h)], fill=theme.fg, width=2)

        for cap in scene.captions:
            if cap.start <= t < cap.end:
                lines = _wrap(draw, cap.text, cap_f, width - 2 * pad)
                y = height - 70 - 50 * len(lines)
                for line in lines:
                    w = draw.textlength(line, font=cap_f)
                    draw.text(((width - w) / 2, y), line, font=cap_f, fill=theme.fg)
                    y += 50
        if scene.footer:
            draw.text((pad, height - 44), scene.footer, font=small_f, fill=theme.muted)
        yield np.asarray(canvas)
