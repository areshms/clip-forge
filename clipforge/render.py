"""1080x1920 reframe + burned ASS captions using the pinned static ffmpeg.

render.py consumes a segment produced by cut.py (which starts on a keyframe before
the requested window) and a transcript window. It:

  1. builds an ASS subtitle file for the words inside the window,
  2. trims the leading offset (start of segment -> start of window),
  3. crops/scales to 1080x1920,
  4. burns the ASS subtitles,
  5. loudnorms audio,
  6. encodes x264 veryfast crf20 + aac + faststart.

Reframe: v0 supports two modes:
  * center (default, always works): center-crop a 9:16 window.
  * face (best-effort): if opencv + mediapipe import, bias the crop center toward
    the detected face over a few sample frames; falls back to center on failure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

from . import config, util
from .util import log

RENDER_VERSION = 1

# --------------------------------------------------------------------------- #
# ASS generation
# --------------------------------------------------------------------------- #
ASS_HEADER = """[Script Info]
Title: clip-forge auto captions
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
PlayDepth: 0
Timer: 100.0000
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{outline},0,2,{ml},{mr},{mv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass(
    words: list[dict],
    clip_start: float,
    clip_end: float,
    width: int = 1080,
    height: int = 1920,
    font: str = "Arial",
    font_size: int = 72,
    outline: int = 3,
    margin_v: int = 360,
    margin_lr: int = 60,
    max_words_per_line: int = 5,
) -> str:
    """Create an ASS string for words that fall inside [clip_start, clip_end]."""
    # Shift to clip-local time
    lines = []
    buf = []
    for w in words:
        s, e = w.get("s", 0), w.get("e", 0)
        if e <= clip_start or s >= clip_end:
            continue
        buf.append(w)
        if len(buf) >= max_words_per_line:
            lines.append(buf)
            buf = []
    if buf:
        lines.append(buf)

    header = ASS_HEADER.format(
        w=width, h=height, font=font, size=font_size, outline=outline,
        ml=margin_lr, mr=margin_lr, mv=margin_v,
    )
    events = []
    for ln in lines:
        start = max(0.0, ln[0]["s"] - clip_start)
        end = max(start + 0.1, ln[-1]["e"] - clip_start)
        # keep inside clip
        end = min(end, clip_end - clip_start)
        text = " ".join(_escape_ass(w["w"]) for w in ln)
        events.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def _escape_ass(word: str) -> str:
    """Escape ASS special chars and collapse whitespace."""
    w = re.sub(r"\s+", " ", str(word)).strip()
    for ch in "\{}*":
    w = w.replace(ch, "\\" if ch == "\\" else "\\" + ch)
    return w


# --------------------------------------------------------------------------- #
# reframe helpers
# --------------------------------------------------------------------------- #
def _probe_size(path: Path) -> tuple[int, int]:
    info = util.ffprobe_json(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return int(v["width"]), int(v["height"])


def _center_crop_rect(src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """Return (crop_w, crop_h, x, y) for a 9:16 center crop."""
    target_ratio = config.OUT_W / config.OUT_H  # 1080/1920 = 0.5625
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        # wider than 9:16: crop width
        crop_h = src_h
        crop_w = max(1, int(round(crop_h * target_ratio)))
        # make crop_w even
        crop_w = crop_w - (crop_w % 2)
        x = max(0, (src_w - crop_w) // 2)
        y = 0
    elif src_ratio < target_ratio:
        # taller: crop height
        crop_w = src_w
        crop_h = max(1, int(round(crop_w / target_ratio)))
        crop_h = crop_h - (crop_h % 2)
        x = 0
        y = max(0, (src_h - crop_h) // 2)
    else:
        crop_w, crop_h, x, y = src_w, src_h, 0, 0
    return crop_w, crop_h, x, y


def _face_center_x(path: Path, n_samples: int = 6) -> float | None:
    """Best-effort face detection returning average face center x (0..1)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import mediapipe as mp
        import cv2
    except Exception as e:
        log.debug("face tracking unavailable: %s", e)
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if total <= 0:
        return None

    face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.4)
    xs = []
    for i in range(n_samples):
        t = (total / (n_samples + 1)) * (i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face.process(rgb)
        if res and res.detections:
            for det in res.detections:
                bbox = det.location_data.relative_bounding_box
                xs.append(bbox.xmin + bbox.width / 2)
    cap.release()
    if xs:
        return sum(xs) / len(xs)
    return None


def render_clip(
    src: Path,
    out: Path,
    words: list[dict],
    window: dict,
    offset: float = 0.0,
    reframe_mode: str = "center",
    font_size: int = 72,
    outline: int = 3,
    margin_v: int = 360,
    verbose: bool = False,
) -> dict:
    """Render a finished 9:16 captioned clip.

    window = {"start": float, "end": float, "duration": float}
    offset = seconds to trim from the start of `src` to align with window.start.
    """
    config.require_ffmpeg()
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    clip_start = float(window["start"])
    clip_end = float(window["end"])
    clip_dur = float(window["duration"])
    offset = max(0.0, float(offset))

    # Build ASS in temp file
    ass_str = build_ass(
        words, clip_start, clip_end,
        width=config.OUT_W, height=config.OUT_H,
        font_size=font_size, outline=outline, margin_v=margin_v,
    )
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(ass_str, encoding="utf-8")

    # Compute crop
    src_w, src_h = _probe_size(src)
    crop_w, crop_h, x, y = _center_crop_rect(src_w, src_h)
    if reframe_mode == "face" and config._has_module("mediapipe") and config._has_module("cv2"):
        fx = _face_center_x(src)
        if fx is not None:
            desired_x = int(fx * src_w - crop_w / 2)
            # clamp so crop stays inside frame
            x = max(0, min(src_w - crop_w, desired_x))
            log.info("face-biased crop: x=%d (face_x=%.2f)", x, fx)

    # Even dimensions required by most codecs
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    x -= x % 2
    y -= y % 2

    vf = (
        f"crop={crop_w}:{crop_h}:{x}:{y},"
        f"scale={config.OUT_W}:{config.OUT_H}:flags=lanczos,"
        f"ass={_ffmpeg_escape(str(ass_path))},"
        f"loudnorm=I={config.LOUDNORM_I}:TP={config.LOUDNORM_TP}"
    )

    t0 = time.perf_counter()
    util.run([
        config.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{offset:.3f}", "-t", f"{clip_dur:.3f}", "-i", str(src),
        "-vf", "".join(vf),
        "-c:v", "libx264", "-preset", config.PRESET, "-crf", str(config.CRF),
        "-pix_fmt", "yuv420p",
        "-r", str(config.FPS_OUT),
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out),
    ], timeout=1800)
    dt = time.perf_counter() - t0

    m = util.media_summary(out)
    m.update({
        "render_s": round(dt, 3),
        "ass_path": str(ass_path),
        "crop": {"w": crop_w, "h": crop_h, "x": x, "y": y},
        "reframe_mode": reframe_mode,
    })
    if verbose:
        log.info("render %s -> %s (%dx%d, %.2fs, %.2fs wall)",
                 src.name, out.name, m.get("w"), m.get("h"), m["duration_s"], dt)
    return m


def _ffmpeg_escape(path: str) -> str:
    """ASS file path may contain chars that need escaping in filtergraph."""
    return path.replace("\\", "/").replace("\", "/").replace(":", "\\:")


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def render_clips(
    src: Path,
    transcript: dict,
    windows: list[dict],
    outdir: Path,
    offsets: dict[int, float] | None = None,
    reframe_mode: str = "center",
    **kw,
) -> list[dict]:
    """Render every window to outdir/clip_01.mp4 ..."""
    words = []
    for seg in transcript.get("segments", []):
        words.extend(seg.get("words", []))
    outdir = util.ensure_dir(Path(outdir))
    results = []
    for i, w in enumerate(windows, 1):
        off = (offsets or {}).get(i, 0.0)
        out = outdir / f"clip_{i:02d}.mp4"
        res = render_clip(src, out, words, w, offset=off,
                          reframe_mode=reframe_mode, **kw)
        res["rank"] = w.get("rank", i)
        res["window"] = w
        results.append(res)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="render a 9:16 captioned clip")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--reframe", default="center", choices=["center", "face"])
    ap.add_argument("--font-size", type=int, default=72)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    util.setup_logging(a.verbose)
    config.require_ffmpeg()

    tr = util.read_json(Path(a.transcript))
    words = []
    for seg in tr.get("segments", []):
        words.extend(seg.get("words", []))
    window = {"start": a.start, "end": a.end, "duration": a.end - a.start}
    res = render_clip(Path(a.src), Path(a.out), words, window,
                      offset=a.offset, reframe_mode=a.reframe,
                      font_size=a.font_size, verbose=a.verbose)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())