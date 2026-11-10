"""Orchestrator: URL -> N ranked 9:16 captioned clips.

Usage:
    python -m clipforge.forge --url "https://www.youtube.com/watch?v=..." --n 10
    python -m clipforge.forge --url ... --n 10 --dry-run

Stages with timings are written to manifest.json and printed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image

from . import __version__
from . import config, cut, ingest, render, score, transcribe, util
from .util import log

FORGE_VERSION = 1


def _make_contact_sheet(clips: list[Path], out: Path, cols: int = 3) -> Path:
    """Build a tiled JPEG contact sheet of middle frames from each clip."""
    frames = []
    for c in clips:
        # extract middle frame via pinned ffmpeg
        mid = Path(str(c).replace(".mp4", "_midframe.png"))
        util.run([
            config.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", "3", "-i", str(c), "-vf", "select=eq(n\\,0)", "-frames:v", "1",
            "-update", "1", str(mid),
        ], timeout=60)
        if mid.exists():
            frames.append(Image.open(mid).convert("RGB"))
            mid.unlink(missing_ok=True)
    if not frames:
        return None
    w, h = frames[0].size
    rows = math.ceil(len(frames) / cols)
    sheet_w, sheet_h = w * cols, h * rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    for i, im in enumerate(frames):
        x = (i % cols) * w
        y = (i // cols) * h
        sheet.paste(im, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=85)
    return out


def run(
    url: str,
    outdir: Path,
    *,
    n: int = 10,
    min_s: float = None,
    max_s: float = None,
    workdir: Path = None,
    dry_run: bool = False,
    reframe_mode: str = "center",
    keep_audio: bool = True,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    outdir = util.ensure_dir(Path(outdir))
    workdir = util.ensure_dir(Path(workdir) if workdir else config.WORK_ROOT / "work")
    min_s = min_s if min_s is not None else config.CLIP_MIN_S
    max_s = max_s if max_s is not None else config.CLIP_MAX_S

    t0_all = time.perf_counter()
    manifest = {
        "version": FORGE_VERSION,
        "url": url,
        "dry_run": dry_run,
        "n_requested": n,
        "min_clip_s": min_s,
        "max_clip_s": max_s,
        "started_s": t0_all,
    }
    timings = {}

    util.setup_logging(verbose)
    config.require_ffmpeg()
    log.info("clip-forge v%s | ffmpeg=%s | dry_run=%s", __version__, config.FFMPEG, dry_run)

    # 1) ingest + transcribe (cached unless --force)
    t0 = time.perf_counter()
    tr = transcribe.build_transcript(url, workdir, keep_audio=keep_audio, force=force)
    timings["transcribe_s"] = round(time.perf_counter() - t0, 2)
    wav = Path(tr["audio"])
    manifest["transcript"] = {
        "model": tr["model"],
        "language": tr["language"],
        "audio_duration_s": tr["audio_duration_s"],
        "word_count": tr["word_count"],
        "rtf": tr["timing"].get("rtf"),
        "wall_s": timings["transcribe_s"],
    }
    log.info("transcribe: %.1fs audio, rtf=%.3f, wall=%.1fs",
             tr["audio_duration_s"], tr["timing"].get("rtf") or -1, timings["transcribe_s"])

    # 2) score
    t0 = time.perf_counter()
    scored = score.score_transcript(tr, workdir, n_clips=n, min_s=min_s, max_s=max_s,
                                    audio=wav, use_chat=True, force=force)
    timings["score_s"] = round(time.perf_counter() - t0, 2)
    windows = scored["clips"]
    manifest["n_selected"] = len(windows)
    manifest["scoring"] = {
        "duration_s": scored["duration_s"],
        "signals": scored["signals"],
        "n_selected": len(windows),
        "wall_s": timings["score_s"],
    }
    log.info("score: selected %d windows in %.1fs", len(windows), timings["score_s"])
    for c in windows:
        log.info("  #%2d %s  %.1fs  score=%.1f  %s",
                 c["rank"], c["timecode"], c["duration"], c["score"], c["reason"])

    if dry_run:
        manifest["timings"] = timings
        manifest["wall_total_s"] = round(time.perf_counter() - t0_all, 2)
        manifest["clips"] = []
        mp = util.write_json(outdir / "manifest.json", manifest)
        log.info("dry-run complete: %s", mp)
        return manifest

    if not windows:
        raise RuntimeError("no clip windows selected")

    # 3) download + cut + render each window
    clip_files = []
    render_results = []
    segdir = util.ensure_dir(workdir / "segments")
    for i, w in enumerate(windows, 1):
        t0_seg = time.perf_counter()
        seg_raw = segdir / f"seg_{i:02d}_raw.mp4"
        seg_cut = segdir / f"seg_{i:02d}.mp4"

        # download exact section; if yt-dlp returns something longer, cut it
        downloaded = ingest.download_segment(url, w["start"], w["end"], seg_raw)
        cut_res = cut.cut_segment(downloaded, w["start"], w["duration"], seg_cut,
                                  tail=0.6, verbose=verbose)

        final = outdir / f"clip_{i:02d}.mp4"
        render_res = render.render_clip(
            seg_cut, final,
            words=[ww for seg in tr["segments"] for ww in seg.get("words", [])],
            window=w,
            offset=cut_res["offset"],
            reframe_mode=reframe_mode,
            verbose=verbose,
        )
        render_res["rank"] = w["rank"]
        render_res["window"] = w
        render_res["segment_download_s"] = round(time.perf_counter() - t0_seg, 2)
        render_results.append(render_res)
        clip_files.append(final)
        log.info("clip %02d/%02d: %s (%.1fs render, %.1fs total)",
                 i, len(windows), final.name, render_res["render_s"],
                 render_res["segment_download_s"])

    timings["render_total_s"] = round(sum(r["render_s"] for r in render_results), 2)
    timings["download_total_s"] = round(
        sum(r["segment_download_s"] - r["render_s"] for r in render_results), 2)

    # 4) contact sheet
    t0 = time.perf_counter()
    sheet = _make_contact_sheet(clip_files, outdir / "contact_sheet.jpg")
    timings["contact_sheet_s"] = round(time.perf_counter() - t0, 2)
    manifest["contact_sheet"] = str(sheet) if sheet else None

    # 5) manifest
    manifest["timings"] = timings
    manifest["wall_total_s"] = round(time.perf_counter() - t0_all, 2)
    manifest["clips"] = render_results
    mp = util.write_json(outdir / "manifest.json", manifest)
    log.info("manifest: %s", mp)
    log.info("DONE: %d clips in %.1fs", len(clip_files), manifest["wall_total_s"])
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="clip-forge: VOD -> vertical captioned clips")
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", default="/tmp/clipforge/out")
    ap.add_argument("--work", default=str(config.WORK_ROOT / "work"))
    ap.add_argument("-n", "--n-clips", type=int, default=10)
    ap.add_argument("--min", type=float, default=config.CLIP_MIN_S)
    ap.add_argument("--max", type=float, default=config.CLIP_MAX_S)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reframe", default="center", choices=["center", "face"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.url, Path(a.out), n=a.n_clips, min_s=a.min, max_s=a.max,
              workdir=Path(a.work), dry_run=a.dry_run, reframe_mode=a.reframe,
              force=a.force, verbose=a.verbose)
    print(json.dumps({k: res[k] for k in ("n_requested", "n_selected", "wall_total_s", "dry_run")},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
