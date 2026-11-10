"""Stream-copy cuts (no re-encode) with keyframe-exact bookkeeping.

`-ss` before `-i` with `-c copy` can only start on a keyframe, so a naive copy-cut
is silently *late* by up to one GOP. cut.py therefore:

  1. probes the last keyframe at or before `start`          -> kf
  2. cuts from `kf` for `dur + (start - kf) + tail`         -> segment  (copy, ~0.07 s)
  3. returns `offset = start - kf` and `actual_start = kf`

render.py consumes `offset` and trims exactly during its re-encode, so the final
clip start is frame-accurate even though the intermediate was a copy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, util
from .util import log


def probe_video(path: Path) -> dict:
    info = util.ffprobe_json(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {
        "path": str(path),
        "codec": v.get("codec_name"),
        "w": v.get("width"),
        "h": v.get("height"),
        "fps": v.get("r_frame_rate"),
        "duration_s": round(float(info["format"].get("duration") or 0), 3),
        "start_time": float(info["format"].get("start_time") or 0.0),
        "nb_streams": len(info["streams"]),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def nearest_keyframe_before(path: Path, t: float, lookback: float = 12.0) -> float:
    """Return the pts_time of the last video keyframe at or before `t`."""
    lo = max(0.0, t - lookback)
    span = (t - lo) + 0.5
    p = util.run([
        config.FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_entries", "frame=pts_time",
        "-of", "csv=p=0",
        "-read_intervals", f"{lo:.3f}%+{span:.3f}",
        str(path),
    ], check=False)
    times = []
    for line in (p.stdout or "").splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            times.append(float(line))
        except ValueError:
            pass
    ok = [x for x in times if x <= t + 1e-3]
    if ok:
        return max(ok)
    return max(0.0, min(times) if times else 0.0)


def cut_segment(
    src: Path,
    start: float,
    dur: float,
    out: Path,
    *,
    tail: float = 0.6,
    accurate: bool = False,
    verbose: bool = False,
) -> dict:
    """Copy-cut [start, start+dur] out of `src`.

    accurate=False -> stream copy from the preceding keyframe (fast, ~0.07 s/clip)
    accurate=True  -> output-seek re-encode (slow on a long VOD; use only for short sources)
    """
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    src_info = probe_video(src)
    start = max(0.0, float(start))
    dur = float(dur)

    if accurate:
        util.run([
            config.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src), "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out),
        ], timeout=3600)
        meta = {"mode": "reencode", "actual_start": start, "offset": 0.0}
    else:
        kf = nearest_keyframe_before(src, start)
        offset = max(0.0, start - kf)
        length = offset + dur + tail
        util.run([
            config.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{kf:.3f}", "-i", str(src), "-t", f"{length:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", str(out),
        ], timeout=1800)
        meta = {"mode": "copy", "keyframe": round(kf, 3), "offset": round(offset, 3)}

    m = util.media_summary(out)
    res = {
        "src": str(src),
        "out": str(out),
        "requested_start": round(start, 3),
        "requested_dur": round(dur, 3),
        **meta,
        "segment_duration_s": m["duration_s"],
        "segment_bytes": m["bytes"],
        "src_v_codec": src_info["codec"],
        "out_v_codec": m["v_codec"],
        "stream_copied": (m["v_codec"] == src_info["codec"]) if not accurate else False,
        "has_audio": m["has_audio"],
    }
    if verbose:
        log.info("cut %s -> %s (%.1f MB, %.2fs, copy=%s, offset=%.2fs)",
                 src.name, out.name, m["bytes"] / 1e6, m["duration_s"],
                 res["stream_copied"], res.get("offset", 0.0))
    return res


def cut_windows(src: Path, windows: list, outdir: Path, prefix: str = "seg", **kw) -> list:
    outdir = util.ensure_dir(Path(outdir))
    results = []
    for i, w in enumerate(windows, 1):
        res = cut_segment(src, w["start"], w["duration"], outdir / f"{prefix}_{i:02d}.mp4", **kw)
        res["rank"] = w.get("rank", i)
        res["window"] = w
        results.append(res)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="stream-copy cut one or more windows")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True, help="output file or directory")
    ap.add_argument("--start", type=float)
    ap.add_argument("--dur", type=float)
    ap.add_argument("--windows", help="scores.json: cut every clip in it")
    ap.add_argument("--accurate", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    util.setup_logging(a.verbose)
    config.require_ffmpeg()
    if a.windows:
        wins = util.read_json(Path(a.windows))["clips"]
        out = cut_windows(Path(a.src), wins, Path(a.out), verbose=a.verbose)
        print(json.dumps(out, indent=1)[:2000])
    else:
        if a.start is None or a.dur is None:
            ap.error("--start and --dur are required without --windows")
        res = cut_segment(Path(a.src), a.start, a.dur, Path(a.out),
                          accurate=a.accurate, verbose=a.verbose)
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
