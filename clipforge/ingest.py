"""Video-segment download helpers (audio-only lives in transcribe.py)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, util
from .util import log


def download_segment(
    url: str,
    start: float,
    end: float,
    out: Path,
    max_height: int = 1080,
    timeout: float = 1800,
) -> Path:
    """Download only the [start,end] section of a VOD via yt-dlp --download-sections.

    This is the winner-only strategy: the full audio file was used for transcript/scoring;
    we pull just the video for the clips that survived selection.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start:.3f}-{end:.3f}"
    fmt = f"bv*[height<={max_height}]+ba/b[height<={max_height}]"
    log.info("downloading section %s from %s", section, url)
    p = util.run([
        config.YTDLP, "--no-warnings", "--no-playlist",
        "-f", fmt,
        "--download-sections", section,
        "-o", str(out.with_suffix(".%(ext)s")),
        "--print", "after_move:filepath",
        url,
    ], timeout=timeout)
    # yt-dlp writes the path with extension; return the actual file
    written = p.stdout.strip().splitlines()[-1]
    return Path(written)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="download a VOD video segment")
    ap.add_argument("--url", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    util.setup_logging(a.verbose)
    out = download_segment(a.url, a.start, a.end, Path(a.out))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
