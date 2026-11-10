"""Shared helpers: subprocess runner, JSON/JSONL io, logging, ffprobe."""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

log = logging.getLogger("clipforge")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


class CmdError(RuntimeError):
    def __init__(self, cmd: Sequence[str], rc: int, out: str, err: str):
        self.cmd, self.rc, self.out, self.err = list(cmd), rc, out, err
        super().__init__(
            f"command failed rc={rc}: {shlex.join(str(c) for c in cmd)}\n"
            f"--- stderr tail ---\n{err[-3000:]}"
        )


def run(
    cmd: Sequence[Any],
    *,
    check: bool = True,
    timeout: float | None = None,
    capture: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run a command, log it, and (by default) raise CmdError on failure."""
    cmd = [str(c) for c in cmd]
    log.debug("$ %s", shlex.join(cmd))
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd else None,
    )
    dt = time.perf_counter() - t0
    log.debug("  -> rc=%s in %.2fs", p.returncode, dt)
    if check and p.returncode != 0:
        raise CmdError(cmd, p.returncode, p.stdout or "", p.stderr or "")
    return p


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False)
    tmp.replace(path)
    return path


def append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def ffprobe_json(path: Path, ffprobe: Path | None = None) -> dict:
    ffprobe = ffprobe or config.FFPROBE
    p = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    return json.loads(p.stdout)


def media_summary(path: Path) -> dict:
    """Compact, human-checkable summary of a produced file."""
    info = ffprobe_json(path)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"].get("duration") or 0)
    out = {
        "path": str(path),
        "bytes": int(info["format"].get("size") or 0),
        "duration_s": round(dur, 3),
        "v_codec": v and v.get("codec_name"),
        "w": v and v.get("width"),
        "h": v and v.get("height"),
        "fps": (v and v.get("r_frame_rate")),
        "a_codec": a and a.get("codec_name"),
        "a_ch": a and a.get("channels"),
        "a_rate": a and a.get("sample_rate"),
    }
    out["has_video"] = v is not None
    out["has_audio"] = a is not None
    out["is_vertical_1080x1920"] = (out["w"] == config.OUT_W and out["h"] == config.OUT_H)
    return out


def which(name: str) -> str | None:
    return shutil.which(name)


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
