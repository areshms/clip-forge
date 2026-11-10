"""Pinned, verified-environment configuration.

Every path here was verified on this machine (Mac16,10 / M4 / macOS 26.5.1).
Anything that silently degrades clip quality (like a libass-less ffmpeg) is a
hard error, not a warning.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

HOME = Path.home()


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# --- ffmpeg: MUST have libass. Homebrew 8.1 does not. ---
FFMPEG = _env_path(
    "CLIPFORGE_FFMPEG", str(HOME / ".hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg")
)
FFPROBE = _env_path(
    "CLIPFORGE_FFPROBE", str(HOME / ".hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffprobe")
)
YTDLP = _env_path("CLIPFORGE_YTDLP", shutil.which("yt-dlp") or "/opt/homebrew/bin/yt-dlp")
PYTHON = _env_path(
    "CLIPFORGE_PYTHON", str(HOME / ".hermes/tools/clipforge-venv/bin/python")
)

# --- output geometry ---
OUT_W, OUT_H = 1080, 1920
FPS_OUT = 30
CRF = 20
PRESET = "veryfast"
LOUDNORM_I = -14.0
LOUDNORM_TP = -1.5

# --- transcription ---
WHISPER_MODEL = os.environ.get("CLIPFORGE_WHISPER_MODEL", "base")
WHISPER_COMPUTE = "float32"  # int8 measured SLOWER than float32 on M4
WHISPER_THREADS = int(os.environ.get("CLIPFORGE_THREADS", "10"))

# --- clip windows ---
CLIP_MIN_S = 25.0
CLIP_MAX_S = 45.0

# --- scoring weights (must sum to 1.0) ---
W_TRANSCRIPT = 0.5
W_AUDIO = 0.3
W_CHAT = 0.2

# --- default workspace ---
WORK_ROOT = _env_path("CLIPFORGE_WORK", "/tmp/clipforge")


class FfmpegMissing(RuntimeError):
    pass


def ffmpeg_has_libass(ffmpeg: Path = None) -> bool:
    ffmpeg = ffmpeg or FFMPEG
    if not Path(ffmpeg).exists():
        return False
    try:
        out = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-h", "filter=ass"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return False
    return "Render ASS subtitles" in out


def require_ffmpeg() -> Path:
    """Fail loudly if the configured ffmpeg cannot burn captions."""
    if not Path(FFMPEG).exists():
        raise FfmpegMissing(
            f"pinned ffmpeg not found at {FFMPEG}. Install the static 9.0.1 build or "
            f"set CLIPFORGE_FFMPEG."
        )
    if not ffmpeg_has_libass():
        raise FfmpegMissing(
            f"{FFMPEG} has NO libass (no `ass`/`subtitles`/`drawtext` filter). "
            f"Caption burn would silently produce uncaptioned video. "
            f"Homebrew ffmpeg 8.1 is known-bad; use the pinned 9.0.1 static build."
        )
    return FFMPEG


def probe() -> dict:
    """Return a dict describing the live environment (used by --dry-run)."""
    return {
        "ffmpeg": str(FFMPEG),
        "ffmpeg_exists": Path(FFMPEG).exists(),
        "ffmpeg_libass": ffmpeg_has_libass(),
        "ffprobe": str(FFPROBE),
        "ffprobe_exists": Path(FFPROBE).exists(),
        "yt_dlp": str(YTDLP),
        "yt_dlp_exists": Path(YTDLP).exists(),
        "python": str(PYTHON),
        "python_exists": Path(PYTHON).exists(),
        "whisper_model": WHISPER_MODEL,
        "whisper_compute": WHISPER_COMPUTE,
        "whisper_threads": WHISPER_THREADS,
        "clip_window_s": [CLIP_MIN_S, CLIP_MAX_S],
        "weights": {"transcript": W_TRANSCRIPT, "audio": W_AUDIO, "chat": W_CHAT},
        "cv2": _has_module("cv2"),
        "mediapipe": _has_module("mediapipe"),
        "chat_downloader": _has_module("chat_downloader"),
    }


def _has_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
