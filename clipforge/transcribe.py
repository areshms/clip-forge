"""yt-dlp audio-only ingest -> faster-whisper base word timestamps -> transcript.json.

Design notes (all measured on this M4, see BUILD_REPORT.md):
  * audio-only download first (2 h VOD ~ 100-130 MB) instead of 4-7 GB of video.
  * faster-whisper float32 is ~1.8x faster than openai-whisper and int8 is SLOWER
    than float32 on Apple silicon, so float32 is the default compute type.
  * VAD on, condition_on_previous_text=False -> fewer hallucinations on
    music/silence/trailers.
  * the transcript is cached by (video_id, model, compute) so the expensive step
    runs once per VOD.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config, util
from .util import log

TRANSCRIPT_VERSION = 1


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def fetch_metadata(url: str, timeout: float = 120) -> dict:
    """yt-dlp -J: full metadata without downloading media."""
    p = util.run(
        [config.YTDLP, "-J", "--no-warnings", "--no-playlist", url],
        timeout=timeout,
    )
    return json.loads(p.stdout)


def download_audio(url: str, work: Path, force: bool = False) -> tuple[Path, Path | None]:
    """Download bestaudio only. Returns (wav16k_mono, original_or_None)."""
    util.ensure_dir(work)
    wav = work / "audio16k.wav"
    src = next(iter(sorted(work.glob("src.*"))), None)
    if wav.exists() and not force:
        log.info("audio cache hit: %s", wav)
        return wav, src

    # 1) audio-only container (small; keeps a copy for [0.2] chat/energy reprocessing)
    p = util.run(
        [
            config.YTDLP, "--no-warnings", "--no-playlist",
            "-f", "bestaudio/best",
            "-o", str(work / "src.%(ext)s"),
            "--print", "after_move:filepath",
            url,
        ],
        timeout=3600,
    )
    src = Path(p.stdout.strip().splitlines()[-1])
    log.info("downloaded audio: %s (%.1f MB)", src.name, src.stat().st_size / 1e6)

    # 2) normalise to 16 kHz mono wav for whisper
    util.run([
        config.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav),
    ], timeout=1800)
    return wav, src


# --------------------------------------------------------------------------- #
# transcribe
# --------------------------------------------------------------------------- #
def transcribe(
    wav: Path,
    model_size: str = None,
    compute_type: str = None,
    cpu_threads: int = None,
    language: str | None = None,
    beam_size: int = 1,
) -> dict:
    from faster_whisper import WhisperModel

    model_size = model_size or config.WHISPER_MODEL
    compute_type = compute_type or config.WHISPER_COMPUTE
    cpu_threads = cpu_threads or config.WHISPER_THREADS

    t0 = time.perf_counter()
    model = WhisperModel(
        model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads
    )
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    seg_iter, info = model.transcribe(
        str(wav),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    segments = []
    for s in seg_iter:
        if s.no_speech_prob is not None and s.no_speech_prob > 0.6:
            continue
        words = [
            {"w": w.word, "s": round(w.start, 3), "e": round(w.end, 3),
             "p": round(w.probability, 3) if w.probability is not None else None}
            for w in (s.words or [])
        ]
        segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": (s.text or "").strip(),
            "no_speech_prob": round(s.no_speech_prob, 3) if s.no_speech_prob is not None else None,
            "words": words,
        })
    t_tx = time.perf_counter() - t1

    audio_s = util.media_summary(wav)["duration_s"] if wav.suffix == ".wav" else 0.0
    return {
        "version": TRANSCRIPT_VERSION,
        "model": model_size,
        "compute_type": compute_type,
        "cpu_threads": cpu_threads,
        "language": info.language,
        "language_probability": round(getattr(info, "language_probability", 0.0) or 0.0, 4),
        "audio": str(wav),
        "audio_duration_s": audio_s,
        "timing": {
            "model_load_s": round(t_load, 3),
            "transcribe_s": round(t_tx, 3),
            "rtf": round(t_tx / audio_s, 4) if audio_s else None,
        },
        "segments": segments,
        "word_count": sum(len(s["words"]) for s in segments),
    }


# --------------------------------------------------------------------------- #
# public entry point (used by forge.py and the smoke test)
# --------------------------------------------------------------------------- #
def cache_path(work: Path, video_id: str, model_size: str, compute_type: str) -> Path:
    return work / f"transcript.{video_id}.{model_size}.{compute_type}.json"


def build_transcript(
    url: str,
    work: Path,
    *,
    model_size: str = None,
    compute_type: str = None,
    force: bool = False,
    keep_audio: bool = True,
) -> dict:
    """Full ingest+transcribe with caching keyed by video id + model."""
    work = util.ensure_dir(Path(work))
    model_size = model_size or config.WHISPER_MODEL
    compute_type = compute_type or config.WHISPER_COMPUTE

    meta = fetch_metadata(url)
    vid = meta.get("id") or "unknown"
    cp = cache_path(work, vid, model_size, compute_type)
    if cp.exists() and not force:
        log.info("transcript cache hit: %s", cp)
        tr = util.read_json(cp)
        tr.setdefault("video", {})
        tr["video"].update(_video_meta(meta))
        return tr

    t_all = time.perf_counter()
    wav, src = download_audio(url, work, force=force)
    tr = transcribe(wav, model_size=model_size, compute_type=compute_type)
    tr["video"] = _video_meta(meta)
    tr["timing"]["total_s"] = round(time.perf_counter() - t_all, 2)
    tr["source_url"] = url
    if not keep_audio and src:
        src.unlink(missing_ok=True)
    util.write_json(cp, tr)
    log.info(
        "transcript: %s segs, %s words, %.1fs audio, rtf=%.3f, wall=%.1fs",
        len(tr["segments"]), tr["word_count"], tr["audio_duration_s"],
        tr["timing"]["rtf"] or -1, tr["timing"]["total_s"],
    )
    return tr


def _video_meta(meta: dict) -> dict:
    return {
        "id": meta.get("id"),
        "title": meta.get("title"),
        "uploader": meta.get("uploader") or meta.get("channel"),
        "duration": meta.get("duration"),
        "webpage_url": meta.get("webpage_url"),
        "upload_date": meta.get("upload_date"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ingest + transcribe a VOD")
    ap.add_argument("--url", required=True)
    ap.add_argument("--work", default=str(config.WORK_ROOT / "work"))
    ap.add_argument("--model", default=config.WHISPER_MODEL)
    ap.add_argument("--compute", default=config.WHISPER_COMPUTE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    util.setup_logging(a.verbose)
    config.require_ffmpeg()
    tr = build_transcript(a.url, Path(a.work), model_size=a.model,
                          compute_type=a.compute, force=a.force)
    print(json.dumps({k: tr[k] for k in
                      ("model", "language", "audio_duration_s", "word_count", "timing")}, indent=1))
    print("transcript ->", cache_path(Path(a.work), tr["video"]["id"] or "unknown", a.model, a.compute))
    return 0


if __name__ == "__main__":
    sys.exit(main())
