# clip-forge

Local, $0-marginal-cost creator clip-service pipeline.

Paste a Twitch/YouTube VOD URL → N ranked 9:16 vertical clips with burned ASS captions.

## Pipeline stages

1. `ingest.py` / `transcribe.py` — yt-dlp audio-only + faster-whisper base word timestamps → JSON
2. `score.py` — LLM transcript + audio energy + chat spikes → top-N non-overlapping 25–45 s windows
3. `cut.py` — ffmpeg stream-copy per window
4. `render.py` — pinned static ffmpeg 9.0.1: 1080×1920 reframe + burned ASS captions
5. `forge.py` — orchestrator with `--dry-run`

## Verified environment

- macOS 26.5.1, Apple M4, 16 GB RAM
- `~/.hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg` (libass/drawtext/ass enabled)
- faster-whisper 1.1.0 / ctranslate2 4.7.1 (MPT venv or `~/.hermes/tools/clipforge-venv`)
- Measured: 2 h VOD transcript ≈ 9 min; 35 s clip render ≈ 4 s

See BUILD_REPORT.md for the v0 run.
