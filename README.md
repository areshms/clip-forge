# clip-forge

Local **stream VOD → N ranked vertical captioned clips** pipeline.
The everygen/Klap/Opus replacement for Lane 1 of the GTA6 playbook (see `~/projects/gta6-playbook/PRIME_BRIEF.md` §6).

**Marginal cost per clip: $0.00.** Everything runs on this M4.

## Components (each independently smoke-testable)

| module | job | smoke test |
|---|---|---|
| `clipforge/transcribe.py` | yt-dlp audio-only → faster-whisper `base` float32 (10 threads, VAD, word timestamps) → `transcript.json` | `tests/smoke_transcribe.py` |
| `clipforge/score.py` | `0.5·transcript-heuristic + 0.3·ebur128/astats audio energy + 0.2·chat spikes` → top-N non-overlapping **25–45 s hook-first** windows | `tests/smoke_score.py` |
| `clipforge/cut.py` | ffmpeg **stream-copy** cuts (no re-encode) | `tests/smoke_cut.py` |
| `clipforge/render.py` | pinned ffmpeg 9.0.1: **1080×1920 reframe + burned ASS captions** from whisper segments + loudnorm + contact sheet | `tests/smoke_render.py` |
| `clipforge/forge.py` | orchestrator, `--dry-run`, cache + state | `--dry-run` |

## Hard requirements (verified, do not "fix")

* **ffmpeg must be the static 9.0.1 build** — `~/.hermes/tools/ffmpeg-9.0.1-darwin-arm64/ffmpeg`.
  Homebrew ffmpeg 8.1 has **no libass/drawtext/subtitles**; caption burn silently fails there.
  `config.py` probes `-h filter=ass` and refuses a build without it.
* Python: `~/.hermes/tools/clipforge-venv/bin/python` (faster-whisper 1.1.0).
* `cv2`/`mediapipe` are **optional**: reframe degrades to centre-crop when absent.

## Use

```bash
PY=~/.hermes/tools/clipforge-venv/bin/python
$PY -m clipforge.forge --url "<YouTube VOD URL>" --n 10 --out /tmp/clipforge/out
$PY -m clipforge.forge --url "<URL>" --n 10 --dry-run      # plan only, no download/render
```

See `BUILD_REPORT.md` for measured timings and known gaps, `RESUME.md` for handoff.
