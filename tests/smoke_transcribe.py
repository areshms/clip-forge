"""Smoke test: transcribe.py on a tiny local wav (no network), then assert shape."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import config, util  # noqa: E402
from clipforge import transcribe as T  # noqa: E402


def main() -> int:
    util.setup_logging()
    ff = config.require_ffmpeg()
    print(f"[1/4] pinned ffmpeg has libass: {config.ffmpeg_has_libass()}  ({ff})")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # synthesise 8 s of speech-free audio (tone) -> whisper should return ~0 segments,
        # which is the correct behaviour with VAD on. Proves the plumbing end to end
        # without needing the network.
        wav = td / "tone.wav"
        subprocess.run([
            str(ff), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
            "-ar", "16000", "-ac", "1", str(wav)], check=True)
        print(f"[2/4] built {wav.name} ({wav.stat().st_size/1024:.0f} KB)")

        tr = T.transcribe(wav)
        print(f"[3/4] transcribed: model={tr['model']} lang={tr['language']} "
              f"segs={len(tr['segments'])} words={tr['word_count']} "
              f"latency={tr['timing']['transcribe_s']}s rtf={tr['timing']['rtf']}")
        assert tr["timing"]["rtf"] is not None, "rtf must be measured"
        assert isinstance(tr["segments"], list)

        out = util.write_json(td / "t.json", tr)
        back = util.read_json(out)
        print(f"[4/4] round-tripped JSON: keys={sorted(back)[:6]}...")

    print("SMOKE transcribe: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
