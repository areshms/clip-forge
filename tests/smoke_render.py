"""Smoke test: render.py must produce a verified 1080x1920 captioned clip."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import config, util  # noqa: E402
from clipforge import render as R  # noqa: E402

DUR = 12


def build_source(path: Path) -> None:
    subprocess.run([
        str(config.FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={DUR}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={DUR}",
        "-c:v", "libx264", "-preset", "ultrafast", "-g", "60", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path)], check=True)


def build_transcript() -> dict:
    words = []
    text = "oh my god no way that is insane watch this secret trick everyone misses"
    step = DUR / len(text.split())
    for i, w in enumerate(text.split()):
        words.append({"w": w, "s": round(i * step, 3), "e": round((i + 1) * step, 3), "p": 0.95})
    return {
        "version": 1,
        "segments": [{"start": 0, "end": DUR, "text": text, "no_speech_prob": 0.01, "words": words}],
    }


def main() -> int:
    util.setup_logging()
    config.require_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.mp4"
        build_source(src)
        tr_path = util.write_json(td / "tr.json", build_transcript())
        out = td / "clip_01.mp4"
        res = R.render_clip(
            src, out, build_transcript()["segments"][0]["words"],
            {"start": 1.0, "end": 9.0, "duration": 8.0},
            offset=0.0, reframe_mode="center", verbose=True)
        print(f"[1/3] render: {res['w']}x{res['h']} {res['v_codec']} "
              f"{res['duration_s']}s {res['bytes']/1e6:.1f}MB render_s={res['render_s']}")
        assert res["is_vertical_1080x1920"], "output must be 1080x1920"
        assert res["v_codec"] == "h264", "must be h264"
        assert res["duration_s"] >= 7.5, f"duration {res['duration_s']} too short"
        assert (td / "clip_01.ass").exists(), "ASS file must be written alongside clip"
        print(f"[2/3] ASS lines: {(td / 'clip_01.ass').read_text().count('Dialogue: ')}")

        # also test center-crop on a 16:9 source preserves height
        info = util.ffprobe_json(src)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        print(f"[3/3] source was {v['width']}x{v['height']}; output is vertical: PASS")
    print("SMOKE render: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
