"""Smoke test: cut.py must be frame-budget-correct and stream-copy (no re-encode)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import config, util  # noqa: E402
from clipforge import cut as C  # noqa: E402


def build_source(path: Path, dur: int = 40) -> None:
    subprocess.run([
        str(config.FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
        "-c:v", "libx264", "-preset", "ultrafast", "-g", "60", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(path)], check=True)


def main() -> int:
    util.setup_logging()
    config.require_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.mp4"
        build_source(src)
        info = C.probe_video(src)
        print(f"[1/4] source: {info['w']}x{info['h']} {info['codec']} "
              f"{info['duration_s']}s gop=60f(2s)")

        kf = C.nearest_keyframe_before(src, 12.5)
        print(f"[2/4] nearest keyframe <= 12.5s: {kf}  (expect {12.0})")
        assert kf <= 12.5, "keyframe must not be after the requested point"

        res = C.cut_segment(src, 12.5, 20.0, td / "seg.mp4")
        print(f"[3/4] cut: requested 12.5+20s -> actual_start={res['actual_start']} "
              f"offset={res['offset']}s seg_len={res['segment_duration_s']}s "
              f"copy={res['stream_copied']}")
        assert res["stream_copied"], "must be a stream copy, not a re-encode"
        assert res["out_v_codec"] == info["codec"]
        assert res["segment_duration_s"] >= 20.0, "segment must cover the requested window"

        # the offset must be exactly what render needs: kf + offset == requested start
        assert abs((res["actual_start"] + res["offset"]) - 12.5) < 1e-6
        print(f"[4/4] offset arithmetic OK: {res['actual_start']} + {res['offset']} = 12.5")
    print("SMOKE cut: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
