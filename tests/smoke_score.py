"""Smoke test: score.py must find a PLANTED viral moment.

Builds 300 s of synthetic audio (quiet tone + one loud 20 s hype burst at t=120)
and a synthetic transcript (humdrum speech everywhere, hook language inside the
burst). The scorer must rank a window overlapping the burst #1. Offline, no network.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import config, util  # noqa: E402
from clipforge import score as S  # noqa: E402

DUR = 300
BURST_START, BURST_END = 110, 135  # planted viral moment


def build_audio(path: Path) -> None:
    """Quiet baseline tone with a loud burst -> astats must see a big RMS lift."""
    fc = (
        f"sine=frequency=200:duration={DUR}[base];"
        f"sine=frequency=800:duration={BURST_END-BURST_START}[burst];"
        f"[base]volume=0.05[b];[burst]volume=0.9[bu];"
        f"[b][bu]amix=inputs=2:duration=first:dropout_transition=0[out]"
    )
    # place the burst at the right offset using adelay
    fc2 = (
        f"[0:a]volume=0.05[b];"
        f"[1:a]volume=0.95,adelay={int(BURST_START*1000)}|{int(BURST_START*1000)},apad[bu];"
        f"[b][bu]amix=inputs=2:duration=longest,atrim=0:{DUR},asetnsamples=n=16000[out]"
    )
    subprocess.run([
        str(config.FFMPEG), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=200:duration={DUR}",
        "-f", "lavfi", "-i", f"sine=frequency=800:duration={BURST_END-BURST_START}",
        "-filter_complex", fc2, "-map", "[out]",
        "-ar", "16000", "-ac", "1", str(path)], check=True)


def build_transcript() -> dict:
    segs = []
    t = 0.0
    i = 0
    filler = [
        "so anyway we are just walking around the map right now",
        "i need to pick up some supplies before the next mission",
        "yeah the traffic in this game is really something else",
        "let me check the map and see where we are heading",
    ]
    while t < DUR - 4:
        in_burst = BURST_START <= t < BURST_END
        if in_burst:
            text = ("oh my god no way that is insane what the hell just happened "
                    "i literally cannot believe that worked") if i % 2 == 0 else \
                   ("watch this you have to see this the secret trick everyone "
                    "misses it is the fastest way to do it")
        else:
            text = filler[i % len(filler)]
        en = min(t + 3.4, DUR)
        words = []
        toks = text.split()
        step = (en - t) / max(1, len(toks))
        for k, w in enumerate(toks):
            words.append({"w": w, "s": round(t + k * step, 3),
                          "e": round(t + (k + 1) * step, 3), "p": 0.9})
        segs.append({"start": round(t, 3), "end": round(en, 3), "text": text,
                     "no_speech_prob": 0.01, "words": words})
        t = en + 0.5
        i += 1
    return {"version": 1, "model": "synthetic", "audio_duration_s": float(DUR),
            "video": {"id": "synthetic", "duration": DUR}, "segments": segs,
            "word_count": sum(len(s["words"]) for s in segs),
            "timing": {"transcribe_s": 0.0, "rtf": 0.0}}


def main() -> int:
    util.setup_logging()
    config.require_ffmpeg()
    tr = build_transcript()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "synth.wav"
        build_audio(wav)
        print(f"[1/3] synthesised {DUR}s audio with a planted burst "
              f"at {BURST_START}-{BURST_END}s ({wav.stat().st_size/1e6:.1f} MB)")
        print(f"[2/3] synthetic transcript: {len(tr['segments'])} segments, "
              f"{tr['word_count']} words")

        a_curve, a_info = S.audio_energy_signal(wav)
        top = max(range(len(a_curve)), key=lambda i: a_curve[i])
        print(f"      audio signal: {a_info['buckets']} buckets, peak lift "
              f"{a_info['peak_lift_db']} dB, argmax={top}s")
        assert BURST_START - 8 <= top <= BURST_END + 8, f"astats missed the burst (argmax={top})"

        res = S.score_transcript(tr, td / "work", n_clips=5, audio=wav, use_chat=False)
        print(f"[3/3] selected {len(res['clips'])} clips "
              f"(chat available={res['signals']['chat'].get('available')}):")
        for c in res["clips"]:
            print(f"      #{c['rank']} {c['timecode']} {c['duration']:.1f}s "
                  f"score={c['score']} hook={c['subscores']['hook']}")
        assert res["clips"], "no clips selected"
        top_clip = res["clips"][0]
        hit = top_clip["start"] <= BURST_END and top_clip["end"] >= BURST_START
        assert hit, f"rank-1 clip {top_clip['timecode']} misses the planted burst"
        assert top_clip["duration"] >= config.CLIP_MIN_S - 1, "clip shorter than 25 s"
        assert top_clip["duration"] <= config.CLIP_MAX_S + 1, "clip longer than 45 s"
        print(f"      rank-1 overlaps the planted burst: PASS")
    print("SMOKE score: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
