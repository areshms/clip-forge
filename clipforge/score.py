"""Clip-candidate scoring: fuse 3 signals -> top-N non-overlapping 25-45 s hook-first windows.

    score(t) = 0.5 * transcript_heuristic(t) + 0.3 * audio_energy(t) + 0.2 * chat_spike(t)

Signals (each is a per-second curve in [0, 1] on a common timeline):
  [T] transcript heuristic -- hook vocabulary, money/percent tokens, direct address,
      questions, exclamations, laughter markers, word density, post-silence sentence
      starts. Pure Python regex; no LLM, no network.
  [A] audio energy -- ffmpeg `astats` RMS per 1 s bucket, robust z-score vs a rolling
      median (shouting / laughter / hype rises above a streamer's own baseline). This
      deliberately uses *relative* thresholds because absolute loudness varies wildly
      between channels.
  [C] chat spikes -- `chat-downloader` (Twitch only, optional). Messages per 5 s
      bucket -> z-score + emote-repeat ratio. Degrades to T+A only (weights
      renormalised) when unavailable, which is the normal case for YouTube.

Then: candidate windows of 25-45 s that START on a hook and END on a sentence
boundary, greedy top-N selection with a non-overlap guard, near-duplicate removal.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from . import config, util
from .util import log

SCORE_VERSION = 2

# --- hook vocabulary (kept deliberately small and auditable) ---
HOOK_PATTERNS = [
    (r"\b(what if|imagine|here'?s (the|how)|the (trick|secret|reason)|listen)\b", 3.0),
    (r"\b(oh my god|oh my gosh|no way|holy (shit|crap|smokes)|what the (hell|fuck))\b", 3.5),
    (r"\b(never|always|every single|nobody|everyone|anybody)\b", 1.5),
    (r"\b(insane|unbelievable|crazy|wild|unreal|nuts|broken|disgusting)\b", 2.0),
    (r"\b(million|thousand|hundred|dollars?|bucks|grand|k\b)", 1.5),
    (r"\$\s?\d|\d+\s?%", 2.5),
    (r"\b(the best|the worst|biggest|fastest|rarest|hardest|first ever)\b", 2.0),
    (r"\b(you (can|could|should|have to|need to|gotta)|let me tell you|watch this|look at this)\b", 2.0),
    (r"\b(actually|literally|honestly|basically)\b", 0.5),
    (r"\b(problem|mistake|wrong|fail(ed|ure)?|lost|died|killed)\b", 1.2),
    (r"\b(why|how)\b\s+\w+", 0.8),
    (r"\[\s?(laughter|laughs|music|applause)\s?\]", 2.0),
    (r"\?", 1.0),
    (r"!", 0.8),
    (r"\b(so|okay|alright|right)[,.]", 0.6),
]
HOOK_RE = [(re.compile(p, re.I), w) for p, w in HOOK_PATTERNS]

STOPWORDS = set(
    "a an the and or but if so to of in on at for with is are was were be been am "
    "i you he she it we they me him her them my your his its our their this that "
    "these those do does did done have has had will would can could should just "
    "not no yes yeah okay ok like really very get got go going one two three".split()
)


# --------------------------------------------------------------------------- #
# timeline helpers
# --------------------------------------------------------------------------- #
def _normalize(x, lo_pct=5.0, hi_pct=95.0):
    """Robust percentile scaling to [0,1]."""
    import numpy as np

    x = np.asarray(x, dtype="float64")
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _smooth(x, k=3):
    import numpy as np

    if len(x) < k:
        return x
    kern = np.ones(k) / k
    return np.convolve(x, kern, mode="same")


# --------------------------------------------------------------------------- #
# [T] transcript heuristic
# --------------------------------------------------------------------------- #
def transcript_signal(transcript: dict, duration: float) -> "list[float]":
    """Per-second hook-interest curve derived from word timestamps."""
    import numpy as np

    n = int(math.ceil(duration)) + 1
    curve = np.zeros(n)
    density = np.zeros(n)

    words = []  # (start, end, text)
    for s in transcript["segments"]:
        for w in s["words"]:
            words.append((w["s"], w["e"], w["w"]))
    for s in transcript["segments"]:  # fall back to segment text if no word times
        if not s["words"] and s["text"]:
            words.append((s["start"], s["end"], s["text"]))
    words.sort(key=lambda x: x[0])

    # A) per-word hook weights, spread over the word's span
    for (st, en, txt) in words:
        w = 0.0
        for rx, weight in HOOK_RE:
            m = rx.search(txt)
            if m:
                w += weight
        if w == 0.0:
            continue
        a, b = int(st), min(n - 1, max(int(st), int(math.ceil(en))))
        for i in range(a, b + 1):
            curve[i] += w
        # a hook word also makes the preceding 2 s more interesting (setup)
        for i in range(max(0, a - 2), a):
            curve[i] += w * 0.4

    # B) word density (talking fast = engaged)
    for (st, en, txt) in words:
        a = int(st)
        if 0 <= a < n:
            density[a] += len(txt.split()) / max(0.25, en - st)

    # C) post-silence sentence starts are natural clip openers
    starts = np.zeros(n)
    prev_end = 0.0
    for s in transcript["segments"]:
        gap = s["start"] - prev_end
        if gap >= 0.6 and s["start"] > 1.0:
            i = int(s["start"])
            if 0 <= i < n:
                starts[i] += 1.0
        prev_end = s["end"]

    curve = _smooth(curve, 3) * 1.0 + _normalize(density) * 1.0 + _smooth(starts, 3) * 1.2
    return _normalize(curve).tolist()


# --------------------------------------------------------------------------- #
# [A] audio energy via ffmpeg astats (1 s buckets)
# --------------------------------------------------------------------------- #
def audio_energy_signal(audio: Path, ffmpeg: Path = None) -> tuple[list[float], dict]:
    """RMS level per 1 s bucket -> robust z-score -> [0,1] curve."""
    import numpy as np

    ffmpeg = ffmpeg or config.FFMPEG
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        stats = Path(td) / "stats.txt"
        util.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(audio),
            "-af", (
                "asetnsamples=n=16000:p=0,"
                "astats=metadata=1:reset=1,"
                f"ametadata=print:key=lavfi.astats.Overall.RMS_level:file={stats}"
            ),
            "-f", "null", "-",
        ], timeout=3600)
        txt = stats.read_text(errors="replace")

    levels = []
    for line in txt.splitlines():
        if "Overall.RMS_level=" not in line:
            continue
        raw = line.split("Overall.RMS_level=")[1].strip()
        if raw in ("-inf", "inf", "nan", ""):
            levels.append(-100.0)
            continue
        try:
            levels.append(float(raw))
        except ValueError:
            pass
    if not levels:
        log.warning("astats produced no RMS levels; audio signal disabled")
        return [], {"buckets": 0, "elapsed_s": round(time.perf_counter() - t0, 2)}

    x = np.asarray(levels, dtype="float64")
    k = 61  # rolling median window (~1 min) = the channel's own baseline
    if x.size >= k:
        pad = np.pad(x, (k // 2, k // 2), mode="edge")
        base = np.array([np.median(pad[i:i + k]) for i in range(x.size)])
    else:
        base = np.full_like(x, float(np.median(x)))
    lift = x - base                      # dB above own rolling baseline
    curve = _normalize(lift)
    curve = _smooth(curve, 3)
    return curve.tolist(), {
        "buckets": int(x.size),
        "baseline_median_db": round(float(np.median(base)), 2),
        "peak_lift_db": round(float(lift.max()), 2),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# [C] chat spikes (Twitch, optional)
# --------------------------------------------------------------------------- #
def chat_signal(video_url: str, duration: float, bucket_s: float = 5.0):
    """Messages-per-bucket z-score + emote-repeat ratio. Returns ([], reason) if unusable."""
    import numpy as np

    try:
        from chat_downloader import ChatDownloader
    except Exception as e:
        return [], {"available": False, "reason": f"chat_downloader not installed ({e})"}
    if "twitch.tv" not in video_url and "youtube.com" not in video_url:
        return [], {"available": False, "reason": "unsupported host"}

    t0 = time.perf_counter()
    try:
        chat = ChatDownloader().get_chat(video_url, timeout=120)
        nb = int(math.ceil(duration / bucket_s)) + 1
        counts = np.zeros(nb)
        emotes = 0
        total = 0
        for m in chat:
            off = m.get("time_in_seconds")
            if off is None:
                continue
            i = min(nb - 1, int(off // bucket_s))
            counts[i] += 1
            total += 1
            if m.get("emotes"):
                emotes += 1
            if total > 400_000:
                break
    except Exception as e:
        return [], {"available": False, "reason": f"chat fetch failed: {e}"}

    if total < 50:
        return [], {"available": False,
                    "reason": f"only {total} messages (chat replay absent or tiny channel)"}

    # z-score vs rolling median, then upsample bucket -> second
    k = 13  # ~65 s
    pad = np.pad(counts, (k // 2, k // 2), mode="edge")
    base = np.array([np.median(pad[i:i + k]) for i in range(counts.size)])
    mad = np.median(np.abs(counts - base)) or 1.0
    z = np.clip((counts - base) / (1.4826 * mad), 0, None)
    z = _normalize(z)
    n = int(math.ceil(duration)) + 1
    curve = np.interp(np.arange(n), np.arange(counts.size) * bucket_s, z)
    return curve.tolist(), {
        "available": True,
        "messages": int(total),
        "emote_ratio": round(emotes / max(1, total), 3),
        "peak_bucket_msgs": int(counts.max()),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# --------------------------------------------------------------------------- #
# fusion + window selection
# --------------------------------------------------------------------------- #
def fuse(duration, t_curve, a_curve, c_curve, weights=None):
    import numpy as np

    weights = weights or {"transcript": config.W_TRANSCRIPT, "audio": config.W_AUDIO,
                          "chat": config.W_CHAT}
    n = int(math.ceil(duration)) + 1

    def pad(x):
        x = np.asarray(x or [], dtype="float64")
        if x.size == 0:
            return None
        if x.size < n:
            x = np.pad(x, (0, n - x.size), mode="edge")
        return x[:n]

    t, a, c = pad(t_curve), pad(a_curve), pad(c_curve)
    parts = [(t, weights["transcript"]), (a, weights["audio"])]
    if c is not None:
        parts.append((c, weights["chat"]))
    total_w = sum(w for x, w in parts if x is not None)
    fused = np.zeros(n)
    used = {}
    for x, w in parts:
        if x is None or total_w <= 0:
            continue
        fused += x * (w / total_w)
        used[name_of(x, t, a, c)] = round(w / total_w, 3)
    return fused, used


def name_of(x, t, a, c):
    for nm, v in (("transcript", t), ("audio", a), ("chat", c)):
        if v is not None and x is v:
            return nm
    return "?"


def _hook_score_at(t_curve, sec, lookahead=3.0) -> float:
    import numpy as np

    i = max(0, int(sec))
    j = min(len(t_curve), i + int(lookahead) + 1)
    return float(np.max(t_curve[i:j])) if j > i else 0.0


def select_windows(
    transcript: dict,
    duration: float,
    t_curve, a_curve, c_curve,
    n_clips: int = 10,
    min_s: float = None,
    max_s: float = None,
    guard_s: float = 8.0,
    dedupe_jaccard: float = 0.55,
):
    """Top-N hook-first windows, non-overlapping, near-duplicate-suppressed."""
    import numpy as np

    min_s = min_s or config.CLIP_MIN_S
    max_s = max_s or config.CLIP_MAX_S
    fused, weight_used = fuse(duration, t_curve, a_curve, c_curve)
    t_arr = np.asarray(t_curve or [0.0] * len(fused), dtype="float64")

    # candidate starts: sentence starts (preferred) + strong local maxima
    starts = set()
    prev_end = 0.0
    for s in transcript["segments"]:
        if s["start"] - prev_end >= 0.5 and 2.0 <= s["start"] <= duration - min_s:
            starts.add(round(s["start"], 2))
        prev_end = s["end"]
    if t_arr.size:
        thr = float(np.percentile(t_arr, 90))
        for i in np.where(t_arr >= thr)[0]:
            if 2.0 <= i <= duration - min_s:
                starts.add(float(i))
    starts = sorted(starts)
    if not starts:
        starts = [float(x) for x in range(2, max(3, int(duration - min_s)), 5)]

    seg_ends = sorted(s["end"] for s in transcript["segments"])

    cands = []
    for st in starts:
        hook = _hook_score_at(t_arr, st)
        # try a few lengths, prefer ending on a sentence boundary
        best = None
        for length in (max_s, (min_s + max_s) / 2, min_s, (min_s + max_s) / 2 + 5):
            en = st + length
            if en > duration:
                continue
            # snap end to the nearest sentence end within +/- 4 s
            near = [e for e in seg_ends if abs(e - en) <= 4.0]
            en2 = min(near, key=lambda e: abs(e - en)) if near else en
            if not (min_s - 1 <= en2 - st <= max_s + 1):
                en2 = en
            mean_f = float(np.mean(fused[int(st):int(en2) + 1])) if int(en2) > int(st) else 0.0
            peak_f = float(np.max(fused[int(st):int(en2) + 1])) if int(en2) > int(st) else 0.0
            # hook-first bonus: interesting within the first 3 s
            score = 100 * (0.65 * mean_f + 0.35 * peak_f) + 8 * hook
            if best is None or score > best[0]:
                best = (score, st, en2, mean_f, peak_f, hook)
        if best:
            cands.append(best)

    cands.sort(key=lambda c: -c[0])
    chosen, taken_spans = [], []
    for score, st, en, mean_f, peak_f, hook in cands:
        if any(not (en + guard_s <= s2 or st - guard_s >= e2) for s2, e2 in taken_spans):
            continue
        chosen.append({
            "start": round(st, 2), "end": round(en, 2),
            "duration": round(en - st, 2),
            "score": round(min(100.0, score), 1),
            "subscores": {"fused_mean": round(mean_f, 3), "fused_peak": round(peak_f, 3),
                          "hook": round(hook, 3)},
        })
        taken_spans.append((st, en))
        if len(chosen) >= max(n_clips * 3, n_clips):
            break

    # near-duplicate suppression on content words
    def kw(win):
        ws = set()
        for s in transcript["segments"]:
            if s["start"] >= win["start"] - 1 and s["end"] <= win["end"] + 1:
                ws |= {w.lower() for w in re.findall(r"[a-z']+", s["text"].lower())
                       if w not in STOPWORDS and len(w) > 2}
        return ws

    kept = []
    for w in chosen:
        k = kw(w)
        w["keywords"] = sorted(k)[:12]
        if any(len(k & kw(x)) / max(1, len(k | kw(x))) > dedupe_jaccard for x in kept):
            continue
        kept.append(w)
        if len(kept) >= n_clips:
            break

    for i, w in enumerate(kept, 1):
        w["rank"] = i
        w["text"] = _window_text(transcript, w["start"], w["end"])
        w["reason"] = _reason(w)
        w["timecode"] = f"{util.hms(w['start'])}-{util.hms(w['end'])}"
    return kept, {"weights_normalised": weight_used,
                  "candidates": len(cands), "after_dedupe": len(kept)}


def _window_text(transcript: dict, st: float, en: float) -> str:
    parts = [s["text"] for s in transcript["segments"]
             if s["end"] > st and s["start"] < en and s["text"]]
    return " ".join(parts).strip()


def _reason(w: dict) -> str:
    sub = w["subscores"]
    bits = []
    if sub["hook"] > 0.5:
        bits.append("hook language in the opening seconds")
    if sub["fused_peak"] > 0.6:
        bits.append("strong peak (energy/interest spike)")
    if sub["fused_mean"] > 0.5:
        bits.append("consistently engaging throughout")
    return "; ".join(bits) or "best available window"


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def score_transcript(
    transcript: dict,
    work: Path,
    *,
    n_clips: int = 10,
    min_s: float = None,
    max_s: float = None,
    audio: Path | None = None,
    use_chat: bool = True,
    force: bool = False,
) -> dict:
    work = util.ensure_dir(Path(work))
    vid = (transcript.get("video") or {}).get("id") or "unknown"
    cp = work / f"scores.{vid}.n{n_clips}.json"
    if cp.exists() and not force:
        log.info("scores cache hit: %s", cp)
        return util.read_json(cp)

    duration = float(transcript.get("audio_duration_s") or 0) or         float((transcript.get("video") or {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("cannot score: unknown duration")

    timings = {}
    t0 = time.perf_counter()
    t_curve = transcript_signal(transcript, duration)
    timings["transcript_signal_s"] = round(time.perf_counter() - t0, 2)

    a_curve, a_info = [], {"buckets": 0}
    if audio and Path(audio).exists():
        a_curve, a_info = audio_energy_signal(Path(audio))
    timings["audio_signal_s"] = a_info.get("elapsed_s")

    c_curve, c_info = [], {"available": False, "reason": "disabled"}
    if use_chat:
        t0 = time.perf_counter()
        c_curve, c_info = chat_signal(transcript.get("source_url") or "", duration)
        timings["chat_signal_s"] = round(time.perf_counter() - t0, 2)

    windows, sel_info = select_windows(
        transcript, duration, t_curve, a_curve, c_curve, n_clips=n_clips,
        min_s=min_s, max_s=max_s,
    )
    out = {
        "version": SCORE_VERSION,
        "video": transcript.get("video"),
        "duration_s": round(duration, 2),
        "clip_window_s": [min_s or config.CLIP_MIN_S, max_s or config.CLIP_MAX_S],
        "signals": {
            "transcript": {"available": True, "seconds": len(t_curve)},
            "audio": {"available": bool(a_curve), **a_info},
            "chat": {"available": bool(c_curve), **c_info},
        },
        "selection": sel_info,
        "timings": timings,
        "clips": windows,
    }
    util.write_json(cp, out)
    log.info("scored %s -> %d clips in %.1fs (chat=%s, audio=%s)",
             vid, len(windows), sum(v for v in timings.values() if v), 
             c_info.get("available"), bool(a_curve))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score a transcript into clip windows")
    ap.add_argument("--transcript", required=True, help="transcript.json path")
    ap.add_argument("--audio", default=None, help="16 kHz wav (for astats energy)")
    ap.add_argument("--work", default=str(config.WORK_ROOT / "work"))
    ap.add_argument("-n", "--n-clips", type=int, default=10)
    ap.add_argument("--min", type=float, default=config.CLIP_MIN_S)
    ap.add_argument("--max", type=float, default=config.CLIP_MAX_S)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    util.setup_logging(a.verbose)
    tr = util.read_json(Path(a.transcript))
    res = score_transcript(tr, Path(a.work), n_clips=a.n_clips, min_s=a.min, max_s=a.max,
                           audio=Path(a.audio) if a.audio else None,
                           use_chat=not a.no_chat, force=a.force)
    for c in res["clips"]:
        print(f"{c['rank']:>2}. {c['timecode']}  {c['duration']:>5.1f}s  "
              f"score={c['score']:>5.1f}  {c['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
