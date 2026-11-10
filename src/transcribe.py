# ClipForge: creator clip-service pipeline v0
"""
Transcribe audio using faster-whisper with word timestamps.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Load config
config_path = Path('~/projects/clip-forge/config/config.yaml').expanduser()
with open(config_path) as f:
    config = yaml.safe_load(f)

ffmpeg_path = config['ffmpeg_path']
whisper_model = config['whisper_model']
whisper_compute_type = config['whisper_compute_type']
whisper_cpu_threads = config['whisper_cpu_threads']
whisper_word_timestamps = config['whisper_word_timestamps']
whisper_vad = config['whisper_vad']

cache_dir = Path(config['cache_dir'])
cache_dir.mkdir(exist_ok=True)

venv_python = config['venv_path']

def transcribe_audio(audio_path: Path, model_name: str = whisper_model,
                     compute_type: str = whisper_compute_type,
                     cpu_threads: int = whisper_cpu_threads,
                     word_timestamps: bool = whisper_word_timestamps,
                     vad: bool = whisper_vad) -> Dict:
    """Transcribe audio using faster-whisper with word timestamps."""
    print(f"Transcribing {audio_path.name} with model {model_name}...")

    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
    segments, info = model.transcribe(str(audio_path), word_timestamps=word_timestamps, vad_filter=vad)

    result = {
        "model": model_name,
        "duration": info.duration,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": []
    }

    for segment in segments:
        seg_data = {"start": segment.start, "end": segment.end, "text": segment.text.strip(), "words": []}
        if word_timestamps and hasattr(segment, 'words'):
            for word in segment.words:
                seg_data["words"].append({"word": word.word, "start": word.start, "end": word.end})
        result["segments"].append(seg_data)

    return result

def get_transcript_cache_key(video_id: str, model: str = whisper_model) -> str:
    return f"{video_id}_{model}"

def save_transcript(transcript: Dict, cache_dir: Path, video_id: str, model: str = whisper_model):
    cache_key = get_transcript_cache_key(video_id, model)
    cache_path = cache_dir / f"{cache_key}.json"
    with open(cache_path, 'w') as f:
        json.dump(transcript, f, indent=2)
    print(f"Transcript saved to {cache_path}")
    return cache_path

def load_transcript(cache_dir: Path, video_id: str, model: str = whisper_model) -> Optional[Dict]:
    cache_key = get_transcript_cache_key(video_id, model)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)
    transcript = transcribe_audio(audio_path)
    print(f"\nTranscription complete! Duration: {transcript['duration']:.2f}s")
    video_id = audio_path.stem
    save_transcript(transcript, cache_dir, video_id)
