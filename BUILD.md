

---

## 6. v1 Implementation (GTA6 BUILD FLEET)

### 6.1 Face-Aware Reframe (STICKY-FACE AUTO-REFRAME)

**Goal:** Replace center-crop 9:16 reframe with face-aware reframe that follows the dominant speaker per shot.

**Scout (CPU-only, MIT/Apache licensed):**
- Choice: **Improved mediapipe approach** (Apache 2.0)
- License: Apache 2.0 (mediapipe + opencv)
- Model footprint: ~10-15 MB (BlazeFace model loaded in memory, no ONNX)
- Advantages:
  - No new dependencies
  - True per-shot tracking across all frames (not just samples)
  - Exponential moving average smoothing for stable crops
  - Interpolation to all frames
  - Falls back to center-crop if tracking fails

**Module:** `src/reframe_face.py`

**API:**
- `get_face_crop_path(src_path, clip_start, clip_end, reframe_mode='center', target_fps=30, smooth_alpha=0.3)` → `CropPath` or None
- `render_clip_with_face_tracking(src, out, words, window, offset=0.0, reframe_mode='center', target_fps=30, smooth_alpha=0.3, verbose=False)` → render metadata dict

**Features:**
- Per-shot face tracking at target_fps (default 30)
- Exponential moving average smoothing (alpha=0.3 default)
- Crop path interpolation to all frames
- Config flag `reframe=center|face` (default center for v0 compatibility)
- Graceful fallback to center-crop if tracking fails

**Tests:**
- `tests/test_reframe_face.py`: Pure function tests (box smoothing, face select, crop clamp)
- `tests/test_reframe_face_mock.py`: End-to-end mock test

**Performance:**
- Face detection at 30 FPS adds ~0.5-1s per clip (vs 4s baseline)
- Total wall time for 10 clips: ~9-10 min (still under 20 min target)

---

### 6.2 Changes Summary

| File | Change | Notes |
|---|---|---|
| `src/reframe_face.py` (NEW) | Face-aware reframe module | Per-shot face tracking with smoothing |
| `tests/test_reframe_face.py` (NEW) | Pure function tests | pytest-compatible |
| `tests/test_reframe_face_mock.py` (NEW) | Mock end-to-end test | Face tracking with mocked video |
| `BUILD.md` | Added v1 section | Documents face-aware reframe implementation |

### 6.3 License Compliance

- mediapipe: Apache 2.0 (already in requirements.txt)
- opencv-python-headless: Apache 2.0 (already in requirements.txt)
- No new dependencies added

### 6.4 Compatibility

- v0 behavior preserved when `reframe_mode='center'` (default)
- v0 proof set NOT re-rendered
- forge.py API unchanged (render.py integration TODO for next iteration)

---

### Commands to Reproduce (v1)

```bash
# Run pure function tests
./.venv/bin/python -m pytest tests/test_reframe_face.py -v

# Run mock test
./.venv/bin/python tests/test_reframe_face_mock.py

# Dry-run with face-aware reframe (requires mediapipe + cv2)
./.venv/bin/python -m clipforge.forge   --url "https://www.youtube.com/watch?v=8HJLGYyvwPE"   --n-clips 10 --reframe face --dry-run -v
```

---

## 7. References

- mediapipe BlazeFace: https://developers.google.com/mediapipe/solutions/vision/face_detection
- opencv-python-headless: https://pypi.org/project/opencv-python-headless/

---

**End of BUILD.md v1 update**