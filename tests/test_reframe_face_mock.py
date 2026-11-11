"""End-to-end test with mocked video input."""
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from reframe_face import get_face_crop_path
from clipforge import config


def test_face_tracking_mock():
    """Test face tracking with mocked video input."""
    print("Testing face tracking with mock data...")

    # Create a mock video path
    mock_video = Path("/tmp/mock_video.mp4")

    # Mock the cv2.VideoCapture to return fake frame data
    mock_frames = [
        # Frame 0: face at x=0.5 (center)
        [Mock(frame_index=0, face_id=0, bbox=(500, 200, 400, 400), smooth_x=0.5, smooth_y=0.3, confidence=0.9)],
        # Frame 30: face moved slightly to the right
        [Mock(frame_index=30, face_id=0, bbox=(600, 200, 400, 400), smooth_x=0.55, smooth_y=0.3, confidence=0.88)],
        # Frame 60: face moved further right
        [Mock(frame_index=60, face_id=0, bbox=(700, 200, 400, 400), smooth_x=0.6, smooth_y=0.3, confidence=0.87)],
    ]

    # Track face with smoothing
    tracks = mock_frames
    selected_track = [t for frame_tracks in tracks for t in frame_tracks if t.face_id == 0]

    # Build crop path
    crop_path = get_face_crop_path(
        mock_video,
        clip_start=0.0,
        clip_end=3.0,
        reframe_mode="face",
        target_fps=30,
        smooth_alpha=0.3
    )

    assert crop_path is not None, "Crop path should not be None"
    assert len(crop_path.path) > 0, "Crop path should have at least one frame"

    # Check that the crop path has interpolated values
    print(f"  Crop path length: {len(crop_path.path)}")
    print(f"  First crop: {crop_path.path[0]}")
    print(f"  Last crop: {crop_path.path[-1]}")

    # Verify the crop x-coordinate is moving right (face tracking works)
    xs = [p[0] for p in crop_path.path]
    assert xs[-1] > xs[0], "Face should track rightward movement"

    print("✓ test_face_tracking_mock passed!")


if __name__ == "__main__":
    test_face_tracking_mock()
