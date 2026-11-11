"""Tests for face-aware reframe module."""
import pytest

from src.reframe_face import (
    test_box_smoothing,
    test_face_select,
    test_crop_clamp,
)


class TestFaceReframePure:
    """Pure function tests for face tracking and smoothing."""

    def test_box_smoothing(self):
        """Test exponential moving average smoothing."""
        test_box_smoothing()

    def test_face_select(self):
        """Test face selection logic."""
        test_face_select()

    def test_crop_clamp(self):
        """Test crop coordinate clamping."""
        test_crop_clamp()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
