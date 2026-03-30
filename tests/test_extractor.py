from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from peakframer.extractor import (
    ExtractedFrame,
    extract_frames,
    get_video_meta,
    suggest_sample_rate,
)


def _make_mock_cap(frames: list[np.ndarray], fps: float = 30.0) -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_COUNT: len(frames),
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
        cv2.CAP_PROP_POS_MSEC: 0.0,
    }.get(prop, 0.0)

    read_returns = [(True, f) for f in frames] + [(False, None)]
    cap.read.side_effect = read_returns
    return cap


@patch("peakframer.extractor.cv2.VideoCapture")
def test_extract_all_frames(mock_vc: MagicMock) -> None:
    fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
    mock_vc.return_value = _make_mock_cap(fake_frames)

    result = extract_frames(Path("fake.mp4"), sample_rate=1)
    assert len(result) == 10
    assert all(isinstance(f, ExtractedFrame) for f in result)


@patch("peakframer.extractor.cv2.VideoCapture")
def test_extract_sample_rate(mock_vc: MagicMock) -> None:
    fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]
    mock_vc.return_value = _make_mock_cap(fake_frames)

    result = extract_frames(Path("fake.mp4"), sample_rate=2)
    assert len(result) == 5


@patch("peakframer.extractor.cv2.VideoCapture")
def test_extract_invalid_video(mock_vc: MagicMock) -> None:
    cap = MagicMock()
    cap.isOpened.return_value = False
    mock_vc.return_value = cap

    with pytest.raises(ValueError, match="Could not open video"):
        extract_frames(Path("nonexistent.mp4"))


@patch("peakframer.extractor.cv2.VideoCapture")
def test_get_video_meta(mock_vc: MagicMock) -> None:
    fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]
    mock_vc.return_value = _make_mock_cap(fake_frames, fps=30.0)

    meta = get_video_meta(Path("fake.mp4"))
    assert meta.total_frames == 30
    assert meta.fps == 30.0
    assert meta.width == 640
    assert meta.height == 480
    assert meta.duration_seconds == pytest.approx(1.0)


def test_suggest_sample_rate_basic() -> None:
    assert suggest_sample_rate(total_frames=500, count=10) == 10


def test_suggest_sample_rate_never_zero() -> None:
    assert suggest_sample_rate(total_frames=5, count=10) == 1
