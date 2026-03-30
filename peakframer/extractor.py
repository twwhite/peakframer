"""Video decoding and frame extraction.

Opens video files, reads metadata, and samples frames at a configurable rate.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from peakframer.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedFrame:
    index: int
    timestamp_ms: float
    image: np.ndarray


@dataclass
class VideoMeta:
    path: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_seconds: float


def get_video_meta(path: Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return VideoMeta(
            path=path,
            total_frames=total,
            fps=fps,
            width=w,
            height=h,
            duration_seconds=total / fps,
        )
    finally:
        cap.release()


def extract_frames(path: Path, sample_rate: int = 1) -> list[ExtractedFrame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames: list[ExtractedFrame] = []
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                frames.append(ExtractedFrame(index=idx, timestamp_ms=ts, image=frame))
            idx += 1
    finally:
        cap.release()

    logger.info(f"Extracted {len(frames)} candidate frames from {path.name}")
    return frames


def suggest_sample_rate(total_frames: int, count: int, multiplier: int = 5) -> int:
    target_candidates = count * multiplier
    return max(1, total_frames // target_candidates)
