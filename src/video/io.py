"""Sequential video metadata and frame iteration without retaining frame lists."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass
class VideoFrame:
    index: int
    timestamp_seconds: float
    image: object


class VideoReadError(RuntimeError):
    pass


def video_metadata(capture) -> VideoMetadata:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if width <= 0 or height <= 0 or count <= 0 or fps <= 0:
        raise VideoReadError("Video metadata is incomplete; dimensions, frame count, or FPS is unavailable")
    return VideoMetadata(width, height, fps, count, count / fps)


def iter_video_frames(path: str | Path) -> Iterator[VideoFrame]:
    """Yield one decoded frame at a time, using codec timestamps when available."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"Could not open video: {path}")
    try:
        info = video_metadata(capture)
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                # OpenCV does not reliably distinguish damaged frame from EOF.
                # Stop cleanly after the last valid frame; never retain frames.
                break
            raw_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
            timestamp = raw_ms / 1000 if raw_ms > 0 else index / info.fps
            yield VideoFrame(index, timestamp, image)
            index += 1
    finally:
        capture.release()
