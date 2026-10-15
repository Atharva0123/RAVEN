"""Transparent, motion-only anomaly fallback for demos without an action model.

This produces an explainable review cue, not a learned action label or threat
probability. It deliberately uses only track motion and dwell observations.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time


@dataclass
class _TrackWindow:
    points: deque = field(default_factory=lambda: deque(maxlen=90))


class MotionAnomalyFallback:
    """Bounded per-track speed/dwell heuristic with stable 0..100 output."""
    def __init__(self, *, max_tracks=512, history_seconds=30.0):
        self.max_tracks = int(max_tracks)
        self.history_seconds = float(history_seconds)
        self._tracks = {}

    def update(self, track_id, center, timestamp=None, frame_shape=None):
        now = float(timestamp if timestamp is not None else time.monotonic())
        x, y = map(float, center)
        state = self._tracks.get(str(track_id))
        if state is None:
            if len(self._tracks) >= self.max_tracks:
                self._tracks.pop(next(iter(self._tracks)))
            state = self._tracks[str(track_id)] = _TrackWindow()
        points = state.points
        if points and now <= points[-1][0]:
            now = points[-1][0] + 1e-3
        if points and frame_shape:
            h, w = frame_shape[:2]
            scale = max(1.0, math.hypot(w, h))
        else:
            scale = 1000.0
        points.append((now, x, y))
        while len(points) > 2 and now - points[0][0] > self.history_seconds:
            points.popleft()
        duration = max(0.0, points[-1][0] - points[0][0])
        speed = 0.0
        if len(points) > 1 and duration > 0:
            distance = sum(math.dist(a[1:], b[1:]) for a, b in zip(points, list(points)[1:]))
            speed = distance / duration / scale
        displacement = math.dist(points[0][1:], points[-1][1:]) / scale if duration else 0.0
        # Dimensionless cues: rapid movement and sustained presence with little
        # displacement. Short histories avoid falsely calling a new track loitering.
        speed_cue = min(1.0, max(0.0, (speed - 0.025) / 0.12))
        dwell_cue = (min(1.0, max(0.0, (duration - 8.0) / 22.0))
                     * min(1.0, max(0.0, (0.045 - displacement) / 0.04)))
        score = int(round(100.0 * max(speed_cue, 0.65 * dwell_cue)))
        reasons = []
        if speed_cue > 0:
            reasons.append("unusually fast image-plane movement")
        if dwell_cue >= 0.25:
            reasons.append("prolonged presence with little movement")
        return {"score": score, "method": "motion_heuristic_fallback",
                "model_available": False, "review_cue_only": True,
                "signals": {"normalized_image_speed": round(speed, 5),
                            "dwell_seconds": round(duration, 2),
                            "normalized_displacement": round(displacement, 5)},
                "reason": "; ".join(reasons) if reasons else "no configured motion cue"}

