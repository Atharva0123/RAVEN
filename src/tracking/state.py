"""Bounded, timestamp-aware state for tracked objects and people."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackState:
    track_id: str
    bbox: list[float]
    confidence: float
    first_seen: float
    last_seen: float
    class_id: int = 0
    class_name: str = "object"
    observed_hits: int = 1
    status: str = "VISIBLE"
    velocity: tuple[float, float] = (0.0, 0.0)
    acceleration: tuple[float, float] = (0.0, 0.0)
    pose: dict | None = None
    pose_history: deque = field(default_factory=lambda: deque(maxlen=30))
    behavior_history: deque = field(default_factory=lambda: deque(maxlen=30))
    anomaly_history: deque = field(default_factory=lambda: deque(maxlen=30))
    trajectory: deque = field(default_factory=lambda: deque(maxlen=20))

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def observe(self, detection: dict, timestamp: float, *, max_speed: float = 5000.0):
        new_box = [float(value) for value in detection["bbox"]]
        old_center = self._center(self.bbox)
        new_center = self._center(new_box)
        elapsed = max(1e-3, timestamp - self.last_seen)
        previous_velocity = self.velocity
        vx = max(-max_speed, min(max_speed, (new_center[0] - old_center[0]) / elapsed))
        vy = max(-max_speed, min(max_speed, (new_center[1] - old_center[1]) / elapsed))
        self.acceleration = ((vx - previous_velocity[0]) / elapsed,
                             (vy - previous_velocity[1]) / elapsed)
        self.velocity = (vx, vy)
        self.status = "REACQUIRED" if self.status in {"OCCLUDED", "LOST"} else "VISIBLE"
        self.bbox = new_box
        self.confidence = float(detection.get("conf", self.confidence))
        self.last_seen = timestamp
        self.observed_hits += 1
        self.trajectory.append({"timestamp": float(timestamp), "center": [float((new_box[0] + new_box[2]) / 2), float(new_box[3])],
                                "bbox": list(new_box)})
        self.class_id = int(detection.get("class_id", self.class_id))
        self.class_name = str(detection.get("class_name", self.class_name))

    def snapshot(self, timestamp: float, *, extrapolate: bool = False) -> dict:
        elapsed = max(0.0, timestamp - self.last_seen)
        box = list(self.bbox)
        if extrapolate and elapsed:
            dx, dy = self.velocity[0] * elapsed, self.velocity[1] * elapsed
            box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
        return {"track_id": self.track_id, "bbox": box, "conf": self.confidence,
                "class_id": self.class_id, "class_name": self.class_name,
                "first_seen": self.first_seen, "last_seen": self.last_seen,
                "track_age": max(0.0, timestamp - self.first_seen),
                "time_since_detection": elapsed, "observed_hits": self.observed_hits,
                "velocity": self.velocity,
                "acceleration": self.acceleration, "occlusion_state": self.status,
                "pose": self.pose}


class TrackStateManager:
    """Maintain finite histories and expire stale tracks automatically."""
    def __init__(self, *, temporal_window: int = 30, stale_after_seconds: float = 5.0):
        if temporal_window < 1 or stale_after_seconds <= 0:
            raise ValueError("temporal_window and stale_after_seconds must be positive")
        self.temporal_window = int(temporal_window)
        self.stale_after_seconds = float(stale_after_seconds)
        self.tracks: dict[str, TrackState] = {}

    def update(self, detections: list[dict], timestamp: float) -> list[dict]:
        self._expire(timestamp)
        output = []
        active = set()
        for index, detection in enumerate(detections):
            key = str(detection.get("track_id") if detection.get("track_id") is not None
                      else f"det-{index}")
            active.add(key)
            state = self.tracks.get(key)
            if state is None:
                state = TrackState(key, [float(v) for v in detection["bbox"]],
                                   float(detection.get("conf", 0)), timestamp, timestamp,
                                   int(detection.get("class_id", 0)),
                                   str(detection.get("class_name", "object")),
                                   pose_history=deque(maxlen=self.temporal_window),
                                   behavior_history=deque(maxlen=self.temporal_window),
                                   anomaly_history=deque(maxlen=self.temporal_window),
                                   trajectory=deque(maxlen=20))
                center = TrackState._center(state.bbox)
                state.trajectory.append({"timestamp": float(timestamp),
                                         "center": [float((state.bbox[0] + state.bbox[2]) / 2), float(state.bbox[3])],
                                         "bbox": list(state.bbox)})
                self.tracks[key] = state
            else:
                state.observe(detection, timestamp)
            row = state.snapshot(timestamp)
            row["trajectory"] = list(state.trajectory)
            row.update(detection)
            row["track_id"] = key
            row["occlusion_state"] = state.status
            output.append(row)
        for key, state in self.tracks.items():
            if key not in active:
                age = timestamp - state.last_seen
                state.status = "OCCLUDED" if age <= self.stale_after_seconds else "LOST"
                if state.status == "OCCLUDED":
                    row = state.snapshot(timestamp, extrapolate=True)
                    row["trajectory"] = list(state.trajectory)
                    row["conf"] = state.confidence * max(0.0, 1.0 - age / self.stale_after_seconds)
                    row["occlusion_state"] = "OCCLUDED"
                    row["predicted"] = True
                    output.append(row)
                else:
                    row = state.snapshot(timestamp)
                    row["trajectory"] = list(state.trajectory)
                    row["occlusion_state"] = "LOST"
                    row["predicted"] = False
                    output.append(row)
        return output

    def predict(self, timestamp: float) -> list[dict]:
        self._expire(timestamp)
        output = []
        for state in self.tracks.values():
            age = max(0.0, timestamp - state.last_seen)
            state.status = "OCCLUDED" if age <= self.stale_after_seconds else "LOST"
            if state.status == "LOST":
                continue
            row = state.snapshot(timestamp, extrapolate=True)
            row["trajectory"] = list(state.trajectory)
            row["conf"] = state.confidence * max(0.0, 1.0 - age / self.stale_after_seconds)
            row["occlusion_state"] = "OCCLUDED"
            row["predicted"] = True
            output.append(row)
        return output

    def record_pose(self, track_id: str, keypoints, label: str, confidence: float,
                    timestamp: float, bbox=None):
        state = self.tracks.get(str(track_id))
        if state is None:
            return
        normalized = None
        if keypoints is not None and bbox is not None:
            x1, y1 = float(bbox[0]), float(bbox[1])
            width = max(1.0, float(bbox[2]) - x1)
            height = max(1.0, float(bbox[3]) - y1)
            normalized = []
            for point in keypoints:
                if len(point) < 2:
                    continue
                px = min(1.0, max(0.0, (float(point[0]) - x1) / width))
                py = min(1.0, max(0.0, (float(point[1]) - y1) / height))
                point_conf = float(point[2]) if len(point) > 2 else float(confidence)
                normalized.append([px, py, min(1.0, max(0.0, point_conf))])
        pose = {"keypoints": keypoints, "label": label, "confidence": float(confidence),
                "normalized_keypoints": normalized, "timestamp": float(timestamp)}
        state.pose = pose
        state.pose_history.append({"keypoints": normalized, "label": label,
                                   "confidence": float(confidence), "timestamp": float(timestamp)})
        if label and label != "unknown":
            state.behavior_history.append({"label": label, "timestamp": float(timestamp)})

    def _expire(self, timestamp: float):
        cutoff = timestamp - 2 * self.stale_after_seconds
        self.tracks = {key: value for key, value in self.tracks.items()
                       if value.last_seen >= cutoff}

    def clear(self):
        self.tracks.clear()
