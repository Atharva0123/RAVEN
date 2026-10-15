"""Composable deployment loop for detector callbacks and dashboard sinks.

The detector/tracker remain injected frozen backbones. A bounded reader decouples
decode from inference; skipped detector frames use a bounded constant-velocity
box estimate. Estimates are marked so the UI never presents them as detections.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from src.dashboard.payload import generate_dashboard_payload
from src.thermal.simulator import ThermalSimulator
from src.tracking.trajectory_smoother import AdaptiveTrajectorySmoother
from src.video.async_reader import AsyncVideoReader


def _center(track):
    x1, y1, x2, y2 = track["bbox"]
    return ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)


@dataclass
class _BoxState:
    bbox: tuple
    velocity: tuple
    timestamp: float
    last_detection_timestamp: float


class BoundingBoxMotionInterpolator:
    """Short-horizon constant-velocity box estimator keyed by stable track ID."""
    def __init__(self, max_gap_seconds=0.75, velocity_smoothing=0.55):
        self.max_gap_seconds = float(max_gap_seconds)
        self.velocity_smoothing = float(velocity_smoothing)
        self.states = {}

    def update(self, rows, timestamp):
        out, active = [], set()
        now = float(timestamp)
        for row in rows or []:
            if "bbox" not in row:
                continue
            item = dict(row)
            track_id = str(item.get("track_id", len(out)))
            active.add(track_id)
            box = tuple(map(float, item["bbox"]))
            previous = self.states.get(track_id)
            velocity = (0.0, 0.0)
            if previous and now > previous.timestamp:
                old_cx = (previous.bbox[0] + previous.bbox[2]) / 2
                old_cy = (previous.bbox[1] + previous.bbox[3]) / 2
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                dt = now - previous.timestamp
                observed = ((cx - old_cx) / dt, (cy - old_cy) / dt)
                velocity = tuple(self.velocity_smoothing * observed[i] + (1-self.velocity_smoothing) * previous.velocity[i] for i in (0, 1))
            self.states[track_id] = _BoxState(box, velocity, now, now)
            item["track_id"] = track_id
            item["predicted"] = False
            out.append(item)
        for track_id, state in list(self.states.items()):
            age = now - state.last_detection_timestamp
            if track_id in active or age <= 0 or age > self.max_gap_seconds:
                if age > self.max_gap_seconds:
                    self.states.pop(track_id, None)
                continue
            dt = now - state.timestamp
            dx, dy = state.velocity[0] * dt, state.velocity[1] * dt
            box = tuple(state.bbox[i] + (dx if i in (0, 2) else dy) for i in range(4))
            self.states[track_id] = _BoxState(box, state.velocity, now, state.last_detection_timestamp)
            out.append({"track_id": track_id, "bbox": box, "conf": 0.0, "predicted": True,
                        "detector_skipped": True, "estimate_age_seconds": round(age, 3)})
        return out


def main_deployment_loop(video_source, *, detector, dashboard_sink, thermal_source=None,
                         thermal_detector=None, fusion=None, frame_stride=2, queue_size=8,
                         live_source=False, max_frames=None, simulate_thermal=False,
                         trained_fusion=False):
    """Process an RGB stream and emit schema-versioned dashboard payloads.

    detector(frame) returns tracked rows with ``bbox``, ``conf``, optional
    ``track_id``. A physical thermal source must be timestamp-synchronized by
    the caller; this generic loop conservatively does not pair unsynchronized
    streams by queue order. RGB-derived pseudo-thermal is marked simulated and
    can never be represented as independent sensor evidence.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    # `fusion` / `trained_fusion` remain accepted for caller compatibility, but
    # this loop does not implement synchronized thermal detection or pairing.
    smoother = AdaptiveTrajectorySmoother()
    simulator = ThermalSimulator()
    interpolator = BoundingBoxMotionInterpolator()
    histories, n, started, cached = {}, 0, time.perf_counter(), []
    thermal_mode = "real_sensor" if thermal_source is not None else "simulated_from_rgb" if simulate_thermal else "unavailable"
    with AsyncVideoReader(video_source, queue_size, drop_oldest=live_source) as reader:
        while max_frames is None or n < max_frames:
            packet = reader.read(timeout=2)
            if packet is None:
                break
            run_detector = packet.index % frame_stride == 0 or not cached
            if run_detector:
                detected = detector(packet.frame) or []
                cached = interpolator.update(detected, packet.timestamp_seconds)
            else:
                cached = interpolator.update([], packet.timestamp_seconds)
            pseudo = simulator.transform(packet.frame) if simulate_thermal and thermal_source is None else None
            thermal_frame = pseudo.frame if pseudo is not None else None
            rows = []
            for track in cached:
                tid = str(track.get("track_id"))
                box = track["bbox"]
                cx, cy = _center(track)
                conf = float(track.get("conf", 0.0))
                # This helper currently has no timestamp-synchronized thermal
                # detector callback. A pseudo view is correlated RGB and must
                # never be passed as a second modality, even in demo mode.
                score = conf * 100.0
                path = histories.setdefault(tid, [])
                path.append({"timestamp": packet.timestamp_seconds, "center": [cx, cy], "bbox": list(box)})
                path[:] = path[-300:]
                row = {"track_id": tid, "bbox": list(box), "rgb_confidence": conf,
                       "thermal_confidence": None,
                       "fusion_score": round(float(score), 1),
                       "fusion_score_type": "rgb_detector_confidence_not_cross_modal_consensus",
                       "thermal_sensor_available": False, "predicted": bool(track.get("predicted")),
                       "detector_skipped": not run_detector,
                       "trajectory": smoother.simplify(path, float(track.get("anomaly_score", 0) or 0)),
                       "frame_index": packet.index, "video_timestamp_seconds": packet.timestamp_seconds}
                if pseudo is not None:
                    row["thermal_mode"] = "simulated_from_rgb"
                rows.append(row)
            elapsed = max(1e-6, time.perf_counter() - started)
            payload = generate_dashboard_payload(rows, frame_index=packet.index,
                timestamp_seconds=packet.timestamp_seconds, processing_fps=(n+1)/elapsed,
                features={"frame_skipping_active": frame_stride > 1,
                          "thermal_simulation_active": pseudo is not None,
                          "fusion_layer_trained": False},
                thermal_mode=thermal_mode)
            dashboard_sink(payload)
            n += 1
    return {"processed_frames": n, "thermal_mode": thermal_mode,
            "note": "Pseudo-thermal input is RGB-derived and not independent IR evidence." if simulate_thermal and thermal_source is None else None}
