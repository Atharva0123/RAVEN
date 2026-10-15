"""Strict, frontend-neutral RAVEN dashboard payload and JSON serializer."""
import json
import math
from datetime import datetime, timezone


def _finite(value, default=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def generate_dashboard_payload(results, *, frame_index=0, timestamp_seconds=0.0,
                               processing_fps=None, features=None, thermal_mode="unavailable",
                               playback_state="playing"):
    """Build the stable sections used below the video panel in the dashboard."""
    rows = list(results or [])
    detections, logs = [], []
    for row in rows:
        bbox = row.get("bbox")
        track_id = row.get("track_id")
        objects = row.get("threat_objects") or []
        if bbox:
            detections.append({"track_id": str(track_id) if track_id is not None else None,
                               "label": "Person" if track_id is not None else "Detected object",
                               "bbox_xyxy": [_finite(v, 0.0) for v in bbox],
                               "confidence": _finite(row.get("rgb_confidence", row.get("confidence"))),
                               "predicted": bool(row.get("predicted", False)),
                               "source": str(row.get("source", "rgb"))})
        if track_id is not None:
            logs.append({"timestamp_seconds": _finite(row.get("video_timestamp_seconds"), timestamp_seconds),
                         "kind": "track_estimated" if row.get("predicted") else "person_detected",
                         "track_id": str(track_id),
                         "text": (f"Track {track_id} position estimated between detector runs" if row.get("predicted")
                                  else f"Person [ID: {track_id}] detected")})
            if row.get("occlusion_recovered_rgb") or row.get("occlusion_recovered_thermal"):
                logs.append({"timestamp_seconds": _finite(row.get("video_timestamp_seconds"), timestamp_seconds),
                             "kind": "track_recovered", "track_id": str(track_id),
                             "text": f"Track [ID: {track_id}] reacquired after a missed observation"})
            if row.get("anomaly_method") == "motion_heuristic_fallback":
                logs.append({"timestamp_seconds": _finite(row.get("video_timestamp_seconds"), timestamp_seconds),
                             "kind": "fallback_mode", "track_id": str(track_id),
                             "text": "PROTOTYPE_REVIEW_CUE: heuristic motion signal; not a trained action model"})
        for obj in objects:
            logs.append({"timestamp_seconds": timestamp_seconds, "kind": "object_detected",
                         "track_id": str(track_id) if track_id is not None else None,
                         "text": f"Configured object class detected: {obj.get('label', 'object')}"})
    logs.sort(key=lambda item: _finite(item.get("timestamp_seconds"), 0.0))
    person_rows = [row for row in rows if row.get("track_id") is not None]
    anomaly_values = [_finite(row.get("anomaly_score")) for row in person_rows]
    anomaly_values = [v for v in anomaly_values if v is not None]
    risk_values = [_finite(row.get("risk_score")) for row in person_rows]
    risk_values = [v for v in risk_values if v is not None]
    active_features = {"low_light_mode": any("low_light" in (r.get("image_quality") or {}).get("issues", []) for r in rows),
                       "occlusion_detected": any(bool(r.get("occlusion_recovered_rgb") or r.get("occlusion_recovered_thermal")) for r in rows),
                       "frame_skipping_active": any(bool(r.get("detector_skipped")) for r in rows)}
    active_features.update(features or {})
    fusion_scores = [_finite(r.get("fusion_score")) for r in person_rows]
    fusion_scores = [v for v in fusion_scores if v is not None]
    payload = {"schema_version": "1.0", "generated_at": datetime.now(timezone.utc).isoformat(),
        "frame": {"index": int(frame_index), "timestamp_seconds": _finite(timestamp_seconds, 0.0),
                  "playback_state": str(playback_state)},
        "detection_log": {"title": "Detection Log: What RAVEN sees", "items": logs,
            "status_message": ("Heuristic motion cue only; estimated thermal is RGB-derived visualization and is excluded from fusion."
                                             if any(r.get("anomaly_method") == "motion_heuristic_fallback" for r in rows)
                                             else "")},
        "live_metrics": {"title": "Live Metrics & Counts", "active_tracks": len({str(r.get('track_id')) for r in person_rows}),
                         "detections": len(detections), "processing_fps": _finite(processing_fps),
                         "features": active_features},
        "scoring_panel": {"title": "Scoring Panel",
            "anomaly": {"score": round(sum(anomaly_values)/len(anomaly_values), 1) if anomaly_values else None,
                        "method": next((r.get("anomaly_method") for r in person_rows if r.get("anomaly_method")), "unavailable"),
                        "signal_label": next((r.get("signal_label") for r in person_rows if r.get("signal_label")),
                                              "PROTOTYPE_REVIEW_CUE" if anomaly_values else "unavailable"),
                        "is_trained_model": any(bool(r.get("anomaly_model_available")) for r in person_rows),
                        "meaning": "Behavioral review cue; does not determine intent."},
            "risk": {"score": round(sum(risk_values)/len(risk_values), 1) if risk_values else None,
                     "method": next((r.get("risk_score_type") for r in person_rows if r.get("risk_score_type")), "unavailable"),
                     "meaning": "System review priority; not a probability of threat."},
            "fusion_consensus": {"score": round(sum(fusion_scores)/len(fusion_scores), 1) if fusion_scores else None,
                                 "method": next((r.get("fusion_score_type") for r in person_rows if r.get("fusion_score_type")), "unavailable"),
                                 "meaning": "Detection agreement/consensus; not anomaly or threat probability."}},
        "detections": detections,
        "thermal": {"mode": thermal_mode, "independent_sensor": thermal_mode == "real_sensor",
                    "warning": ("RGB-derived simulation; not independent thermal evidence." if thermal_mode == "simulated_from_rgb" else None)}}
    return payload


def serialize_dashboard_payload(payload):
    """JSON that rejects NaN/Infinity so strict browser parsers remain safe."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
