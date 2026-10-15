"""Incremental annotated video encoding."""
from __future__ import annotations

from pathlib import Path
import logging
import os
import time
import cv2

from src.video.io import VideoReadError, video_metadata
from src.video.async_reader import AsyncVideoReader

LOGGER = logging.getLogger(__name__)


def _smooth_path(points, samples_per_segment=4):
    """Return a gently interpolated Catmull-Rom path through sparse samples."""
    if len(points) < 3:
        return points
    smooth = []
    extended = [points[0], *points, points[-1]]
    for index in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[index - 1:index + 3]
        for step in range(samples_per_segment):
            t = step / samples_per_segment
            t2, t3 = t * t, t * t * t
            x = .5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                      + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                      + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = .5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                      + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                      + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            smooth.append((int(round(x)), int(round(y))))
    smooth.append(points[-1])
    return smooth


def _annotate(frame, results, timestamp_seconds, *, side_by_side=False, thermal_feed=False,
              show_trajectories=True, trajectory_sample_every=5):
    output = frame.copy()
    if show_trajectories:
        for row in results:
            trajectory = (row.get("trajectory") or [])[-20:]
            bbox = row.get("bbox_thermal") if thermal_feed else row.get("bbox")
            points = []
            for item in trajectory:
                box = item.get("bbox") if isinstance(item, dict) else None
                if box and len(box) == 4:
                    points.append((int(round((float(box[0]) + float(box[2])) / 2)),
                                   int(round(float(box[3])))))
                elif isinstance(item, dict) and len(item.get("center", [])) == 2:
                    points.append(tuple(map(int, item["center"])))
            if bbox and len(bbox) == 4:
                points.append((int(round((float(bbox[0]) + float(bbox[2])) / 2)),
                               int(round(float(bbox[3])))))
            color = (255, 180, 0)
            for start, end in zip(points, points[1:]):
                cv2.line(output, start, end, color, 1, cv2.LINE_8)
    for row in results:
        bbox = row.get("bbox_thermal") if thermal_feed else row.get("bbox")
        bbox = bbox or (row.get("bbox") if thermal_feed else None)
        if bbox:
            x1, y1, x2, y2 = map(int, bbox)
            color = ((0, 0, 255) if row.get("alert_status") == "alert" else
                     (0, 190, 255) if row.get("needs_review") else (0, 220, 0))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            posture = row.get("posture_thermal") if thermal_feed else row.get("posture_rgb")
            if thermal_feed:
                posture = posture or row.get("posture_rgb")
            track_id = row.get("display_id") or row.get("track_id") or "object"
            # Keep output readable in crowded scenes; detailed measurements stay
            # in the per-person dashboard review.
            label = f"ID {track_id}"
            if posture and posture != "unknown":
                label += f" | {posture.upper()}"
            if row.get("alert_status") == "alert":
                label += " ALERT"
            elif row.get("needs_review"):
                label += " REVIEW"
            face_landmarks = row.get("thermal_face_landmarks")
            if thermal_feed and face_landmarks and face_landmarks.get("landmarks") and row.get("bbox_thermal"):
                points = face_landmarks["landmarks"]
                bx1, by1, _, _ = map(int, row["bbox_thermal"])
                height, width = output.shape[:2]
                for point in points:
                    px, py = bx1 + int(point[0]), by1 + int(point[1])
                    if len(point) >= 2 and 0 <= px < width and 0 <= py < height:
                        cv2.circle(output, (px, py), 1, (255, 255, 0), -1)
            cv2.putText(output, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        for obj in row.get("threat_objects", []):
            ox1, oy1, ox2, oy2 = map(int, obj["bbox"])
            cv2.rectangle(output, (ox1, oy1), (ox2, oy2), (255, 0, 255), 2)
            cv2.putText(output, f"{obj['label']} {obj['confidence']:.2f}",
                        (ox1, max(18, oy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 2)
        if side_by_side and row.get("bbox_thermal"):
            half = output.shape[1] // 2
            tx1, ty1, tx2, ty2 = map(int, row["bbox_thermal"])
            cv2.rectangle(output, (half + tx1, ty1), (half + tx2, ty2), (255, 255, 255), 2)
        event = row.get("alert_event")
        if event:
            cv2.putText(output, f"{event.get('level')}: {event.get('reason')}",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    cv2.putText(output, f"t={timestamp_seconds:.3f}s", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .6, (255, 255, 255), 2)
    return output


def export_annotated_video(source_path, output_path, pipeline, *, mode="Performance",
                           progress_callback=None, side_by_side=False,
                           estimated_thermal_output_path=None, event_callback=None,
                           input_kind="rgb", show_trajectories=True):
    """Process and encode frames sequentially; source/output frames are not accumulated."""
    source_path, output_path = Path(source_path), Path(output_path)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"Could not open video: {source_path}")
    writer = None
    thermal_writer = None
    reader = None
    thermal_output_path = Path(estimated_thermal_output_path) if estimated_thermal_output_path else None
    succeeded = False
    try:
        info = video_metadata(capture)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))
        if not writer.isOpened():
            writer.release()
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"),
                                     info.fps, (info.width, info.height))
        if not writer.isOpened():
            raise VideoReadError("No MP4 writer is available for the requested output path")
        if thermal_output_path is not None:
            thermal_output_path.parent.mkdir(parents=True, exist_ok=True)
            thermal_writer = cv2.VideoWriter(str(thermal_output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             info.fps, (info.width, info.height))
            if not thermal_writer.isOpened():
                thermal_writer.release()
                thermal_writer = cv2.VideoWriter(str(thermal_output_path), cv2.VideoWriter_fourcc(*"avc1"),
                                                 info.fps, (info.width, info.height))
            if not thermal_writer.isOpened():
                raise VideoReadError("No MP4 writer is available for the estimated thermal output")
        pipeline.reset_video_state()
        pipeline.set_processing_mode(mode)
        reader = AsyncVideoReader(
            str(source_path), queue_size=int(os.getenv("RAVEN_DECODE_QUEUE_SIZE", "8")),
            drop_oldest=False,
        ).start()
        count = 0
        processing_errors = 0
        processing_error_samples = []
        track_ids = set()
        track_summaries = {}
        chronological_ids = {}
        threat_totals = {}
        event_ids = set()
        events = []
        max_concurrent_tracks = 0
        inference_seconds = 0.0
        stage_latency_totals = {}
        stage_latency_samples = 0
        thermal_output_enabled = thermal_writer is not None or input_kind == "thermal-palette"
        loop_started = time.perf_counter()
        while True:
            packet = reader.read(timeout=None)
            if packet is None:
                if reader.error:
                    raise VideoReadError(f"Video decode failed: {reader.error}")
                break
            frame = packet.frame
            timestamp = packet.timestamp_seconds
            frame_started = time.perf_counter()
            try:
                estimated_thermal = None
                if input_kind == "thermal-palette":
                    if frame.ndim == 3 and frame.shape[2] == 3:
                        thermal_display = frame.copy()
                    else:
                        thermal_gray = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
                        thermal_display = cv2.applyColorMap(thermal_gray, cv2.COLORMAP_INFERNO)
                    results = pipeline.process_rgb_frame(frame, metadata={
                        "frame_index": count, "video_timestamp_seconds": timestamp,
                        "thermal_source": "thermal_palette", "temperature_calibrated": False})
                    for row in results:
                        row["source"] = "thermal-palette"
                        row["thermal_sensor_available"] = False
                        row["thermal_temperature_c"] = None
                        row["temperature_calibrated"] = False
                    estimated_thermal = thermal_display
                elif thermal_writer is not None:
                    results, estimated_thermal = pipeline.process_derived_thermal_frame(frame, metadata={
                        "frame_index": count, "video_timestamp_seconds": timestamp,
                        "thermal_source": "estimated_from_rgb", "temperature_calibrated": False})
                elif side_by_side:
                    split = frame.shape[1] // 2
                    if split < 1:
                        raise VideoReadError("Video is too narrow for side-by-side RGB/thermal splitting")
                    results = pipeline.process_frame_pair(frame[:, :split], frame[:, split:], metadata={
                        "thermal_source": "thermal_imaging", "temperature_calibrated": False,
                        "frame_index": count, "video_timestamp_seconds": timestamp})
                else:
                    results = pipeline.process_rgb_frame(frame, metadata={
                        "frame_index": count, "video_timestamp_seconds": timestamp})
            except Exception as exc:
                processing_errors += 1
                if len(processing_error_samples) < 5:
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    processing_error_samples.append({"frame_index": count,
                                                     "timestamp_seconds": round(timestamp, 3),
                                                     "error": error})
                    LOGGER.warning("video frame inference failed at frame %s: %s",
                                   count, error, exc_info=True)
                # Preserve tracker/Re-ID history across a transient frame error.
                # BoT-SORT's lost-track buffer bridges the skipped frame; resetting
                # here would turn every isolated inference error into an ID split.
                results = []
                estimated_thermal = None
            inference_seconds += time.perf_counter() - frame_started
            frame_stats = getattr(pipeline, "last_frame_stats", {}) or {}
            for stage, latency_ms in (frame_stats.get("stage_latency_ms") or {}).items():
                stage_latency_totals[stage] = stage_latency_totals.get(stage, 0.0) + float(latency_ms)
            if frame_stats.get("stage_latency_ms"):
                stage_latency_samples += 1
            frame_tracks = {str(item.get("reid_canonical_track_id") or item["track_id"])
                            for item in results if item.get("track_id") is not None}
            for item in results:
                if item.get("track_id") is None:
                    continue
                canonical = str(item.get("reid_canonical_track_id") or item["track_id"])
                if canonical not in chronological_ids:
                    chronological_ids[canonical] = f"P{len(chronological_ids) + 1:04d}"
                item["display_id"] = chronological_ids[canonical]
            track_ids.update(str(item.get("tracker_id", item["track_id"]))
                             for item in results if item.get("track_id") is not None)
            max_concurrent_tracks = max(max_concurrent_tracks, len(frame_tracks))
            for item in results:
                track_id = str(item.get("tracker_id", item["track_id"])) if item.get("track_id") is not None else None
                canonical_track_id = str(item.get("reid_canonical_track_id") or track_id) if track_id is not None else None
                object_labels = []
                for obj in item.get("threat_objects", []):
                    label = str(obj.get("label", "unknown"))
                    object_labels.append(label)
                    aggregate = threat_totals.setdefault(label, {"count": 0, "max_confidence": 0.0})
                    aggregate["count"] += 1
                    aggregate["max_confidence"] = max(aggregate["max_confidence"],
                                                       float(obj.get("confidence", 0)))
                if canonical_track_id is not None:
                    aggregate = track_summaries.setdefault(canonical_track_id, {
                        "track_id": canonical_track_id, "first_seen_seconds": round(timestamp, 3),
                        "display_id": chronological_ids.get(canonical_track_id),
                        "last_seen_seconds": round(timestamp, 3), "frames_observed": 0,
                        "max_risk_score": 0, "needs_review": False,
                        "alert_status": "normal", "identity_source": item.get("identity_source", "tracker"),
                        "anomaly_method": item.get("anomaly_method"),
                        "anomaly_model_available": bool(item.get("anomaly_model_available", False)),
                        "signal_label": item.get("signal_label"),
                        "demo_label": item.get("demo_label"),
                        "model_validation_status": item.get("model_validation_status"),
                        "reid_status": item.get("reid_status", "unknown"),
                        "identity_timeline": [],
                        "reid_similarity": item.get("reid_similarity"),
                        "reid_method": item.get("reid_method"), "posture": None,
                        "detected_objects": {}, "evidence": [], "evidence_count": 0,
                        "alert_count": 0, "review_frames": 0, "alert_frames": 0,
                        "review_reasons": [],
                        "trajectory": [], "merged_tracker_ids": [],
                    })
                    if track_id and track_id != canonical_track_id and track_id not in aggregate["merged_tracker_ids"]:
                        aggregate["merged_tracker_ids"].append(track_id)
                    aggregate["last_seen_seconds"] = round(timestamp, 3)
                    aggregate["frames_observed"] += int(not item.get("predicted", False))
                    if item.get("model_validation_status") != "synthetic_unvalidated_demo":
                        aggregate["max_risk_score"] = max(aggregate["max_risk_score"],
                                                           int(item.get("risk_score", 0) or 0))
                    aggregate["needs_review"] = aggregate["needs_review"] or bool(item.get("needs_review"))
                    if not item.get("predicted", False):
                        if item.get("alert_status") == "review" and item.get("needs_review"):
                            aggregate["review_frames"] += 1
                            reason = ("experimental action cue" if
                                      item.get("model_validation_status") == "synthetic_unvalidated_demo" else
                                      "action/anomaly cue" if item.get("anomaly_model_available") else
                                      "motion cue" if item.get("anomaly_method") == "motion_heuristic_fallback" else
                                      "other configured review rule")
                            if reason not in aggregate["review_reasons"]:
                                aggregate["review_reasons"].append(reason)
                        elif item.get("alert_status") == "alert":
                            aggregate["alert_frames"] += 1
                    severity = {"low": 0, "normal": 0, "review": 1, "warning": 1, "alert": 2}
                    if severity.get(item.get("alert_status"), 0) > severity.get(aggregate["alert_status"], 0):
                        aggregate["alert_status"] = item.get("alert_status")
                    aggregate["identity_source"] = item.get("identity_source", aggregate["identity_source"])
                    reid_status = item.get("reid_status")
                    if reid_status == "appearance_match":
                        aggregate["reid_status"] = reid_status
                        aggregate["reid_similarity"] = item.get("reid_similarity")
                    elif (reid_status and reid_status != "unknown"
                          and aggregate["reid_status"] not in {"appearance_match", "tracker_continuity"}):
                        aggregate["reid_status"] = reid_status
                    if (reid_status == "appearance_match"
                            and item.get("reid_similarity") is not None):
                        aggregate["reid_similarity"] = item["reid_similarity"]
                    if item.get("reid_status") == "appearance_match":
                        aggregate["trajectory"] = (aggregate["trajectory"] + (item.get("trajectory") or []))[-20:]
                    elif item.get("trajectory"):
                        aggregate["trajectory"] = item["trajectory"][-20:]
                    aggregate["identity_timeline"].append({"timestamp": round(timestamp, 3),
                                                            "tracker_id": track_id,
                                                            "canonical_track_id": canonical_track_id,
                                                            "reid_status": item.get("reid_status"),
                                                            "similarity": item.get("reid_similarity")})
                    aggregate["identity_timeline"] = aggregate["identity_timeline"][-120:]
                    item_score = item.get("anomaly_score")
                    if item_score is not None:
                        # Keep a model score from being replaced or mixed with
                        # the earlier motion-only fallback score.
                        same_scoring_mode = (aggregate.get("demo_label") == item.get("demo_label") and
                                             aggregate.get("anomaly_method") == item.get("anomaly_method"))
                        if not same_scoring_mode:
                            aggregate["anomaly_score"] = int(item_score)
                        else:
                            aggregate["anomaly_score"] = max(
                                int(aggregate.get("anomaly_score", 0) or 0), int(item_score))
                        aggregate["anomaly_method"] = item.get("anomaly_method")
                        aggregate["anomaly_model_available"] = bool(item.get("anomaly_model_available", False))
                        aggregate["signal_label"] = item.get("signal_label")
                        aggregate["demo_label"] = item.get("demo_label")
                        aggregate["model_validation_status"] = item.get("model_validation_status")
                    posture = item.get("posture_rgb")
                    if posture and posture != "unknown":
                        aggregate["posture"] = posture
                    for label in object_labels:
                        value = next((obj for obj in item.get("threat_objects", [])
                                      if str(obj.get("label", "unknown")) == label), {})
                        previous = aggregate["detected_objects"].get(label, 0.0)
                        aggregate["detected_objects"][label] = max(previous, float(value.get("confidence", 0)))
                    evidence = item.get("evidence") or {}
                    if evidence:
                        aggregate["evidence_count"] += 1
                        aggregate["evidence"].append(evidence)
                        aggregate["evidence"] = aggregate["evidence"][-20:]
                event = item.get("alert_event")
                if not event:
                    continue
                event_id = event.get("event_id") or f"{event.get('track_id')}:{event.get('reason')}:{count}"
                if event_id not in event_ids and len(events) < 500:
                    event_ids.add(event_id)
                    events.append({"event_id": event_id, "timestamp_seconds": round(timestamp, 3),
                                   "track_id": canonical_track_id or event.get("track_id"), "level": event.get("level"),
                                   "display_id": chronological_ids.get(canonical_track_id),
                                   "model_validation_status": event.get("model_validation_status"),
                                   "demo_label": event.get("demo_label"),
                                   "signal_label": event.get("signal_label"),
                                   "operator_review_required": bool(event.get("operator_review_required", True)),
                                   "confirmed_threat": False,
                                   "reason": event.get("reason"), "risk_score": event.get("risk_score"),
                                   "thresholds": event.get("thresholds"),
                                   "threat_confidence": max((float(obj.get("confidence", 0))
                                                             for obj in event.get("threat_objects", [])),
                                                            default=None),
                                   "threats": sorted({str(obj.get("label", "unknown"))
                                                      for obj in event.get("threat_objects", [])})})
                    if canonical_track_id is not None and canonical_track_id in track_summaries:
                        track_summaries[canonical_track_id]["alert_count"] += 1
                    if event_callback is not None:
                        event_callback(events[-1])
            writer.write(_annotate(frame, results, timestamp, side_by_side=side_by_side,
                                   show_trajectories=show_trajectories))
            if thermal_writer is not None and estimated_thermal is not None:
                thermal_writer.write(_annotate(estimated_thermal, results, timestamp, thermal_feed=True,
                                               show_trajectories=show_trajectories))
            count += 1
            progress_interval = max(1, min(25, info.frame_count // 20))
            if progress_callback is not None and (count % progress_interval == 0 or count == info.frame_count):
                progress_callback(count, info.frame_count)
        if count == 0:
            raise VideoReadError("The video contained no decodable frames")
        pipeline_capabilities = pipeline.analysis_capabilities()
        alert_policy = {"warning_threshold": pipeline.alert_generator.warning_threshold,
                        "alert_threshold": pipeline.alert_generator.alert_threshold,
                        "threat_confidence_threshold": pipeline.alert_generator.threat_threshold}
        tracks = sorted(track_summaries.values(),
                        key=lambda item: (item["first_seen_seconds"], item["display_id"] or ""))
        stage_profile = {stage: round(total / max(stage_latency_samples, 1), 1)
                         for stage, total in stage_latency_totals.items()}
        threat_capabilities = pipeline_capabilities["threat_detection"]
        maximum_score = max((item["max_risk_score"] for item in tracks), default=0)
        explanation = None
        if not events:
            details = []
            supported = threat_capabilities.get("supported_labels", [])
            unsupported_firearms = [label for label in threat_capabilities.get("unsupported_labels", [])
                                    if label in {"gun", "weapon", "firearm", "handgun", "pistol",
                                                 "revolver", "rifle", "shotgun"}]
            if threat_totals:
                detected = ", ".join(sorted(threat_totals))
                details.append(f"Object detections ({detected}) were recorded but no alert event was emitted; check cooldown and notification policy")
            elif supported:
                details.append("No configured object class crossed its confidence threshold")
            else:
                details.append("No configured threat-object class is supported by the loaded model")
            details.append(f"peak review priority {maximum_score} stayed below warning threshold {alert_policy['warning_threshold']}")
            if unsupported_firearms:
                details.append("firearm classes are not supported by this model")
            if not pipeline_capabilities["anomaly_detection"]["available"]:
                details.append("action/anomaly detection is not configured")
            explanation = ". ".join(details) + "."
        succeeded = True
        return {"output_path": str(output_path),
                "estimated_thermal_output_path": str(thermal_output_path) if thermal_output_path else None,
                "frames_written": count, "unique_tracks": len(track_summaries),
                "max_concurrent_tracks": max_concurrent_tracks, "events": events,
                "unique_tracker_ids": len(track_ids),
                "event_count": len(events), "tracks": tracks[:100],
                "threat_detections": [{"label": label, **values}
                                      for label, values in sorted(threat_totals.items())],
                "alert_policy": alert_policy, "no_alert_explanation": explanation,
                "capabilities": pipeline_capabilities,
                "compute_device": pipeline.device_name,
                "inference_seconds": round(inference_seconds, 3),
                "inference_fps": round(count / max(inference_seconds, .001), 3),
                "stage_profile_ms": stage_profile,
                "export_seconds": round(time.perf_counter() - loop_started, 3),
                "available_models": {"rgb_detector": True,
                                     "threat_detector": bool(threat_capabilities.get("model_loaded")),
                                     "thermal_estimate": thermal_output_enabled,
                                     "pose_classifier": True,
                                     "appearance_reid": bool(pipeline_capabilities["appearance_reid"]["available"]),
                                     "confidence_fusion": bool(pipeline_capabilities["confidence_fusion"]["available"]),
                                     "motion_anomaly_heuristic": bool(pipeline_capabilities["anomaly_detection"]["available"]),
                                     "temporal_cross_modal_fusion": bool(getattr(pipeline, "temporal_fusion_runtime", None)),
                                     "temporal_fusion_scoring_eligible": bool(getattr(pipeline, "temporal_fusion_scoring_eligible", False)),
                                     "synthetic_temporal_fusion_demo": bool(getattr(pipeline, "temporal_fusion_checkpoint_synthetic", False)),
                                     "action_anomaly_model": bool(getattr(pipeline, "action_runtime", None))},
                "width": info.width, "height": info.height, "fps": info.fps,
                "duration_seconds": count / info.fps,
                "truncated": count < info.frame_count,
                "processing_errors": processing_errors,
                "processing_error_samples": processing_error_samples}
    finally:
        if reader is not None:
            reader.close()
        capture.release()
        if writer is not None:
            writer.release()
        if thermal_writer is not None:
            thermal_writer.release()
        pipeline.reset_video_state()
        if not succeeded and output_path.exists():
            output_path.unlink()
        if not succeeded and thermal_output_path is not None and thermal_output_path.exists():
            thermal_output_path.unlink()
