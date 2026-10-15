"""Validated RGB/thermal processing pipeline for RAVEN."""
from __future__ import annotations
import base64
import binascii
from io import BytesIO
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from src.tracking.tracker import SentryTracker
from src.reid.reid_buffer import ReIDBuffer
from src.posture.rules import PostureClassifier
from src.fusion.consensus import ConsensusFusion
from src.fusion.confidence_head import ConfidenceFusionHead, PostHocRiskHead
from src.fusion.learned_consensus import LearnedConsensusFusion, IDENTITY_TIERS
from src.fusion.temporal_cross_modal import (TemporalCrossModalFusion,
                                             TemporalCrossModalRuntime)
from src.detection.threats import ThreatObjectDetector
from src.alerts.generator import AlertGenerator
from src.image_quality import assess_image_quality, compare_detection_sets, estimate_frame_quality
from src.tracking.state import TrackStateManager
from src.tracking.trajectory_smoother import AdaptiveTrajectorySmoother
from src.anomaly.action_model import (TemporalPoseAnomalyHead, TrackAnomalyRuntime,
                                      encode_track_features)
from src.anomaly.heuristic_fallback import MotionAnomalyFallback

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = max(0, a[2]-a[0]) * max(0, a[3]-a[1]) + max(0, b[2]-b[0]) * max(0, b[3]-b[1]) - inter
    return inter / union if union else 0.0


def _distance(a, b):
    ca = ((a[0]+a[2])/2, (a[1]+a[3])/2)
    cb = ((b[0]+b[2])/2, (b[1]+b[3])/2)
    scale = max(1.0, math.hypot(a[2]-a[0], a[3]-a[1]))
    return math.dist(ca, cb) / scale


def _motion_anomaly(track, person_height):
    """Return a bounded, explainable motion-outlier score (not an action label)."""
    velocity = track.get("velocity") or (0.0, 0.0)
    acceleration = track.get("acceleration") or (0.0, 0.0)
    height = max(40.0, float(person_height))
    speed_body_lengths = math.hypot(float(velocity[0]), float(velocity[1])) / height
    acceleration_body_lengths = math.hypot(float(acceleration[0]), float(acceleration[1])) / height
    speed_signal = min(1.0, max(0.0, (speed_body_lengths - 2.0) / 3.0))
    acceleration_signal = min(1.0, max(0.0, (acceleration_body_lengths - 10.0) / 24.0))
    return int(round(100 * max(speed_signal, .65 * acceleration_signal))), {
        "relative_speed_body_lengths_per_second": round(speed_body_lengths, 3),
        "relative_acceleration_body_lengths_per_second_squared": round(acceleration_body_lengths, 3),
    }


class RavenPipeline:
    """Synchronized RGB and thermal processing; dependencies are injectable."""
    def __init__(self, config_path=None, *, rgb_tracker=None, thermal_tracker=None,
                 pose_classifier=None, rgb_reid=None, thermal_reid=None, fusion=None,
                 alert_store=None, object_detector=None, enable_thermal=True,
                 quality_detector=None, disable_threat_detector=False, dashboard_preview=False,
                 input_modality="rgb", temporal_fusion_model=None):
        self.config_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "models.yaml"
        if not self.config_path.is_absolute():
            candidate = Path.cwd() / self.config_path
            self.config_path = candidate if candidate.exists() else PROJECT_ROOT / self.config_path
        with self.config_path.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream) or {}
        detect = self.config.get("detection", {})
        pose = self.config.get("pose", {})
        self.threshold = float(detect.get("conf_threshold", .45))
        object_config = self.config.get("threat_detection", {})
        # Do not let a lower weapon-object threshold leak into person tracking.
        self.rgb_detection_threshold = self.threshold
        self.image_size = int(detect.get("image_size", 640))
        self.iou_threshold = float(detect.get("iou_threshold", .45))
        self.association_iou = float(detect.get("association_iou", .05))
        self.max_center_distance = float(detect.get("max_center_distance", 1.5))
        self.enable_thermal = bool(enable_thermal)
        self.dashboard_preview = bool(dashboard_preview)
        self.input_modality = "thermal" if str(input_modality).lower() in {"thermal", "ir", "thermal-palette"} else "rgb"
        self.person_filter = self.config.get("person_filter", {})
        self.rgb_detection_threshold = (float(self.person_filter.get("thermal_min_confidence", self.threshold))
                                        if self.input_modality == "thermal" else self.threshold)
        processing = self.config.get("processing", {})
        self.temporal_window = int(processing.get("temporal_window", 32))
        self.track_states = TrackStateManager(
            temporal_window=self.temporal_window,
            stale_after_seconds=float(processing.get("track_stale_seconds", 5.0)))
        self.thermal_track_states = TrackStateManager(
            temporal_window=self.temporal_window,
            stale_after_seconds=float(processing.get("track_stale_seconds", 5.0)))
        self.processing_modes = processing.get("modes", {
            "Accuracy": {"detection_interval": 1, "pose_interval": 1},
            "Balanced": {"detection_interval": 1, "pose_interval": 2},
            "Performance": {"detection_interval": 3, "pose_interval": 3,
                            "threat_interval": 2},
        })
        self.processing_mode = "Accuracy"
        self.processing_settings = dict(self.processing_modes[self.processing_mode])
        self.stage_latency_ema = {}
        self._frame_index = 0
        self._last_detection_frame = None
        self._frames_since_detector = 0
        self._processed_frame_count = 0
        # Re-run the compact preview detectors every 3 frames (~0.3 s for the
        # 10 FPS demo clips) so governance boxes do not appear stale to viewers.
        default_stride = "3" if self.dashboard_preview else "5"
        self.video_inference_stride = max(1, int(os.getenv("RAVEN_VIDEO_INFERENCE_STRIDE", default_stride)))
        self._last_pair_output = None
        self._last_pair_frame_index = None
        self.last_visual_boxes = []
        self.identity_multipliers = self.config.get("identity_multipliers", {"authorized": .5, "unknown": 1.0, "flagged": 1.5})
        self.alert_thresholds = self.config.get("alerts", {"warning": 50, "alert": 70})
        tracking = self.config.get("tracking", {}).get("tracker_yaml", "botsort.yaml")
        rgb_model_key = "RAVEN_THERMAL_MODEL" if self.input_modality == "thermal" else "RAVEN_RGB_MODEL"
        rgb_config_key = "thermal_model" if self.input_modality == "thermal" else "rgb_model"
        rgb_model_path = (str(PROJECT_ROOT / "yolov8n.pt") if self.dashboard_preview else
                          self._configured_model(rgb_model_key, detect.get(rgb_config_key, "yolo26l.pt")))
        self.rgb_tracker = rgb_tracker or SentryTracker(
            rgb_model_path,
            tracker_yaml=tracking, image_size=self.image_size,
            iou_threshold=self.iou_threshold)
        self.quality_detector = quality_detector
        self.quality_model_path = self._configured_model("RAVEN_RGB_MODEL", detect.get("rgb_model", "yolo26l.pt"))
        self.tracker_yaml = tracking
        thermal_model_path = (str(PROJECT_ROOT / "yolov8n.pt") if self.dashboard_preview else
                              self._configured_model("RAVEN_THERMAL_MODEL", detect.get("thermal_model", "yolov8n.pt")))
        self.thermal_tracker = (thermal_tracker or SentryTracker(
            thermal_model_path,
            tracker_yaml=tracking, image_size=self.image_size,
            iou_threshold=self.iou_threshold)) if self.enable_thermal else None
        reid_config = self.config.get("reid", {})
        self.rgb_reid = rgb_reid or ReIDBuffer(
            buffer_timeout=float(reid_config.get("buffer_timeout_seconds", 10.0)),
            enabled=reid_config.get("enabled", True),
            similarity_threshold=float(reid_config.get("similarity_threshold", .90)),
            modality=self.input_modality)
        self.thermal_reid = (thermal_reid or ReIDBuffer(
            buffer_timeout=float(reid_config.get("buffer_timeout_seconds", 10.0)),
            enabled=reid_config.get("enabled", True),
            similarity_threshold=float(reid_config.get("similarity_threshold", .90)))) if self.enable_thermal else None
        self.pose_classifier = (None if self.dashboard_preview else pose_classifier or
                                PostureClassifier(self._configured_model("RAVEN_POSE_MODEL", pose.get("model", "yolo26l-pose.pt")),
                                                  conf_thresh=float(pose.get("conf_threshold", .5))))
        self._thermal_landmarker = None
        self._thermal_landmarker_error = None
        self.fusion = fusion or ConsensusFusion()
        self.learned_fusion = None
        learned_fusion_path = os.getenv("RAVEN_CONSENSUS_WEIGHTS", str(PROJECT_ROOT / "data" / "models" / "consensus_fusion.pt"))
        if not self.dashboard_preview and Path(learned_fusion_path).is_file():
            try:
                import torch
                self.learned_fusion = LearnedConsensusFusion.load(learned_fusion_path)
                self.learned_fusion.eval()
            except Exception as exc:
                LOGGER.warning("Learned consensus weights could not be loaded; using configured fallback: %s", exc)
        self.temporal_fusion_runtime = None
        # Unknown provenance is treated as non-operational just like synthetic.
        self.temporal_fusion_checkpoint_synthetic = False
        self.temporal_fusion_checkpoint_verified = False
        self.temporal_fusion_scoring_eligible = False
        temporal_weights = os.getenv("RAVEN_TEMPORAL_FUSION_WEIGHTS",
                                     str(PROJECT_ROOT / "data" / "models" / "best_model.pth"))
        if not self.dashboard_preview and (temporal_fusion_model is not None or Path(temporal_weights).is_file()):
            try:
                temporal_model = (temporal_fusion_model or
                                  TemporalCrossModalFusion.load(temporal_weights))
                temporal_metadata = getattr(temporal_model, "training_metadata", {}) or {}
                self.temporal_fusion_checkpoint_synthetic = bool(temporal_metadata.get("synthetic", False))
                self.temporal_fusion_checkpoint_verified = bool(
                    temporal_metadata and not self.temporal_fusion_checkpoint_synthetic
                    and temporal_metadata.get("validated_on_real_data") is True)
                # Metadata alone is not proof of experimental validity, but an
                # explicit real-data validation record is required before score use.
                self.temporal_fusion_scoring_eligible = bool(
                    self.temporal_fusion_checkpoint_verified)
                temporal_device = (f"cuda:{getattr(self.rgb_tracker, 'device', 0)}"
                                   if str(self.device_name).startswith("CUDA") else "cpu")
                self.temporal_fusion_runtime = TemporalCrossModalRuntime(
                    temporal_model, window=int(processing.get("temporal_window", 32)),
                    minimum_history=min(8, int(processing.get("temporal_window", 32))),
                    device=temporal_device)
                if self.temporal_fusion_checkpoint_synthetic:
                    LOGGER.warning("Synthetic TemporalCrossModalFusion checkpoint loaded for architecture/data-flow checks; "
                                   "it is disabled for scoring and review decisions")
                elif not self.temporal_fusion_checkpoint_verified:
                    LOGGER.warning("TemporalCrossModalFusion checkpoint lacks verified real-data provenance; "
                                   "it is disabled for scoring and review decisions")
            except Exception as exc:
                LOGGER.warning("Temporal cross-modal checkpoint could not be loaded: %s", exc)
        default_fusion_weights = PROJECT_ROOT / "data" / "models" / "confidence_fusion.pt"
        fusion_weights = os.getenv("RAVEN_CONFIDENCE_FUSION_WEIGHTS", str(default_fusion_weights))
        self.confidence_fusion = None
        if not self.dashboard_preview and fusion_weights and Path(fusion_weights).is_file():
            try:
                self.confidence_fusion = ConfidenceFusionHead.load(fusion_weights)
                self.confidence_fusion.eval()
            except Exception as exc:
                LOGGER.warning("Confidence-fusion weights could not be loaded; using rule fusion: %s", exc)
        governance_weights = os.getenv("RAVEN_GOVERNANCE_WEIGHTS", str(default_fusion_weights))
        self.governance_head = None
        self.governance_load_error = None
        if governance_weights and Path(governance_weights).is_file():
            try:
                self.governance_head = PostHocRiskHead.load(governance_weights, map_location="cpu")
            except Exception as exc:
                self.governance_load_error = str(exc)
                LOGGER.info("S-DEC governance head unavailable; retaining compatible confidence fallback: %s", exc)
        self.trajectory_smoother = AdaptiveTrajectorySmoother()
        self.action_runtime = None
        self.action_model_error = None
        self.action_model_synthetic = False
        # Explicitly labeled review-only fallback; not an action classifier.
        self.heuristic_fallback = MotionAnomalyFallback()
        action_weights = os.getenv("RAVEN_ACTION_MODEL_WEIGHTS",
                                   str(PROJECT_ROOT / "data" / "models" / "action_anomaly.pt"))
        if not self.dashboard_preview and action_weights and Path(action_weights).is_file():
            try:
                action_model = TemporalPoseAnomalyHead.load(action_weights)
                self.action_model_synthetic = bool(
                    getattr(action_model, "training_metadata", {}).get("synthetic", False))
                action_device = (f"cuda:{getattr(self.rgb_tracker, 'device', 0)}"
                                 if str(self.device_name).startswith("CUDA") else "cpu")
                self.action_runtime = TrackAnomalyRuntime(action_model, device=action_device)
            except Exception as exc:
                self.action_model_error = str(exc)
                LOGGER.warning("Temporal action model could not be loaded; using motion heuristic: %s", exc)
        elif action_weights:
            self.action_model_error = f"Action checkpoint not found: {Path(action_weights).resolve()}"
            LOGGER.info("Action anomaly model unavailable: %s", self.action_model_error)
        self.alert_store = Path(alert_store) if alert_store else PROJECT_ROOT / "data" / "alerts.jsonl"
        object_model_path = os.getenv("RAVEN_THREAT_MODEL", object_config.get("model_path"))
        self.custom_threat_model_configured = bool(os.getenv("RAVEN_THREAT_MODEL"))
        self.object_detector = (None if disable_threat_detector or self.dashboard_preview else object_detector or ThreatObjectDetector(
            model_path=self._path(object_model_path) if object_model_path else None,
            model=getattr(self.rgb_tracker, "model", None) if not object_model_path else None,
            labels=object_config.get("labels"),
            confidence=float(object_config.get("confidence", 0.35)),
            image_size=self.image_size, iou_threshold=self.iou_threshold))
        alert_config = self.config.get("alerts", {})
        self.alert_generator = AlertGenerator(
            self.alert_store,
            warning_threshold=alert_config.get("warning", 50),
            alert_threshold=alert_config.get("alert", 70),
            threat_threshold=object_config.get("confidence", 0.35),
            cooldown_seconds=alert_config.get("cooldown_seconds", 5),
            trigger_actions=alert_config.get("trigger_actions", ["notify_operator", "log_event"]))

    def _path(self, value):
        path = Path(value)
        return str((path if path.is_absolute() else self.config_path.parent.parent / path).resolve())

    def _configured_model(self, env_name, config_value):
        """Allow deployment environments to select exported ONNX weights."""
        return self._path(os.getenv(env_name, config_value))

    def configure_single_stream(self, modality="rgb"):
        """Select the detector/Re-ID branch for a one-camera uploaded clip."""
        target = "thermal" if str(modality).lower() in {"thermal", "ir", "thermal-palette"} else "rgb"
        if target == self.input_modality:
            return
        detect = self.config.get("detection", {})
        reid_config = self.config.get("reid", {})
        env_key, config_key = (("RAVEN_THERMAL_MODEL", "thermal_model") if target == "thermal"
                               else ("RAVEN_RGB_MODEL", "rgb_model"))
        self.rgb_tracker = SentryTracker(
            self._configured_model(env_key, detect.get(config_key, "yolo26l.pt" if target != "thermal" else "yolov8n.pt")),
            tracker_yaml=self.tracker_yaml, image_size=self.image_size,
            iou_threshold=self.iou_threshold)
        self.rgb_reid = ReIDBuffer(
            buffer_timeout=float(reid_config.get("buffer_timeout_seconds", 10.0)),
            enabled=reid_config.get("enabled", True),
            similarity_threshold=float(reid_config.get("similarity_threshold", .90)),
            modality=target)
        self.input_modality = target
        self.rgb_detection_threshold = (float(self.person_filter.get("thermal_min_confidence", self.threshold))
                                        if target == "thermal" else self.threshold)
        self.reset_video_state()

    def _filter_person_detections(self, detections, frame_shape, modality="rgb"):
        """Drop low-confidence, flat, tiny, and implausibly large person boxes.

        This is a proposal-quality gate that reduces shadows/reflections; it
        cannot guarantee rejection of a person-shaped reflection.
        """
        height, width = frame_shape[:2]
        min_conf = (min(0.22, float(self.person_filter.get("thermal_min_confidence", .38)))
                    if self.dashboard_preview and modality == "thermal"
                    else float(self.person_filter.get(
                        "thermal_min_confidence" if modality == "thermal" else "min_confidence", .52)))
        min_height = max(float(self.person_filter.get("min_height_px", 20)),
                         height * float(self.person_filter.get("min_height_fraction", .025)))
        min_ratio = float(self.person_filter.get("min_width_height_ratio", .12))
        max_ratio = float(self.person_filter.get("max_width_height_ratio", 1.45))
        max_area = float(self.person_filter.get("max_frame_area_fraction", .55))
        accepted = []
        for item in detections:
            if int(item.get("class_id", 0)) != 0:
                continue
            x1, y1, x2, y2 = map(float, item.get("bbox", (0, 0, 0, 0)))
            box_w, box_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if (float(item.get("conf", 0.0)) < min_conf or box_h < min_height
                    or box_w / max(box_h, 1.0) < min_ratio
                    or box_w / max(box_h, 1.0) > max_ratio
                    or box_w * box_h > width * height * max_area):
                continue
            accepted.append(item)
        return accepted

    @staticmethod
    def _quality_score(frame):
        quality = assess_image_quality(frame)
        return estimate_frame_quality(
            quality["brightness_mean"], quality["laplacian_variance"],
            quality["width"], quality["height"])

    @staticmethod
    def _normalize_pose_keypoints(keypoints, bbox):
        # Ultralytics returns keypoints as NumPy arrays; their truth value is
        # ambiguous for multi-element arrays. Check absence/size explicitly.
        if keypoints is None or bbox is None or len(keypoints) == 0 or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = map(float, bbox)
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        return [[min(1.0, max(0.0, (float(point[0]) - x1) / width)),
                 min(1.0, max(0.0, (float(point[1]) - y1) / height)),
                 min(1.0, max(0.0, float(point[2]) if len(point) > 2 else 0.0))]
                for point in keypoints if len(point) >= 2]

    @staticmethod
    def _fusion_features(track, keypoints, posture_confidence, quality, feature_dim):
        """Pack box, appearance descriptor, pose, confidence, and quality consistently."""
        bbox = track.get("bbox") or (0, 0, 0, 0)
        x1, y1, x2, y2 = map(float, bbox)
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        frame_h, frame_w = track.get("frame_shape", (1, 1))[:2]
        values = [x1 / max(1, frame_w), y1 / max(1, frame_h),
                  x2 / max(1, frame_w), y2 / max(1, frame_h)]
        appearance = track.get("reid_embedding")
        if appearance is None:
            appearance = []
        values.extend(float(v) for v in appearance)
        points = keypoints or []
        for index in range(17):
            if index < len(points) and len(points[index]) >= 2:
                values.extend((min(1., max(0., (float(points[index][0]) - x1) / width)),
                               min(1., max(0., (float(points[index][1]) - y1) / height)),
                               float(points[index][2]) if len(points[index]) > 2 else 0.))
            else:
                values.extend((0., 0., 0.))
        values.extend((float(posture_confidence), float(track.get("conf", 0)), float(quality)))
        values = values[:feature_dim]
        values.extend([0.] * (feature_dim - len(values)))
        return values

    def _learned_consensus(self, rgb_track, thermal_track, rgb_keypoints, thermal_keypoints,
                           rgb_data, thermal_data, identity, weapon_confidence=0.0):
        """Run the trained dual-stream model; return None when no checkpoint exists."""
        if self.learned_fusion is None or thermal_track is None or thermal_data is None:
            return None
        try:
            import torch
            dim = self.learned_fusion.feature_dim
            rgb_track = dict(rgb_track, frame_shape=getattr(self, "_rgb_frame_shape", (1, 1)))
            thermal_track = dict(thermal_track, frame_shape=getattr(self, "_thermal_frame_shape", (1, 1)))
            rgb_feature = torch.tensor(self._fusion_features(rgb_track, rgb_keypoints,
                rgb_data.get("posture_conf", 0), rgb_data.get("quality_score", 1), dim), dtype=torch.float32)
            thermal_feature = torch.tensor(self._fusion_features(thermal_track, thermal_keypoints,
                thermal_data.get("posture_conf", 0), thermal_data.get("quality_score", 1), dim), dtype=torch.float32)
            tier = identity if identity in IDENTITY_TIERS else "unknown"
            device = next(self.learned_fusion.parameters()).device
            with torch.inference_mode():
                logit, details = self.learned_fusion(rgb_feature.to(device), thermal_feature.to(device),
                    identity_tier=tier, weapon_context=torch.tensor([weapon_confidence, float(weapon_confidence > 0)], device=device),
                    return_details=True)
            probability = float(details["probability"].detach().cpu().item())
            threshold = float(details["alert_threshold"].detach().cpu().item())
            evidence = {"fusion_method": "learned_cross_modal_attention_consensus",
                        "score_is_risk_probability": True, "probability": probability,
                        "learned_alert_threshold": threshold,
                        "identity_tier": tier, "weapon_context_confidence": float(weapon_confidence),
                        "cross_attention": details["cross_attention"].detach().cpu().tolist()}
            return int(round(probability * 100)), evidence, probability < threshold
        except Exception as exc:
            LOGGER.warning("Learned consensus inference failed; using configured fallback: %s", exc)
            return None

    def _apply_action_scores(self, results, frame_shape):
        # Synthetic weights are kept loadable for software demonstrations, but
        # do not produce review decisions. The motion heuristic is the fallback.
        if self.action_runtime is None or self.action_model_synthetic:
            return results
        calibrated_threshold = float(getattr(self.action_runtime.model, "review_threshold", 70.0))
        try:
            threshold = float(os.getenv("RAVEN_ACTION_REVIEW_THRESHOLD", str(calibrated_threshold)))
        except ValueError as exc:
            raise ValueError("RAVEN_ACTION_REVIEW_THRESHOLD must be a number from 0 to 100") from exc
        if not math.isfinite(threshold) or not 0 <= threshold <= 100:
            raise ValueError("RAVEN_ACTION_REVIEW_THRESHOLD must be a number from 0 to 100")
        for row in results:
            if row.get("track_id") is None or row.get("predicted"):
                continue
            features = encode_track_features(row, frame_shape)
            timestamp = row.get("video_timestamp_seconds")
            if timestamp is None:
                timestamp = time.time()
            track_key = row.get("reid_canonical_track_id", row["track_id"])
            score = self.action_runtime.update(track_key, features, timestamp)
            if score is None:
                continue
            row["anomaly_score"] = int(round(score))
            row["anomaly_method"] = "weakly_supervised_temporal_pose_gru"
            row["anomaly_model_available"] = True
            row["signal_label"] = "PROTOTYPE_REVIEW_CUE"
            row["model_validation_status"] = "not_independently_validated"
            row["anomaly_decision_threshold"] = threshold
            row["anomaly_threshold_source"] = ("RAVEN_ACTION_REVIEW_THRESHOLD" if
                                                "RAVEN_ACTION_REVIEW_THRESHOLD" in os.environ
                                                else "validation_calibrated")
            row["needs_review"] = bool(row.get("needs_review") or score >= threshold)
            row.setdefault("available_models", {})["action_anomaly_model"] = True
        return results

    def _apply_heuristic_fallback(self, results, frame_shape):
        """Populate labeled motion cues if no eligible model is scoring."""
        if self.action_runtime is not None and not self.action_model_synthetic:
            return results
        if os.getenv("RAVEN_ENABLE_MOTION_FALLBACK", "1").strip().lower() not in {"1", "true", "yes"}:
            return results
        for row in results:
            # Preserve learned temporal/action scores; motion is a fallback only
            # when this observation has no model output.
            if (row.get("track_id") is None or row.get("predicted") or not row.get("bbox")
                    or row.get("anomaly_model_available")):
                continue
            x1, y1, x2, y2 = map(float, row["bbox"])
            score = self.heuristic_fallback.update(
                row["track_id"], ((x1 + x2) / 2, (y1 + y2) / 2),
                row.get("video_timestamp_seconds"), frame_shape)
            row["anomaly_score"] = score["score"]
            row["anomaly_method"] = score["method"]
            row["anomaly_model_available"] = False
            row["anomaly_mode"] = "heuristic_fallback"
            row["motion_heuristic"] = score
            row["signal_label"] = "PROTOTYPE_REVIEW_CUE"
            row["model_validation_status"] = "heuristic_not_a_trained_model"
            # Keep this a display/review cue; it does not generate an alert.
            row["risk_score"] = score["score"]
            row["risk_score_type"] = "motion_heuristic_review_priority_not_probability"
            row["needs_review"] = bool(score["score"] >= self.alert_generator.warning_threshold)
        return results

    def set_processing_mode(self, mode: str):
        if mode not in self.processing_modes:
            raise ValueError(f"processing mode must be one of {tuple(self.processing_modes)}")
        self.processing_mode = mode
        self.processing_settings = dict(self.processing_modes[mode])

    @property
    def device_name(self):
        device = str(getattr(self.rgb_tracker, "device", "cpu") or "cpu")
        return f"CUDA:{device}" if device.isdigit() else device.upper() if device == "mps" else "CPU"

    @property
    def gpu_memory_mb(self):
        if not self.device_name.startswith("CUDA"):
            return None
        try:
            import torch
            return round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
        except Exception:
            return None

    @staticmethod
    def _validate_frame(frame, name):
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
            raise ValueError(f"{name} must be a non-empty image array")
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
            raise ValueError(f"{name} must have 1, 3, or 4 channels")

    def _associate(self, rgb, thermal):
        choices = []
        for i, r in enumerate(rgb):
            for j, t in enumerate(thermal):
                overlap, distance = _iou(r["bbox"], t["bbox"]), _distance(r["bbox"], t["bbox"])
                if overlap >= self.association_iou or distance <= self.max_center_distance:
                    choices.append((overlap - distance * .01, i, j))
        pairs, used_r, used_t = {}, set(), set()
        for _, i, j in sorted(choices, reverse=True):
            if i not in used_r and j not in used_t:
                pairs[i] = j
                used_r.add(i)
                used_t.add(j)
        return pairs

    def process_frame_pair(self, rgb_frame, thermal_frame, zone_context=0, *, metadata=None):
        self._validate_frame(rgb_frame, "rgb_frame")
        self._validate_frame(thermal_frame, "thermal_frame")
        self._rgb_frame_shape = rgb_frame.shape
        self._thermal_frame_shape = thermal_frame.shape
        if not self.enable_thermal:
            raise ValueError("This pipeline was loaded for RGB-only processing")
        if not isinstance(zone_context, (int, float, bool)) or not math.isfinite(float(zone_context)):
            raise ValueError("zone_context must be a finite number")
        metadata = metadata or {}
        frame_index = int(metadata.get("frame_index", self._frame_index))
        timestamp_seconds = float(metadata.get("video_timestamp_seconds", time.time()))
        stride = self.video_inference_stride if metadata.get("frame_index") is not None else 1
        run_full_inference = (self._last_pair_frame_index is None
                              or frame_index % stride == 0)
        if not run_full_inference and self._last_pair_output is not None:
            # Motion extrapolation supplies boxes on skipped video frames;
            # cached pose/Re-ID/fusion results avoid expensive model passes.
            rgb_predictions = {str(item.get("track_id")): item
                               for item in self.track_states.predict(timestamp_seconds)}
            thermal_predictions = {str(item.get("track_id")): item
                                   for item in self.thermal_track_states.predict(timestamp_seconds)}
            self.last_visual_boxes = ([{"modality": "rgb", **item} for item in rgb_predictions.values()]
                                      + [{"modality": "thermal", **item}
                                         for item in thermal_predictions.values()])
            output = []
            for cached in self._last_pair_output:
                item = dict(cached)
                item["video_timestamp_seconds"] = timestamp_seconds
                item["predicted"] = True
                item["detection_skipped"] = True
                item["processing_ms"] = 0.0
                item.pop("alert_event", None)
                rgb_prediction = rgb_predictions.get(str(item.get("tracker_id")))
                if rgb_prediction is not None:
                    item["bbox"] = rgb_prediction["bbox"]
                    item["velocity"] = rgb_prediction.get("velocity", item.get("velocity"))
                    item["confidence"] = float(rgb_prediction.get("conf", item.get("confidence", 0)))
                    item["occlusion_state"] = rgb_prediction.get("occlusion_state", item.get("occlusion_state", "VISIBLE"))
                thermal_prediction = thermal_predictions.get(str(item.get("thermal_track_id")))
                if thermal_prediction is not None:
                    item["bbox_thermal"] = thermal_prediction["bbox"]
                    item["occlusion_state"] = thermal_prediction.get("occlusion_state", item.get("occlusion_state", "VISIBLE"))
                output.append(item)
            self._last_pair_frame_index = frame_index
            # Cached paths were already smoothed on the inference frame.
            return output
        start = time.perf_counter()
        rgb_thresh = min(0.25, self.rgb_detection_threshold) if self.dashboard_preview else self.rgb_detection_threshold
        rgb_detections = self.rgb_tracker.track(rgb_frame, conf_thresh=rgb_thresh)
        rgb_visual_boxes = [dict(item) for item in rgb_detections
                            if int(item.get("class_id", 0)) == 0]
        rgb_detections = self._filter_person_detections(rgb_detections, rgb_frame.shape, "rgb")
        rgb_state_detections = self.track_states.update(rgb_detections, timestamp_seconds)
        min_hits = 1 if self.dashboard_preview else max(1, int(self.person_filter.get("min_confirmed_hits", 2)))
        rgb_people = [self._ensure_track_id(item, index) for index, item in enumerate(rgb_state_detections)
                      if int(item.get("class_id", 0)) == 0
                      and int(item.get("observed_hits", 1)) >= min_hits
                      and item.get("occlusion_state", "VISIBLE") != "LOST"]
        rgb = self.rgb_reid.update_and_match(rgb_people, rgb_frame, timestamp=timestamp_seconds)
        th_conf = (min(0.22, float(self.person_filter.get("thermal_min_confidence", self.threshold)))
                   if self.dashboard_preview else float(self.person_filter.get("thermal_min_confidence", self.threshold)))
        thermal_detections = self.thermal_tracker.track(
            thermal_frame, conf_thresh=th_conf,
            classes=[0])
        thermal_visual_boxes = [dict(item) for item in thermal_detections
                                if int(item.get("class_id", 0)) == 0]
        self.last_visual_boxes = ([{"modality": "rgb", **item} for item in rgb_visual_boxes]
                                  + [{"modality": "thermal", **item}
                                     for item in thermal_visual_boxes])
        thermal_detections = self._filter_person_detections(thermal_detections, thermal_frame.shape, "thermal")
        thermal_detections = self.thermal_track_states.update(thermal_detections, timestamp_seconds)
        thermal_people = [self._ensure_track_id(item, index) for index, item in enumerate(thermal_detections)
                          if item.get("occlusion_state", "VISIBLE") != "LOST"]
        thermal = self.thermal_reid.update_and_match(thermal_people, thermal_frame, timestamp=timestamp_seconds)
        pairs = self._associate(rgb, thermal)
        output = []
        for i, r in enumerate(rgb):
            if self.dashboard_preview:
                rkpts, rpose, rc = None, "not_run_in_preview", 0.0
            else:
                rkpts, rpose = self.pose_classifier.get_pose(rgb_frame, r["bbox"])
                rc = float(getattr(self.pose_classifier, "last_confidence", .5 if rpose != "unknown" else 0))
            rd = {"posture_conf": rc, "track_conf": float(r.get("conf", 0)), "recovered": bool(r.get("recovered")), "posture": rpose}
            t = thermal[pairs[i]] if i in pairs else None
            tkpts = tpose = None
            td = None
            if t:
                if self.dashboard_preview:
                    tkpts, tpose, tc = None, "not_run_in_preview", 0.0
                else:
                    tkpts, tpose = self.pose_classifier.get_pose(thermal_frame, t["bbox"])
                    tc = float(getattr(self.pose_classifier, "last_confidence", .5 if tpose != "unknown" else 0))
                td = {"posture_conf": tc, "track_conf": float(t.get("conf", 0)), "recovered": bool(t.get("recovered")), "posture": tpose}
            if not self.dashboard_preview:
                rd["quality_score"] = self._quality_score(rgb_frame)
                if td is not None:
                    td["quality_score"] = self._quality_score(thermal_frame)
            thermal_face = None
            if (not self.dashboard_preview and t and (metadata or {}).get("thermal_source", "sensor") in {"sensor", "thermal_imaging"}
                    and os.getenv("RAVEN_THERMAL_FACE_LANDMARKS", "0").lower() in {"1", "true", "yes"}):
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
                thermal_face = self._thermal_face_landmarks(thermal_frame[y1:y2, x1:x2])
                if thermal_face and thermal_face.get("landmarks"):
                    thermal_face["landmarks"] = [[float(point[0]) + x1, float(point[1]) + y1]
                                                  for point in thermal_face["landmarks"]]
            px1, py1, px2, py2 = [float(value) for value in r["bbox"]]
            state = self.track_states.tracks.get(str(r.get("tracker_id", r["track_id"])))
            _motion_score, motion_signals = _motion_anomaly(r, py2 - py1)
            thermal_source = (metadata or {}).get("thermal_source", "sensor")
            thermal_available = thermal_source in {"sensor", "thermal_imaging"}
            score, evidence, review = self.fusion.evaluate(rd, td, float(zone_context))
            identity = (metadata or {}).get("identity", "unknown")
            learned = (None if self.dashboard_preview else
                       self._learned_consensus(r, t, rkpts, tkpts, rd, td, identity))
            if learned is not None:
                score, evidence, review = learned
            temporal_anomaly_score = None
            if (not self.dashboard_preview and self.temporal_fusion_runtime is not None
                    and self.temporal_fusion_scoring_eligible
                    and t is not None and td is not None):
                feature_dim = self.temporal_fusion_runtime.model.feature_dim
                rgb_features = self._fusion_features(
                    dict(r, frame_shape=rgb_frame.shape), rkpts, rc, rd.get("quality_score", 1), feature_dim)
                thermal_features = self._fusion_features(
                    dict(t, frame_shape=thermal_frame.shape), tkpts,
                    td.get("posture_conf", 0), td.get("quality_score", 1), feature_dim)
                temporal_anomaly_score = self.temporal_fusion_runtime.update(
                    r.get("reid_canonical_track_id", r.get("track_id")),
                    rgb_features, thermal_features)
            governance_action = None
            governance_abstain = False
            if self.governance_head is not None and thermal_available and t is not None and learned is None:
                score, governance_action, governance_abstain = self.governance_head.predict_governance(
                    r.get("bbox"), float(r.get("conf", 0)), t.get("bbox"),
                    float(t.get("conf", 0)), rgb_frame.shape)
                evidence["fusion_method"] = "post_hoc_sdec_absolute_risk_governance"
                evidence["governance_action"] = governance_action
                evidence["governance_abstained"] = governance_abstain
                review = review or governance_abstain
            elif self.confidence_fusion is not None and thermal_available and t is not None and learned is None:
                score = round(100 * self.confidence_fusion.predict_score(
                    float(r.get("conf", 0)), float(t.get("conf", 0))))
                evidence["fusion_method"] = "paper_confidence_only_calibration"
                evidence["consensus_score_is_risk_probability"] = False
                review = review or False
            if not thermal_available:
                evidence["thermal_source"] = thermal_source
                evidence["independent_thermal_sensor"] = False
            multiplier = float(self.identity_multipliers.get(identity, self.identity_multipliers["unknown"]))
            result = {"timestamp": datetime.now(timezone.utc).isoformat(), "track_id": str(r["track_id"]),
                      "tracker_id": str(r.get("tracker_id", r["track_id"])),
                      "reid_canonical_track_id": str(r.get("reid_canonical_track_id", r["track_id"])),
                      "video_timestamp_seconds": (metadata or {}).get("video_timestamp_seconds"),
                      "thermal_track_id": str(t["track_id"]) if t else None, "identity": identity,
                      "bbox": r["bbox"], "bbox_thermal": t["bbox"] if t else None,
                      "posture_rgb": rpose, "posture_thermal": tpose,
                      "kpts_rgb": self._json_safe(rkpts), "kpts_thermal": self._json_safe(tkpts),
                      "rgb_confidence": float(r.get("conf", 0)), "thermal_confidence": float(t.get("conf", 0)) if t else 0,
                      "risk_score": None, "fusion_score": int(score), "identity_multiplier": multiplier,
                      "needs_review": bool(governance_abstain),
                      "alert_status": "review_required" if governance_abstain else "normal",
                      "governance_action": governance_action,
                      "reid_status": r.get("reid_status", "unknown"),
                      "reid_similarity": r.get("reid_similarity"),
                      "reid_method": r.get("reid_method", "color_gradient_descriptor"),
                      "identity_source": r.get("identity_source", "tracker"),
                      "anomaly_score": (round(temporal_anomaly_score * 100) if temporal_anomaly_score is not None else None),
                      "anomaly_model_available": temporal_anomaly_score is not None,
                      "anomaly_method": ("temporal_cross_modal_attention_gru" if temporal_anomaly_score is not None
                                         else "motion_heuristic_fallback"),
                      "motion_signals": motion_signals,
                      "risk_score_type": ("synthetic_demo_model_score"
                                           if temporal_anomaly_score is not None and self.temporal_fusion_checkpoint_synthetic
                                           else "trained_temporal_model" if temporal_anomaly_score is not None
                                           else "unavailable"),
                      "model_validation_status": ("not_independently_validated"
                                                   if temporal_anomaly_score is not None
                                                   else "synthetic_checkpoint_disabled_for_scoring"
                                                   if self.temporal_fusion_checkpoint_synthetic
                                                   else "no_temporal_score"),
                      "demo_label": None,
                      "signal_label": ("PROTOTYPE_REVIEW_CUE" if temporal_anomaly_score is not None else None),
                      "source": "rgb+thermal" if thermal_available and t else "rgb+estimated-thermal" if t else "rgb",
                      "thermal_sensor_available": thermal_available,
                      "thermal_source": thermal_source if t else None,
                      "thermal_temperature_c": (metadata or {}).get("thermal_temperature_c"),
                      "thermal_face_landmarks": thermal_face,
                      "thermal_face_landmarks_available": bool(t is not None and os.getenv("RAVEN_THERMAL_FACE_LANDMARKS", "0").lower() in {"1", "true", "yes"}),
                      "temperature_calibrated": bool((metadata or {}).get("temperature_calibrated", False)),
                      "available_models": {"detector": True, "pose": True,
                                            "st_gcn": False, "patchcore": False,
                                            "confidence_fusion": self.confidence_fusion is not None,
                                            "governance_head": self.governance_head is not None,
                                            "action_anomaly_model": self.action_runtime is not None,
                                            "temporal_cross_modal_fusion": self.temporal_fusion_runtime is not None},
                      "occlusion_state": r.get("occlusion_state", "VISIBLE"),
                      "occlusion_recovered_rgb": bool(r.get("recovered") or r.get("occlusion_state") == "REACQUIRED"),
                      "occlusion_recovered_thermal": bool(t and (t.get("recovered") or t.get("occlusion_state") == "REACQUIRED")),
                      "confidence": round((float(r.get("conf", 0)) + float(t.get("conf", 0))) / 2, 3) if t else float(r.get("conf", 0)),
                      "keypoints_normalized": self._normalize_pose_keypoints(rkpts, r["bbox"]),
                      "velocity": r.get("velocity", (0.0, 0.0)),
                      "acceleration": r.get("acceleration", (0.0, 0.0)),
                      "evidence": self._json_safe(evidence)}
            output.append(result)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        for item in output:
            item["processing_ms"] = elapsed
        threat_interval = max(1, int(os.getenv("RAVEN_THREAT_INTERVAL",
                                               self.config.get("threat_detection", {}).get("interval", 1))))
        frame_index = int(metadata.get("frame_index", 0))
        for item in output:
            if item.get("keypoints_normalized") is None:
                item["keypoints_normalized"] = self._normalize_pose_keypoints(item.get("kpts_rgb"), item.get("bbox"))
            if self.action_runtime is not None and not self.dashboard_preview:
                item["available_models"] = {**item.get("available_models", {}),
                                             "action_anomaly_model": True}
        if not self.dashboard_preview:
            output = self._apply_action_scores(output, rgb_frame.shape)
            output = self._apply_heuristic_fallback(output, rgb_frame.shape)
        output = self._finalize_results(
            output, rgb_frame, observations=rgb_detections,
            run_object_detection=not self.dashboard_preview,
            run_threat_detection=(not self.dashboard_preview and frame_index % threat_interval == 0),
        )
        output = self.trajectory_smoother.apply_to_results(output)
        if self.dashboard_preview and self.governance_head is not None:
            matched_rgb = set(pairs)
            matched_thermal = set(pairs.values())
            for index, track in enumerate(rgb):
                if index in matched_rgb:
                    continue
                score, action, abstain = self.governance_head.predict_governance(
                    track.get("bbox"), float(track.get("conf", 0)), None, 0.0,
                    rgb_frame.shape, available_actions=("ir_vetoed_rgb_dominant",))
                output.append({"track_id": str(track.get("track_id", f"rgb-{index}")),
                               "tracker_id": str(track.get("tracker_id", track.get("track_id", index))),
                               "video_timestamp_seconds": timestamp_seconds,
                               "thermal_track_id": None, "bbox": track.get("bbox"),
                               "bbox_thermal": None, "rgb_confidence": float(track.get("conf", 0)),
                               "thermal_confidence": 0.0, "fusion_score": score,
                               "governance_action": action, "needs_review": abstain,
                               "alert_status": "review_required" if abstain else "normal",
                               "thermal_sensor_available": True, "thermal_source": "thermal_imaging",
                               "occlusion_state": track.get("occlusion_state", "VISIBLE"),
                               "occlusion_recovered_rgb": bool(track.get("recovered") or track.get("occlusion_state") == "REACQUIRED"),
                               "occlusion_recovered_thermal": False,
                               "source": "rgb-only", "velocity": track.get("velocity", (0.0, 0.0)),
                               "evidence": {"fusion_method": "post_hoc_sdec_absolute_risk_governance",
                                            "governance_action": action, "matched_flag": False}})
            for index, track in enumerate(thermal):
                if index in matched_thermal:
                    continue
                score, action, abstain = self.governance_head.predict_governance(
                    None, 0.0, track.get("bbox"), float(track.get("conf", 0)),
                    rgb_frame.shape, available_actions=("rgb_vetoed_thermal_dominant",))
                thermal_id = str(track.get("track_id", f"ir-{index}"))
                output.append({"track_id": f"IR-{thermal_id}", "tracker_id": None,
                               "video_timestamp_seconds": timestamp_seconds,
                               "thermal_track_id": thermal_id, "bbox": None,
                               "bbox_thermal": track.get("bbox"), "rgb_confidence": 0.0,
                               "thermal_confidence": float(track.get("conf", 0)),
                               "fusion_score": score, "governance_action": action,
                               "needs_review": abstain,
                               "alert_status": "review_required" if abstain else "normal",
                               "thermal_sensor_available": True, "thermal_source": "thermal_imaging",
                               "occlusion_state": track.get("occlusion_state", "VISIBLE"),
                               "occlusion_recovered_rgb": False,
                               "occlusion_recovered_thermal": bool(track.get("recovered") or track.get("occlusion_state") == "REACQUIRED"),
                               "source": "thermal-only", "velocity": track.get("velocity", (0.0, 0.0)),
                               "evidence": {"fusion_method": "post_hoc_sdec_absolute_risk_governance",
                                            "governance_action": action, "matched_flag": False}})
        self._last_pair_output = [dict(item) for item in output]
        self._last_pair_frame_index = frame_index
        return output

    def process_rgb_frame(self, rgb_frame, zone_context=0, *, metadata=None,
                          enhanced_frame=None, quality_info=None):
        """Analyze one RGB frame without treating a display transform as IR data."""
        self._validate_frame(rgb_frame, "rgb_frame")
        if not isinstance(zone_context, (int, float, bool)) or not math.isfinite(float(zone_context)):
            raise ValueError("zone_context must be a finite number")
        start = time.perf_counter()
        metadata = metadata or {}
        frame_index = int(metadata.get("frame_index", self._frame_index))
        self._frame_index = frame_index + 1
        self._processed_frame_count += 1
        source_time = metadata.get("video_timestamp_seconds")
        timestamp_seconds = float(source_time) if source_time is not None else time.time()
        detection_interval = int(self.processing_settings["detection_interval"])
        if metadata.get("frame_index") is not None:
            detection_interval = max(detection_interval, self.video_inference_stride)
        source_gap = (frame_index - self._last_detection_frame
                      if self._last_detection_frame is not None else detection_interval)
        run_detection = (self._last_detection_frame is None or source_gap <= 0
                         or self._frames_since_detector + 1 >= detection_interval
                         or source_gap > max(10, detection_interval * 5))
        stage_ms = {"yolo_botsort": 0.0, "pose": 0.0, "reid": 0.0, "fusion": 0.0,
                    "enhanced_comparison": 0.0}
        object_observations = []
        if run_detection:
            stage_start = time.perf_counter()
            raw_detections = self.rgb_tracker.track(rgb_frame, conf_thresh=self.rgb_detection_threshold)
            self.last_visual_boxes = [dict(item) for item in raw_detections
                                      if int(item.get("class_id", 0)) == 0]
            raw_detections = self._filter_person_detections(raw_detections, rgb_frame.shape,
                                                             self.input_modality)
            object_observations = list(raw_detections)
            stage_ms["yolo_botsort"] = (time.perf_counter() - stage_start) * 1000
            detections = self.track_states.update(raw_detections, timestamp_seconds)
            self._last_detection_frame = frame_index
            self._frames_since_detector = 0
        else:
            detections = self.track_states.predict(timestamp_seconds)
            self.last_visual_boxes = [dict(item) for item in detections
                                      if int(item.get("class_id", 0)) == 0]
            self._frames_since_detector += 1
        stage_ms["yolo_botsort"] = round(stage_ms["yolo_botsort"], 2)
        quality = quality_info or assess_image_quality(rgb_frame)
        comparison = None
        if enhanced_frame is not None:
            self._validate_frame(enhanced_frame, "enhanced_frame")
            if self.quality_detector is None:
                # Load this additional model only on first opt-in use, then
                # retain it with the cached pipeline across Streamlit reruns.
                self.quality_detector = SentryTracker(self.quality_model_path,
                    tracker_yaml=self.tracker_yaml, image_size=self.image_size,
                    iou_threshold=self.iou_threshold)
            comparison_start = time.perf_counter()
            enhanced_detections = self.quality_detector.track(
                enhanced_frame, conf_thresh=self.rgb_detection_threshold
            )
            source_h, source_w = rgb_frame.shape[:2]
            enhanced_h, enhanced_w = enhanced_frame.shape[:2]
            scale_x, scale_y = source_w / enhanced_w, source_h / enhanced_h
            enhanced_detections = [
                {**item, "bbox": [
                    item["bbox"][0] * scale_x, item["bbox"][1] * scale_y,
                    item["bbox"][2] * scale_x, item["bbox"][3] * scale_y,
                ]}
                for item in enhanced_detections
            ]
            comparison = compare_detection_sets(detections, enhanced_detections)
            stage_ms["enhanced_comparison"] = round((time.perf_counter() - comparison_start) * 1000, 2)
        people = [self._ensure_track_id(item, index) for index, item in enumerate(detections)
                  if int(item.get("class_id", 0)) == 0
                  and int(item.get("observed_hits", 1)) >= max(1, int(self.person_filter.get("min_confirmed_hits", 2)))
                  and item.get("occlusion_state", "VISIBLE") != "LOST"]
        observed_people = [item for item in people if not item.get("predicted")]
        predicted_people = [item for item in people if item.get("predicted")]
        reid_start = time.perf_counter()
        tracks = self.rgb_reid.update_and_match(observed_people, rgb_frame, timestamp=timestamp_seconds)
        for track in tracks:
            tracker_id = str(track.get("tracker_id", track.get("track_id")))
            track["tracker_id"] = tracker_id
            canonical_id = str(track.get("reid_canonical_track_id", tracker_id))
            if canonical_id != tracker_id:
                state = self.track_states.tracks.pop(tracker_id, None)
                canonical_state = self.track_states.tracks.get(canonical_id)
                if state is not None and canonical_state is None:
                    state.track_id = canonical_id
                    self.track_states.tracks[canonical_id] = state
                    state.bbox = [float(v) for v in track["bbox"]]
                    state.last_seen = float(track.get("last_seen", state.last_seen))
                    center = state._center(state.bbox)
                    state.trajectory.append({"timestamp": state.last_seen,
                                             "center": [float((state.bbox[0] + state.bbox[2]) / 2), float(state.bbox[3])],
                                             "bbox": list(state.bbox)})
                elif state is not None and canonical_state is not None:
                    canonical_state.trajectory.extend(state.trajectory)
                    canonical_state.trajectory = type(canonical_state.trajectory)(
                        list(canonical_state.trajectory)[-20:], maxlen=20)
                    canonical_state.first_seen = min(canonical_state.first_seen, state.first_seen)
                    canonical_state.pose = state.pose or canonical_state.pose
                track["track_id"] = canonical_id
                track["first_seen"] = (self.track_states.tracks.get(canonical_id).first_seen
                                       if canonical_id in self.track_states.tracks else track.get("first_seen"))
                track["trajectory"] = (list(self.track_states.tracks[canonical_id].trajectory)
                                       if canonical_id in self.track_states.tracks else track.get("trajectory", []))
        current_canonical_ids = {str(item.get("track_id")) for item in tracks}
        tracks.extend({**item, "recovered": False} for item in predicted_people
                      if str(item.get("track_id")) not in current_canonical_ids)
        stage_ms["reid"] = round((time.perf_counter() - reid_start) * 1000, 2)
        pose_interval = int(self.processing_settings["pose_interval"])
        if metadata.get("frame_index") is not None:
            pose_interval = max(pose_interval, self.video_inference_stride)
        pose_interval_due = ((self._processed_frame_count - 1) % pose_interval == 0)
        pose_candidates = []
        for candidate in tracks:
            candidate_state = self.track_states.tracks.get(str(candidate["track_id"]))
            visible = candidate.get("occlusion_state", "VISIBLE") in {"VISIBLE", "REACQUIRED"}
            missing_pose = candidate_state is None or candidate_state.pose is None
            if visible and not candidate.get("predicted") and (pose_interval_due or missing_pose):
                pose_candidates.append(candidate)
        pose_candidates.sort(key=lambda item: float(item.get("conf", 0.0)), reverse=True)
        pose_budget = self.processing_settings.get("max_pose_tracks_per_frame")
        if pose_budget is not None:
            pose_candidates = pose_candidates[:max(0, int(pose_budget))]
        pose_track_ids = {str(item["track_id"]) for item in pose_candidates}
        output = []
        for track in tracks:
            track_id = str(track["track_id"])
            state = self.track_states.tracks.get(track_id)
            track_visible = track.get("occlusion_state", "VISIBLE") in {"VISIBLE", "REACQUIRED"}
            run_pose = track_visible and not track.get("predicted") and track_id in pose_track_ids
            if run_pose:
                pose_start = time.perf_counter()
                keypoints, posture = self.pose_classifier.get_pose(rgb_frame, track["bbox"])
                posture_confidence = float(getattr(self.pose_classifier, "last_confidence", .5 if posture != "unknown" else 0))
                stage_ms["pose"] += (time.perf_counter() - pose_start) * 1000
                self.track_states.record_pose(track_id, self._json_safe(keypoints), posture,
                                              posture_confidence, timestamp_seconds, track["bbox"])
            else:
                cached_pose = state.pose if state is not None and track_visible else None
                keypoints = cached_pose.get("keypoints") if cached_pose else None
                posture = cached_pose.get("label", "unknown") if cached_pose else "unknown"
                posture_confidence = float(cached_pose.get("confidence", 0.0)) if cached_pose else 0.0
            x1, y1, x2, y2 = [float(value) for value in track["bbox"]]
            _motion_score, motion_signals = _motion_anomaly(track, y2 - y1)
            rgb_data = {"posture_conf": posture_confidence, "track_conf": float(track.get("conf", 0)),
                        "recovered": bool(track.get("recovered")), "posture": posture}
            fusion_start = time.perf_counter()
            score, evidence, review = self.fusion.evaluate(rgb_data, None, float(zone_context))
            stage_ms["fusion"] += (time.perf_counter() - fusion_start) * 1000
            identity = (metadata or {}).get("identity", "unknown")
            multiplier = float(self.identity_multipliers.get(identity, self.identity_multipliers["unknown"]))
            output.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                           "video_timestamp_seconds": source_time,
                           "track_id": str(track["track_id"]), "thermal_track_id": None,
                           "tracker_id": str(track.get("tracker_id", track["track_id"])),
                           "reid_canonical_track_id": str(track.get("reid_canonical_track_id", track["track_id"])),
                           "identity": identity, "bbox": track["bbox"], "bbox_thermal": None,
                           "posture_rgb": posture, "posture_thermal": None,
                           "kpts_rgb": self._json_safe(keypoints), "kpts_thermal": None,
                           "rgb_confidence": float(track.get("conf", 0)), "thermal_confidence": 0.0,
                           "risk_score": None, "fusion_score": score, "identity_multiplier": multiplier,
                           "needs_review": False, "alert_status": "normal",
                            "reid_status": track.get("reid_status", "unknown"),
                            "reid_similarity": track.get("reid_similarity"),
                            "reid_method": track.get("reid_method", "color_gradient_descriptor"),
                            "identity_source": track.get("identity_source", "tracker"),
                           "anomaly_score": None, "anomaly_model_available": False,
                           "anomaly_method": ("weakly_supervised_temporal_pose_gru" if self.action_runtime is not None and not self.action_model_synthetic
                                              else "motion_heuristic_fallback" if os.getenv("RAVEN_ENABLE_MOTION_FALLBACK", "1").strip().lower() in {"1", "true", "yes"}
                                              else "disabled_no_trained_checkpoint"),
                           "motion_signals": motion_signals,
                           "risk_score_type": ("trained_temporal_model" if self.action_runtime is not None and not self.action_model_synthetic
                                                else "unavailable"), "source": "rgb-only",
                           "keypoints_normalized": self._normalize_pose_keypoints(keypoints, track["bbox"]),
                           "thermal_sensor_available": False, "confidence": float(track.get("conf", 0)),
                           "evidence": self._json_safe(evidence), "threat_objects": [],
                           "occlusion_recovered_rgb": bool(track.get("recovered")),
                           "occlusion_recovered_thermal": False,
                           "occlusion_state": track.get("occlusion_state", "VISIBLE"),
                           "predicted": bool(track.get("predicted", False)),
                      "first_seen": track.get("first_seen"), "last_seen": track.get("last_seen"),
                      "trajectory": track.get("trajectory", []),
                           "track_age": track.get("track_age", 0.0),
                           "time_since_detection": track.get("time_since_detection", 0.0),
                           "velocity": track.get("velocity", (0.0, 0.0)),
                           "acceleration": track.get("acceleration", (0.0, 0.0)),
                           "trajectory": track.get("trajectory", []),
                           "thermal_temperature_c": None, "temperature_calibrated": False,
                           "keypoints_normalized": ((state.pose or {}).get("normalized_keypoints")
                                                    if track_visible else None),
                           "image_quality": self._json_safe(quality),
                           "enhanced_detection_comparison": self._json_safe(comparison)})
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        stage_ms["pose"] = round(stage_ms["pose"], 2)
        stage_ms["fusion"] = round(stage_ms["fusion"], 2)
        stage_ms["total_pipeline"] = elapsed
        for stage, value in stage_ms.items():
            previous = self.stage_latency_ema.get(stage)
            self.stage_latency_ema[stage] = value if previous is None else .2 * value + .8 * previous
        smoothed_latency = {key: round(value, 2) for key, value in self.stage_latency_ema.items()}
        self.last_frame_stats = {"frame_index": frame_index, "detection_skipped": not run_detection,
                                 "stage_latency_ms": smoothed_latency,
                                 "device": self.device_name, "gpu_memory_mb": self.gpu_memory_mb}
        for item in output:
            item["processing_ms"] = elapsed
            item["stage_latency_ms"] = dict(smoothed_latency)
            item["detection_skipped"] = not run_detection
            item["processing_mode"] = self.processing_mode
            item["available_models"] = {"detector": True, "pose": True,
                                         "st_gcn": False, "patchcore": False,
                                         "action_anomaly_model": self.action_runtime is not None}
        configured_threat_interval = int(os.getenv(
            "RAVEN_THREAT_INTERVAL", self.config.get("threat_detection", {}).get("interval", 1)))
        threat_interval = max(1, int(self.processing_settings.get(
            "threat_interval", configured_threat_interval)))
        for item in output:
            if item.get("keypoints_normalized") is None:
                item["keypoints_normalized"] = self._normalize_pose_keypoints(item.get("kpts_rgb"), item.get("bbox"))
        output = self._apply_action_scores(output, rgb_frame.shape)
        output = self._apply_heuristic_fallback(output, rgb_frame.shape)
        output = self._finalize_results(
            output, rgb_frame, observations=object_observations,
            run_object_detection=run_detection,
            run_threat_detection=(run_detection and frame_index % threat_interval == 0),
        )
        return self.trajectory_smoother.apply_to_results(output)

    @staticmethod
    def _ensure_track_id(detection, index):
        """Supply Re-ID with a temporary key when the tracker has no ID yet."""
        item = detection.copy()
        if item.get("track_id") is None:
            item["track_id"] = f"det-{index}"
        return item

    def process_derived_thermal_frame(self, rgb_frame, zone_context=0, *, metadata=None):
        """Create a display-only thermal-style frame and analyze the RGB once.

        The colorized frame is derived from RGB and adds no independent sensor
        evidence. Running detector, tracker and pose a second time on it doubled
        inference without improving object detection or posture estimates.
        """
        self._validate_frame(rgb_frame, "rgb_frame")
        from src.thermal.processing import render_thermal_feed
        estimated_thermal = render_thermal_feed(rgb_frame)
        result = self.process_rgb_frame(rgb_frame, zone_context,
                                        metadata={**(metadata or {}), "thermal_source": "estimated_from_rgb"})
        for item in result:
            item["source"] = "rgb+estimated-thermal"
            item["thermal_source"] = "estimated_from_rgb"
            item["thermal_sensor_available"] = False
            item["temperature_calibrated"] = False
        return result, estimated_thermal

    def analysis_capabilities(self):
        threat = (self.object_detector.capabilities()
                  if self.object_detector is not None and hasattr(self.object_detector, "capabilities")
                  else {"model_loaded": bool(self.object_detector), "supported_labels": [],
                        "unsupported_labels": [], "firearm_classes_supported": [],
                        "object_detection_only": True, "weapon_use_action_detection": False})
        return {
            "threat_detection": {
                **threat,
                "model_type": ("custom threat checkpoint" if self.custom_threat_model_configured
                               else "pretrained open-vocabulary detector" if threat.get("open_vocabulary")
                               else "RGB detector label set"),
            },
            "confidence_fusion": {
                "available": self.confidence_fusion is not None,
                "training_data": "LLVIP paired visible/infrared person-detection consensus",
                "scope": "two-confidence-input calibration; not anomaly or threat probability",
                "validation_status": "experimental; verify checkpoint and run provenance",
            },
            "risk_governance": {
                "available": self.governance_head is not None,
                "configuration": "S-DEC Config B (8 features)",
                "trainable_parameters": (sum(p.numel() for p in self.governance_head.parameters())
                                         if self.governance_head is not None else 867),
                "abstention_threshold": (self.governance_head.abstain_threshold
                                         if self.governance_head is not None else 0.45),
                "checkpoint_error": self.governance_load_error,
            },
            "anomaly_detection": {"available": self.action_runtime is not None or self.temporal_fusion_runtime is not None or
                                                os.getenv("RAVEN_ENABLE_MOTION_FALLBACK", "1").strip().lower() in {"1", "true", "yes"},
                                  "trained_model_available": ((self.action_runtime is not None and not self.action_model_synthetic) or
                                                               self.temporal_fusion_scoring_eligible),
                                  "synthetic_demo": self.temporal_fusion_checkpoint_synthetic or self.action_model_synthetic,
                                  "synthetic_action_demo": self.action_model_synthetic,
                                  "scoring_model_available": ((self.action_runtime is not None and not self.action_model_synthetic) or
                                                               self.temporal_fusion_scoring_eligible),
                                  "temporal_checkpoint_role": ("synthetic_architecture_smoke_test_only"
                                      if self.temporal_fusion_checkpoint_synthetic else
                                      "checkpoint_provenance_unknown_scoring_disabled"
                                      if self.temporal_fusion_runtime is not None and not self.temporal_fusion_scoring_eligible else
                                      "experimental_temporal_model_not_independently_validated"
                                      if self.temporal_fusion_runtime is not None else "not_loaded"),
                                  "model": ("TemporalCrossModalFusion" if self.temporal_fusion_scoring_eligible
                                            else "synthetic_temporal_cross_modal_prototype" if self.temporal_fusion_runtime is not None and self.temporal_fusion_checkpoint_synthetic
                                            else "unverified_temporal_checkpoint" if self.temporal_fusion_runtime is not None
                                            else "TemporalPoseAnomalyHead" if self.action_runtime is not None and not self.action_model_synthetic
                                            else "synthetic_pose_motion_baseline" if self.action_runtime is not None
                                            else None),
                                  "method": ("synthetic_temporal_cross_modal_prototype" if self.temporal_fusion_runtime is not None and self.temporal_fusion_checkpoint_synthetic
                                             else "unverified_temporal_checkpoint_disabled" if self.temporal_fusion_runtime is not None and not self.temporal_fusion_scoring_eligible
                                             else "temporal_cross_modal_attention_gru" if self.temporal_fusion_runtime is not None
                                             else "synthetic_pose_motion_demo" if self.action_model_synthetic
                                             else "weakly_supervised_temporal_pose_gru" if self.action_runtime is not None
                                             else "rule_based_motion_outlier"),
                                  "action_model_available": self.action_runtime is not None,
                                  "action_checkpoint": str(Path(os.getenv("RAVEN_ACTION_MODEL_WEIGHTS",
                                      str(PROJECT_ROOT / "data" / "models" / "action_anomaly.pt")).strip()).resolve()),
                                  "action_model_error": self.action_model_error,
                                  "reason": ("Synthetic temporal weights are loaded only to demonstrate architecture and data flow; they are disabled for review scoring. Motion cues are heuristic."
                                             if self.temporal_fusion_checkpoint_synthetic else
                                             "Temporal checkpoint provenance is missing; its scores are disabled. Motion cues are heuristic."
                                             if self.temporal_fusion_runtime is not None and not self.temporal_fusion_scoring_eligible else
                                             "Synthetic pose/motion weights are loaded only as a software baseline and are disabled for review scoring. Motion cues are heuristic."
                                             if self.action_model_synthetic else
                                             "An experimental temporal model is loaded; scores are review cues and have not been independently validated on target-camera data."
                                             if self.action_runtime is not None or (self.temporal_fusion_runtime is not None and not self.temporal_fusion_checkpoint_synthetic)
                                             else "Motion outlier review cue is active. It is a transparent heuristic, not a trained action model or a threat probability, and does not trigger an alert by itself.")},
            "appearance_reid": {"available": bool(getattr(self.rgb_reid, "enabled", False)),
                                "method": ("custom feature extractor" if getattr(self.rgb_reid, "feature_extractor", None)
                                           else "color and gradient descriptor with video-time, scene-position, and ambiguity gates"),
                                "similarity_threshold": getattr(self.rgb_reid, "similarity_threshold", None),
                                "identifies_person_by_name": False},
            "thermal_face_landmarks": {
                "available": bool(self._thermal_landmarker is not None),
                "enabled": os.getenv("RAVEN_THERMAL_FACE_LANDMARKS", "0").lower() in {"1", "true", "yes"},
                "model": "thermal-face-alignment pretrained T-FAKE landmarker",
                "reason": self._thermal_landmarker_error or "Opt-in; processes real thermal feed only."},
        }

    def _thermal_face_landmarks(self, thermal_frame):
        if self._thermal_landmarker is None and self._thermal_landmarker_error is None:
            try:
                from src.thermal.processing import ThermalLandmarkAdapter
                device = "cuda" if str(self.device_name).startswith("CUDA") else "cpu"
                self._thermal_landmarker = ThermalLandmarkAdapter(device=device, n_landmarks=70)
            except Exception as exc:
                self._thermal_landmarker_error = str(exc)
                LOGGER.warning("Thermal face landmarks unavailable: %s", exc)
        if self._thermal_landmarker is None:
            return None
        return self._thermal_landmarker.process(thermal_frame)

    def _finalize_results(self, results, frame, observations=None, run_object_detection=True,
                          run_threat_detection=True):
        detector_model = getattr(self.object_detector, "model", None)
        shared_model = (detector_model is not None
                        and detector_model is getattr(self.rgb_tracker, "model", None))
        threats = []
        if self.object_detector and run_object_detection and run_threat_detection:
            if shared_model and observations is not None:
                try:
                    threats = self.object_detector.detect(frame, observations=observations)
                except TypeError as exc:
                    # A long-running Streamlit process may still hold a detector
                    # instance created before the observations API was added.
                    if "unexpected keyword argument 'observations'" not in str(exc):
                        raise
                    threats = self.object_detector.detect(frame)
            else:
                threats = self.object_detector.detect(frame)
        people = [item for item in results if item.get("track_id") is not None
                  and item.get("bbox") and not item.get("predicted")]
        for result in results:
            result["threat_objects"] = []
        unmatched = []
        for threat in threats:
            x1, y1, x2, y2 = [float(value) for value in threat["bbox"]]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            containing = []
            for person in people:
                px1, py1, px2, py2 = [float(value) for value in person["bbox"]]
                if px1 <= center[0] <= px2 and py1 <= center[1] <= py2:
                    containing.append((max(1.0, (px2 - px1) * (py2 - py1)), person))
            if containing:
                person = min(containing, key=lambda pair: pair[0])[1]
                person["threat_objects"].append(threat)
                if self.learned_fusion is not None and person.get("source") == "rgb+thermal":
                    identity = person.get("identity", "unknown")
                    match = next((row for row in results if str(row.get("track_id")) == str(person.get("track_id"))), None)
                    if match is not None:
                        try:
                            rgb_data = {"posture_conf": 1.0 if match.get("posture_rgb") != "unknown" else 0.0,
                                        "track_conf": match.get("rgb_confidence", 0.0)}
                            thermal_data = {"posture_conf": 1.0 if match.get("posture_thermal") != "unknown" else 0.0,
                                            "track_conf": match.get("thermal_confidence", 0.0)} if match.get("thermal_track_id") else None
                            learned = self._learned_consensus(
                                {"bbox": match.get("bbox"), "conf": match.get("rgb_confidence", 0)},
                                {"bbox": match.get("bbox_thermal"), "conf": match.get("thermal_confidence", 0)} if match.get("thermal_track_id") else None,
                                match.get("kpts_rgb"), match.get("kpts_thermal"), rgb_data, thermal_data,
                                identity, max(float(obj.get("confidence", 0)) for obj in person["threat_objects"]))
                            if learned is not None:
                                match["fusion_score"], match["evidence"], _ = learned
                        except Exception:
                            LOGGER.exception("Could not reinfer consensus with weapon context")
            else:
                unmatched.append(threat)
        for threat in unmatched:
            results.append({"timestamp": datetime.now(timezone.utc).isoformat(), "track_id": None,
                            "identity": "unknown", "bbox": threat["bbox"], "posture_rgb": "unknown",
                            "risk_score": 0, "anomaly_score": None, "anomaly_model_available": False,
                            "risk_score_type": "object_detector_confidence",
                            "source": "rgb-object-detector", "thermal_sensor_available": False,
                            "threat_objects": [threat], "evidence": {"object_class": threat["label"],
                            "object_confidence": threat["confidence"]}, "needs_review": True})
        for result in results:
            objects = [obj for obj in result.get("threat_objects", [])
                       if float(obj.get("confidence", 0)) >= self.alert_generator.threat_threshold]
            model_score = (result.get("anomaly_score") if result.get("anomaly_model_available") else None)
            consensus_score = (float(result.get("fusion_score", 0))
                               if result.get("evidence", {}).get("fusion_method") == "learned_cross_modal_attention_consensus"
                               else None)
            risk_score = model_score if model_score is not None else consensus_score
            result["risk_score"] = float(risk_score) if risk_score is not None else 0.0
            result["risk_score_type"] = ("synthetic_demo_model_score"
                                          if model_score is not None and result.get("risk_score_type") == "synthetic_demo_model_score"
                                          else "trained_temporal_model" if model_score is not None
                                          else "learned_consensus_probability" if consensus_score is not None
                                          else "object_detector_confidence" if objects else "unavailable")
            if result.get("anomaly_method") == "motion_heuristic_fallback":
                result["risk_score"] = float(result.get("anomaly_score", 0) or 0)
                result["risk_score_type"] = "motion_heuristic_review_priority_not_probability"
            multiplier = float(result.get("identity_multiplier", 1.0))
            motion_only = result.get("risk_score_type") == "motion_heuristic_review_priority_not_probability"
            adjusted_score = (0.0 if motion_only else
                              min(100.0, float(result["risk_score"]) * multiplier))
            result["risk_score_adjusted"] = adjusted_score
            heuristic_review = (motion_only and result.get("anomaly_method") == "motion_heuristic_fallback"
                                and float(result.get("anomaly_score", 0) or 0)
                                >= self.alert_generator.warning_threshold)
            model_review = (risk_score is not None and not motion_only and
                            adjusted_score >= self.alert_generator.warning_threshold)
            governance_review = (result.get("governance_action") == "abstain"
                                or result.get("alert_status") == "review_required")
            result["needs_review"] = bool(objects or heuristic_review or model_review or governance_review)
            # Action/anomaly scores request human review. A high model score by
            # itself is not a confirmed threat alert; ALERT is reserved for a
            # configured object detection crossing its confidence threshold.
            result["alert_status"] = ("alert" if objects else
                                      "review_required" if governance_review else
                                      "review" if heuristic_review or model_review else "low")
        events = self.alert_generator.evaluate([item for item in results
                                                if not item.get("predicted") and
                                                item.get("risk_score_type") != "motion_heuristic_review_priority_not_probability"])
        event_by_track = {event.get("track_id"): event for event in events}
        for result in results:
            result["alert_event"] = event_by_track.get(result.get("track_id"))
        return results

    def set_alert_policy(self, warning_threshold, alert_threshold, threat_threshold=None,
                         trigger_actions=None):
        self.alert_generator.set_thresholds(warning_threshold, alert_threshold, threat_threshold)
        if trigger_actions is not None:
            allowed = {"notify_operator", "log_event"}
            if not set(trigger_actions).issubset(allowed):
                raise ValueError("trigger actions may only include notify_operator and log_event")
            self.alert_generator.trigger_actions = list(trigger_actions)

    def reset_video_state(self):
        """Clear per-video tracking/Re-ID state when playback seeks backwards."""
        for tracker in (self.rgb_tracker, self.thermal_tracker, self.quality_detector):
            if tracker is None:
                continue
            reset = getattr(tracker, "reset", None)
            if reset:
                reset()
        for reid in (self.rgb_reid, self.thermal_reid):
            if reid is not None:
                reid.buffer.clear()
        self.track_states.clear()
        self.thermal_track_states.clear()
        if self.action_runtime is not None:
            self.action_runtime.clear()
        if self.temporal_fusion_runtime is not None:
            self.temporal_fusion_runtime.clear()
        self._last_detection_frame = None
        self._frame_index = 0
        self._frames_since_detector = 0
        self._processed_frame_count = 0
        self._last_pair_output = None
        self._last_pair_frame_index = None
        self.last_visual_boxes = []
        self.alert_generator.last_triggered.clear()

    def process_request(self, payload):
        if not isinstance(payload, dict) or "rgb" not in payload or "thermal" not in payload:
            raise ValueError("request must be a JSON object containing rgb and thermal base64 images")
        frames = [self._decode_image(payload[k], k) for k in ("rgb", "thermal")]
        results = self.process_frame_pair(*frames, zone_context=payload.get("zone_context", 0), metadata=payload.get("metadata"))
        return {"timestamp": datetime.now(timezone.utc).isoformat(), "source": "rgb+thermal",
                "result_count": len(results), "results": results,
                "available_models": {"detector": True, "pose": True,
                                     "st_gcn": False, "patchcore": False,
                                     "action_anomaly_model": self.action_runtime is not None}}

    @staticmethod
    def _decode_image(value, name):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be base64-encoded image data")
        max_bytes = int(os.getenv("RAVEN_MAX_IMAGE_BYTES", 8 * 1024 * 1024))
        if value.startswith("data:") and "," in value:
            header, value = value.split(",", 1)
            declared_mime = header[5:].split(";", 1)[0].lower()
        else:
            declared_mime = None
        if len(value) > ((max_bytes + 2) // 3) * 4:
            raise ValueError(f"{name} exceeds the configured image size limit")
        try:
            encoded = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{name} is not valid base64") from exc
        if not encoded or len(encoded) > max_bytes:
            raise ValueError(f"{name} exceeds the configured image size limit or is empty")
        signatures = {
            "image/jpeg": encoded.startswith(b"\xff\xd8\xff"),
            "image/png": encoded.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/webp": encoded.startswith(b"RIFF") and encoded[8:12] == b"WEBP",
            "image/bmp": encoded.startswith(b"BM"),
            "image/tiff": encoded.startswith((b"II*\x00", b"MM\x00*")),
        }
        actual_mime = next((mime for mime, matches in signatures.items() if matches), None)
        if actual_mime is None:
            raise ValueError(f"{name} must contain a supported JPEG, PNG, WebP, BMP, or TIFF image")
        if declared_mime and declared_mime != actual_mime:
            raise ValueError(f"{name} data URL type does not match its image bytes")
        try:
            with Image.open(BytesIO(encoded)) as image_header:
                pillow_mime = Image.MIME.get(image_header.format)
                if pillow_mime != actual_mime:
                    raise ValueError(f"{name} image header is inconsistent")
                max_pixels = int(os.getenv("RAVEN_MAX_IMAGE_PIXELS", 40_000_000))
                if image_header.width * image_header.height > max_pixels:
                    raise ValueError(f"{name} exceeds the configured image pixel limit")
                image_header.verify()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"{name} image is corrupt or cannot be decoded") from exc
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{name} image is corrupt or cannot be decoded")
        return image

    @staticmethod
    def _json_safe(value):
        if value is None: return None
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, (np.integer, np.floating)): return value.item()
        if isinstance(value, dict): return {str(k): RavenPipeline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [RavenPipeline._json_safe(v) for v in value]
        return value
