"""Pose inference and conservative geometric posture labels."""

import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO
from src.device import select_torch_device, select_onnx_device


class PostureClassifier:
    def __init__(self, model_path="yolo26l-pose.pt", conf_thresh=0.5, model=None):
        model_path = os.getenv("RAVEN_POSE_MODEL", str(model_path))
        self.device, self.use_half = select_torch_device()
        if model is None and Path(model_path).suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("ONNX pose weights require onnxruntime or onnxruntime-gpu") from exc
            available = ort.get_available_providers()
            self.device, _provider = select_onnx_device(available)
            self.use_half = False
        self.model = model if model is not None else YOLO(str(model_path))
        self.conf_thresh = float(conf_thresh)
        self.last_confidence = 0.0

    def get_pose(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        height, width = frame.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        self.last_confidence = 0.0
        if crop.size == 0:
            return None, "unknown"
        max_side = max(32, int(os.getenv("RAVEN_MAX_INFERENCE_SIDE", "640")))
        crop_scale = min(1.0, max_side / float(max(crop.shape[:2])))
        if crop_scale < 1.0:
            crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * crop_scale))),
                                     max(1, int(round(crop.shape[0] * crop_scale)))),
                              interpolation=cv2.INTER_AREA)
        options = {"conf": self.conf_thresh, "verbose": False}
        if self.device:
            options["device"] = self.device
        if getattr(self, "use_half", False):
            options["quantize"] = 16
        try:
            import torch
            context = torch.inference_mode()
        except ImportError:
            context = nullcontext()
        with context:
            results = self.model(crop, **options)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None, "unknown"
        points = results[0].keypoints.xy.detach().cpu().numpy()[0]
        if crop_scale < 1.0:
            points = points / crop_scale
        points[:, 0] += x1
        points[:, 1] += y1
        if getattr(results[0].keypoints, "conf", None) is not None:
            conf = results[0].keypoints.conf.detach().cpu().numpy()[0]
            valid = conf >= self.conf_thresh
            self.last_confidence = float(np.mean(conf[valid])) if np.any(valid) else 0.0
        else:
            valid = np.all(points > 0, axis=1)
            self.last_confidence = 0.5 if np.any(valid) else 0.0
        points = points.astype(float)
        visible = lambda ids: all(i < len(points) and valid[i] for i in ids)
        posture = "unknown"
        if visible([5, 6, 11, 12]):
            shoulders = (points[5] + points[6]) / 2
            hips = (points[11] + points[12]) / 2
            dx, dy = hips - shoulders
            spine_angle = abs(math.degrees(math.atan2(dy, dx)))
            person_h = max(1, y2 - y1)
            person_w = max(1, x2 - x1)
            aspect = person_h / person_w
            if visible([15, 16]):
                ankles = (points[15] + points[16]) / 2
                knee_ids = [13, 14]
                if visible(knee_ids):
                    knees = (points[13] + points[14]) / 2
                    if ankles[1] < knees[1] - 0.12 * person_h:
                        posture = "crawling"
            if posture == "unknown":
                if spine_angle < 30 or aspect < 0.85:
                    posture = "lying"
                elif aspect > 1.25 and spine_angle > 55:
                    posture = "standing/walking"
                elif spine_angle > 45:
                    posture = "climbing"
                else:
                    posture = "crouching"
        return points, posture
