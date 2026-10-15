"""YOLO object tracker with one isolated tracker instance per video stream."""

from pathlib import Path
import os
from contextlib import nullcontext

import cv2
from ultralytics import YOLO
from src.device import select_torch_device, select_onnx_device


class SentryTracker:
    def __init__(self, model_path="yolo26l.pt", tracker_yaml="botsort.yaml", model=None,
                 image_size=640, iou_threshold=0.45):
        self.model_path = str(model_path)
        self.tracker_yaml = tracker_yaml
        self.track_buffer = 90
        self.match_threshold = 0.75
        self.image_size = int(image_size)
        self.iou_threshold = float(iou_threshold)
        self.device, self.use_half = select_torch_device()
        if model is None and Path(self.model_path).suffix.lower() == ".onnx":
            self.use_half = False
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("ONNX weights require onnxruntime or onnxruntime-gpu") from exc
            available = ort.get_available_providers()
            self.device, _provider = select_onnx_device(available)
        self.model = model if model is not None else YOLO(self.model_path)

    def track(self, frame, conf_thresh=0.45, classes=None):
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("frame must be a non-empty image")
        # Reduce host-to-model work for large inputs. Keep tracker/output
        # coordinates in the caller's original frame space.
        source_h, source_w = frame.shape[:2]
        max_side = max(32, int(os.getenv("RAVEN_MAX_INFERENCE_SIDE", "640")))
        scale = min(1.0, max_side / float(max(source_h, source_w)))
        inference_frame = frame
        if scale < 1.0:
            target_w = max(1, int(round(source_w * scale)))
            target_h = max(1, int(round(source_h * scale)))
            inference_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self._ensure_tracker_state()
        options = {"persist": True, "tracker": self.tracker_yaml,
                   "conf": conf_thresh, "iou": self.iou_threshold,
                   "imgsz": self.image_size, "verbose": False}
        if self.device:
            options["device"] = self.device
        if self.use_half:
            options["quantize"] = 16
        if classes is not None:
            options["classes"] = classes
        try:
            import torch
            inference_context = torch.inference_mode()
        except ImportError:
            inference_context = nullcontext()
        try:
            with inference_context:
                results = self.model.track(inference_frame, **options)
        except Exception as exc:
            try:
                import torch
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
            except (ImportError, AttributeError):
                is_oom = "out of memory" in str(exc).lower()
            if is_oom:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise RuntimeError("CUDA out of memory during detection; lower image_size or select CPU") from exc
            raise
        tracked = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            ids = (boxes.id.int().detach().cpu().tolist()
                   if boxes.id is not None else [None] * len(boxes))
            names = getattr(result, "names", getattr(self.model, "names", {}))
            for index, box in enumerate(boxes):
                class_id = int(box.cls[0].detach().cpu().item())
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                bbox = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
                if scale < 1.0:
                    bbox = [bbox[0] / scale, bbox[1] / scale,
                            bbox[2] / scale, bbox[3] / scale]
                row = {"bbox": bbox,
                                "track_id": int(ids[index]) if ids[index] is not None else None,
                                "conf": float(box.conf[0].detach().cpu().item()), "class_id": class_id,
                                "class_name": str(class_name).lower()}
                tracked.append(row)
        return tracked


    def reset(self):
        """Reset IDs in place; Ultralytics requires ``trackers`` to stay a list."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        trackers = getattr(predictor, "trackers", None)
        if trackers is None:
            # Ultralytics' persistent on_predict_start callback checks hasattr,
            # so a present-but-None attribute prevents tracker initialization.
            try:
                delattr(predictor, "trackers")
            except AttributeError:
                pass
            return
        for tracker in trackers:
            reset = getattr(tracker, "reset", None)
            if reset:
                reset()
        if hasattr(predictor, "vid_path"):
            predictor.vid_path = [None] * len(trackers)

    def _ensure_tracker_state(self):
        """Recover old/broken predictor state before Ultralytics callbacks run."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is None or not hasattr(predictor, "trackers"):
            return
        trackers = predictor.trackers
        if trackers is None or len(trackers) == 0:
            try:
                delattr(predictor, "trackers")
            except AttributeError:
                # A fake/immutable predictor cannot have the stale attribute
                # removed. Avoid assigning None or an empty list that makes
                # Ultralytics' persistent callback index trackers[0].
                try:
                    predictor.trackers = [item for item in trackers or [] if item is not None]
                except Exception:
                    pass
