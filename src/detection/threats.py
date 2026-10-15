"""Configurable detector for weapon and prohibited-object classes."""

import os
from contextlib import nullcontext
from pathlib import Path
from src.device import select_torch_device, select_onnx_device


class ThreatObjectDetector:
    """Detect configured threat labels with a YOLO-compatible model.

    The default YOLO-World checkpoint uses configured text prompts without
    project-specific training. Prompt detections are best-effort candidates,
    not proof that an object is a weapon or that a person intends harm.
    """

    def __init__(self, model=None, model_path=None, labels=None, confidence=0.25,
                 image_size=640, iou_threshold=0.45):
        self.device, self.use_half = select_torch_device()
        if model is None:
            if not model_path:
                self.model = None
            else:
                # Ultralytics downloads this named pretrained checkpoint on
                # first use. It supports prompt classes without fine-tuning.
                model_source = str(model_path)
                if (not Path(model_source).is_file()
                        and Path(model_source).name == "yolov8s-worldv2.pt"):
                    Path(model_source).parent.mkdir(parents=True, exist_ok=True)
                if Path(model_source).suffix.lower() == ".onnx":
                    try:
                        import onnxruntime as ort
                    except ImportError as exc:
                        raise RuntimeError("ONNX threat weights require onnxruntime or onnxruntime-gpu") from exc
                    available = ort.get_available_providers()
                    self.device, _provider = select_onnx_device(available)
                    self.use_half = False
                from ultralytics import YOLO
                self.model = YOLO(model_source)
        else:
            self.model = model
        self.labels = {str(label).strip().lower() for label in (labels or [
            "knife", "scissors", "gun", "weapon", "firearm", "handgun",
            "pistol", "revolver", "rifle", "shotgun", "explosive",
            "machete", "sword", "baseball bat", "crowbar",
        ])}
        model_name = Path(str(model_path or "")).name
        checkpoint_name = Path(str(getattr(self.model, "ckpt_path", ""))).name
        self.open_vocabulary = bool(
            self.model is not None
            and (model_name == "yolov8s-worldv2.pt" or checkpoint_name == "yolov8s-worldv2.pt")
        )
        if self.open_vocabulary and hasattr(self.model, "set_classes"):
            self.prompt_labels = sorted(self.labels)
            self.model.set_classes(self.prompt_labels)
        else:
            self.prompt_labels = []
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.iou_threshold = float(iou_threshold)

    def capabilities(self):
        """Describe only classes this model actually exposes in its label map."""
        names = getattr(self.model, "names", {}) if self.model is not None else {}
        if isinstance(names, dict):
            model_labels = {str(label).strip().lower() for label in names.values()}
        elif isinstance(names, (list, tuple)):
            model_labels = {str(label).strip().lower() for label in names}
        else:
            model_labels = set()
        supported = sorted(self.labels & model_labels)
        unsupported = sorted(self.labels - model_labels)
        firearm_terms = {"gun", "weapon", "firearm", "handgun", "pistol", "revolver",
                         "rifle", "shotgun"}
        return {
            "model_loaded": self.model is not None,
            "supported_labels": supported,
            "unsupported_labels": unsupported,
            "firearm_classes_supported": sorted(self.labels & model_labels & firearm_terms),
            "object_detection_only": True,
            "weapon_use_action_detection": False,
            "open_vocabulary": self.open_vocabulary,
            "detection_note": ("Pretrained open-vocabulary prompts; detections require human review"
                               if self.open_vocabulary else None),
        }

    def detect(self, frame, observations=None):
        if observations is not None:
            return [{"label": str(item.get("class_name", "")).lower(),
                     "class_id": int(item.get("class_id", -1)),
                     "confidence": float(item.get("conf", item.get("confidence", 0))),
                     "bbox": [float(v) for v in item["bbox"]]}
                    for item in observations
                    if str(item.get("class_name", "")).lower() in self.labels
                    and float(item.get("conf", item.get("confidence", 0))) >= self.confidence]
        if self.model is None:
            return []
        names = getattr(self.model, "names", {})
        options = {"conf": self.confidence, "iou": self.iou_threshold,
                   "imgsz": self.image_size, "verbose": False}
        if self.device:
            options["device"] = self.device
        if self.use_half:
            options["quantize"] = 16
        try:
            import torch
            context = torch.inference_mode()
        except ImportError:
            context = nullcontext()
        with context:
            results = self.model.predict(frame, **options)
        threats = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].detach().cpu().item())
                label = (names.get(class_id, str(class_id)) if isinstance(names, dict)
                         else names[class_id] if isinstance(names, (list, tuple)) and class_id < len(names)
                         else str(class_id))
                label = str(label).lower()
                if label not in self.labels:
                    continue
                threats.append({"label": label, "class_id": class_id,
                                "confidence": float(box.conf[0].detach().cpu().item()),
                                "bbox": box.xyxy[0].detach().cpu().numpy().astype(float).tolist()})
        return threats
