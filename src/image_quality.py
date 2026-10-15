"""Opt-in, conservative image enhancement and frame quality measurements.

Enhancement improves visibility in some conditions; it cannot restore detail that
was never captured. The original frame should remain the evidence/display source.
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def estimate_frame_quality(brightness_mean: float, blur_variance: float,
                           width: int, height: int) -> float:
    """Return a bounded descriptive frame-quality heuristic (not learned fusion)."""
    brightness = max(0.0, 1.0 - abs(float(brightness_mean) - 120.0) / 120.0)
    detail = min(1.0, max(0.0, float(blur_variance) / 150.0))
    resolution = min(1.0, max(0.0, min(int(width), int(height)) / 480.0))
    return min(1.0, max(0.0, 0.45 * brightness + 0.35 * detail + 0.20 * resolution))


def assess_image_quality(frame: np.ndarray) -> dict:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty image array")
    if frame.ndim == 2:
        gray = frame
    elif frame.shape[2] == 1:
        gray = frame[:, :, 0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    height, width = gray.shape[:2]
    issues = []
    if blur < 35:
        issues.append("blur_or_low_detail")
    if brightness < 45:
        issues.append("low_light")
    if min(width, height) < 320:
        issues.append("low_resolution")
    return {"width": width, "height": height, "brightness_mean": round(brightness, 2),
            "laplacian_variance": round(blur, 2), "issues": issues,
            "reliability_warning": bool(issues)}


def enhance_frame(frame: np.ndarray, *, upscale_factor: float = 1.0) -> tuple[np.ndarray, dict]:
    """Apply mild luminance equalization and denoising, optionally resize.

    No generative deblurring is used; upscaling is interpolation only.
    """
    quality = assess_image_quality(frame)
    started = time.perf_counter()
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        gray = frame if frame.ndim == 2 else frame[:, :, 0]
        lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    else:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    output = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    output = cv2.fastNlMeansDenoisingColored(output, None, 3, 3, 7, 21)
    if upscale_factor < 1 or upscale_factor > 2:
        raise ValueError("upscale_factor must be between 1.0 and 2.0")
    if upscale_factor > 1:
        output = cv2.resize(output, None, fx=upscale_factor, fy=upscale_factor,
                            interpolation=cv2.INTER_CUBIC)
    quality.update({"enhancement_applied": True, "enhancement_method": "CLAHE + mild denoising",
                    "upscale_factor": float(upscale_factor),
                    "enhancement_ms": round((time.perf_counter() - started) * 1000, 2),
                    "detail_restoration_claimed": False})
    return output, quality


def compare_detection_sets(original: list[dict], enhanced: list[dict], iou_threshold: float = 0.5) -> dict:
    """Summarize detection agreement without promoting enhanced-only detections."""
    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        return inter / max(1e-9, area_a + area_b - inter)
    matched = set()
    matches = []
    for raw in original:
        candidates = [(iou(raw["bbox"], item["bbox"]), index, item)
                      for index, item in enumerate(enhanced)
                      if index not in matched and raw.get("class_id") == item.get("class_id")]
        if candidates:
            overlap, index, item = max(candidates)
            if overlap >= iou_threshold:
                matched.add(index)
                matches.append({"class_id": raw.get("class_id"), "iou": round(overlap, 3),
                                "original_confidence": raw.get("conf"),
                                "enhanced_confidence": item.get("conf")})
    return {"original_detection_count": len(original), "enhanced_detection_count": len(enhanced),
            "matched_count": len(matches), "matches": matches,
            "enhanced_only_count": len(enhanced) - len(matched),
            "original_only_count": len(original) - len(matches),
            "enhanced_results_are_uncertain": True}
