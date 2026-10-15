"""Thermal visualization and optional palette-to-temperature conversion.

Palette conversion is an approximate mapping and is only meaningful when the
source palette and its displayed temperature range are known.
"""
from __future__ import annotations

import cv2
import numpy as np


class ThermalLandmarkAdapter:
    """Lazy adapter around pretrained thermal-face-alignment landmarks."""
    def __init__(self, device: str = "cpu", n_landmarks: int = 70):
        try:
            from tfan import ThermalLandmarks
        except ImportError as exc:
            raise RuntimeError("Install thermal-face-alignment to enable thermal face landmarks") from exc
        self._model = ThermalLandmarks(device=device, n_landmarks=n_landmarks)
        self.n_landmarks = int(n_landmarks)

    def process(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        landmarks, confidences = self._model.process(gray, mode="pixel", multi=False)
        if landmarks is None:
            return []
        # multi=False returns one (N,2) set and one (N,) confidence vector.
        points = np.asarray(landmarks, dtype=np.float32)
        conf = (np.asarray(confidences, dtype=np.float32).reshape(-1)
                if confidences is not None else np.ones(len(points), dtype=np.float32))
        return {"landmarks": points.tolist(), "confidences": conf.tolist(),
                "mean_confidence": float(conf.mean()) if conf.size else 0.0}


def render_thermal_feed(rgb_frame: np.ndarray) -> np.ndarray:
    """Create a display-only thermal-style view from an RGB frame."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY) if rgb_frame.ndim == 3 else rgb_frame
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)


def palette_temperature_map(frame: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Refuse temperature estimates when the source palette cannot be decoded."""
    return None, {"available": False, "method": None,
                  "reason": "Palette-to-temperature conversion needs a verified palette decoder or radiometric input."}


def estimate_palette_image(path: str, minimum: float, maximum: float) -> np.ndarray:
    """Use thermaPy for single palette images; video streams use separate frames."""
    try:
        import thermaPy
    except ImportError as exc:
        raise RuntimeError("Install thermaPy to enable thermal palette image analysis") from exc
    converter = getattr(thermaPy, "imgtotempmat", None)
    if converter is None:
        raise RuntimeError("Installed thermaPy has no imgtotempmat converter")
    return np.asarray(converter(path, float(minimum), float(maximum)), dtype=np.float32)
