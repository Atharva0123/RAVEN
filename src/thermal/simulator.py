"""RGB-derived thermal-style display simulation (not independent IR evidence)."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ThermalSimulation:
    frame: np.ndarray
    metadata: dict


class ThermalSimulator:
    def __init__(self, *, colormap=cv2.COLORMAP_INFERNO):
        self.colormap = int(colormap)

    def transform(self, rgb_frame):
        if not isinstance(rgb_frame, np.ndarray) or rgb_frame.size == 0:
            raise ValueError("rgb_frame must be a non-empty image")
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY) if rgb_frame.ndim == 3 else rgb_frame
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        pseudo = cv2.applyColorMap(enhanced, self.colormap)
        return ThermalSimulation(pseudo, {"mode": "simulated_from_rgb", "independent_sensor": False,
                                         "temperature_calibrated": False,
                                         "warning": "Simulated thermal-style view is derived from RGB; it is not IR sensor evidence."})
