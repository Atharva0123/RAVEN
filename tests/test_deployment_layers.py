import json
import numpy as np

from src.anomaly.heuristic_fallback import MotionAnomalyFallback
from src.dashboard.payload import generate_dashboard_payload, serialize_dashboard_payload
from src.deployment_loop import BoundingBoxMotionInterpolator
from src.thermal.simulator import ThermalSimulator


def test_payload_sections_are_separate_and_json_safe():
    rows = [{"track_id": 4, "bbox": [1, 2, 10, 20], "rgb_confidence": .8,
             "anomaly_score": 25, "anomaly_method": "motion_heuristic_fallback",
             "anomaly_model_available": False, "risk_score": 25,
             "risk_score_type": "motion_heuristic_review_priority_not_probability",
             "video_timestamp_seconds": 1.0}]
    payload = generate_dashboard_payload(rows, frame_index=3, timestamp_seconds=1.0,
                                         thermal_mode="simulated_from_rgb")
    assert payload["detection_log"]["title"] == "Detection Log: What RAVEN sees"
    assert payload["live_metrics"]["active_tracks"] == 1
    assert payload["scoring_panel"]["anomaly"]["score"] == 25
    assert payload["scoring_panel"]["anomaly"]["is_trained_model"] is False
    assert payload["thermal"]["independent_sensor"] is False
    serialized = serialize_dashboard_payload(payload)
    assert "PROTOTYPE_REVIEW_CUE" in serialized
    assert "RGB-derived visualization and is excluded from fusion" in serialized
    assert json.loads(serialize_dashboard_payload(payload))["schema_version"] == "1.0"


def test_heuristic_fallback_has_bounded_explicit_method():
    model = MotionAnomalyFallback()
    one = model.update("a", (10, 10), 1.0, (100, 100, 3))
    two = model.update("a", (11, 10), 2.0, (100, 100, 3))
    assert one["score"] == 0
    assert 0 <= two["score"] <= 100
    assert two["method"] == "motion_heuristic_fallback"
    assert two["model_available"] is False


def test_thermal_simulator_marks_correlated_source():
    output = ThermalSimulator().transform(np.zeros((32, 32, 3), dtype=np.uint8))
    assert output.frame.shape == (32, 32, 3)
    assert output.metadata["independent_sensor"] is False


def test_box_interpolator_marks_estimated_boxes():
    tracker = BoundingBoxMotionInterpolator()
    tracker.update([{"track_id": "p1", "bbox": (0, 0, 10, 10)}], 0.0)
    tracker.update([{"track_id": "p1", "bbox": (10, 0, 20, 10)}], 1.0)
    predicted = tracker.update([], 1.2)
    assert len(predicted) == 1
    assert predicted[0]["predicted"] is True
    assert predicted[0]["detector_skipped"] is True
