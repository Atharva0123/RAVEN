import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.train_action_anomaly import calibrate_threshold
from src.anomaly.action_model import TemporalPoseAnomalyHead
from src.anomaly.action_model import TrackAnomalyRuntime


class _ModelTracker:
    def __init__(self, detections):
        self.detections = detections
        self.model = object()

    def track(self, *_args, **_kwargs):
        return [dict(row) for row in self.detections]

    def reset(self):
        pass

    def extract_person_masks(self, _frame, detections):
        return detections


class _Pose:
    last_confidence = .9

    def get_pose(self, _frame, _bbox):
        return [[20 + i, 30 + i, .9] for i in range(17)], "walking"


class _NoThreats:
    model = None

    def detect(self, *_args, **_kwargs):
        return []


class TestTemporalActionTraining(unittest.TestCase):
    def test_validation_threshold_is_selected_from_both_classes(self):
        result = calibrate_threshold([.1, .2, .8, .9], [0, 0, 1, 1])
        self.assertEqual(result["threshold"], .8)
        self.assertEqual(result["balanced_accuracy"], 1.0)

    def test_checkpoint_round_trips_calibrated_threshold_and_metadata(self):
        model = TemporalPoseAnomalyHead()
        model.review_threshold = 63.5
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "action.pt"
            model.save(path, metadata={"validation_video_auc": .82})
            loaded = TemporalPoseAnomalyHead.load(path)
        self.assertEqual(loaded.review_threshold, 63.5)
        self.assertEqual(loaded.training_metadata["validation_video_auc"], .82)

    def test_track_runtime_warms_up_then_returns_bounded_scores(self):
        model = TemporalPoseAnomalyHead()
        runtime = TrackAnomalyRuntime(model, window=8, minimum_history=4, infer_every=1,
                                      sample_period_seconds=0, device="cpu")
        scores = [runtime.update("p1", np.zeros(60, np.float32), timestamp=i)
                  for i in range(5)]
        self.assertIsNone(scores[2])
        self.assertIsNotNone(scores[3])
        self.assertTrue(all(score is None or 0 <= score <= 100 for score in scores))

    def test_motion_fallback_does_not_replace_a_model_score(self):
        from types import SimpleNamespace
        from src.anomaly.heuristic_fallback import MotionAnomalyFallback
        from src.pipeline import RavenPipeline

        pipeline = RavenPipeline.__new__(RavenPipeline)
        pipeline.action_runtime = None
        pipeline.heuristic_fallback = MotionAnomalyFallback()
        pipeline.alert_generator = SimpleNamespace(warning_threshold=50)
        row = {"track_id": "synthetic", "bbox": [10, 10, 40, 100],
               "anomaly_score": 83, "anomaly_model_available": True,
               "demo_label": "SYNTHETIC_DEMO"}
        pipeline._apply_heuristic_fallback([row], (120, 120, 3))
        self.assertEqual(row["anomaly_score"], 83)

    def test_pipeline_applies_loaded_action_model_to_rgb_only_results(self):
        from src.pipeline import RavenPipeline

        model = TemporalPoseAnomalyHead()
        model.review_threshold = 0
        detections = [{"class_id": 0, "class_name": "person", "bbox": [10, 10, 100, 240],
                       "conf": .95, "track_id": 1}]
        with tempfile.TemporaryDirectory() as folder:
            weights = Path(folder) / "action.pt"
            model.save(weights)
            with patch.dict(os.environ, {"RAVEN_ACTION_MODEL_WEIGHTS": str(weights),
                                         "RAVEN_VIDEO_INFERENCE_STRIDE": "5"}):
                pipeline = RavenPipeline(rgb_tracker=_ModelTracker(detections),
                                         pose_classifier=_Pose(), enable_thermal=False,
                                         object_detector=_NoThreats(), disable_threat_detector=True)
                pipeline.person_filter["min_height_px"] = 10
                pipeline.person_filter["min_height_fraction"] = 0
                pipeline.person_filter["min_width_height_ratio"] = .01
                pipeline.person_filter["min_confirmed_hits"] = 1
                pipeline.video_inference_stride = 1
                pipeline.rgb_reid.enabled = False
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                results = []
                for index in range(10):
                    results = pipeline.process_rgb_frame(
                        frame, metadata={"frame_index": index,
                                         "video_timestamp_seconds": index * .6})
        person = next(row for row in results if row.get("track_id") == "1")
        self.assertTrue(person["anomaly_model_available"])
        self.assertEqual(person["anomaly_method"], "weakly_supervised_temporal_pose_gru")
        self.assertIsNotNone(person["anomaly_score"])
        self.assertEqual(person["risk_score_type"], "trained_temporal_model")
        self.assertEqual(len(person["keypoints_normalized"]), 17)


if __name__ == "__main__":
    unittest.main()
