import base64
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image

from src.alerts.generator import AlertGenerator
from src.image_quality import assess_image_quality, compare_detection_sets, enhance_frame
from src.pipeline import RavenPipeline
from src.tracking.tracker import SentryTracker
from src.tracking.state import TrackStateManager
from src.video.export import _annotate, export_annotated_video
from src.video.io import iter_video_frames, video_metadata
from src.video.files import remove_owned_temp_file


class FakeTracker:
    def __init__(self, detections):
        self.detections = detections
        self.model = object()
        self.reset_count = 0
        self.calls = 0

    def track(self, frame, **kwargs):
        self.calls += 1
        return [dict(item) for item in self.detections]

    def reset(self):
        self.reset_count += 1

    def extract_person_masks(self, _frame, detections):
        return detections


class FakeThreatDetector:
    def __init__(self):
        self.model = object()

    def detect(self, frame, observations=None):
        return [{"label": item["class_name"], "class_id": item["class_id"],
                 "confidence": item["conf"], "bbox": item["bbox"]}
                for item in observations or [] if item["class_name"] == "knife"]


class FakePose:
    last_confidence = 0.0

    def __init__(self):
        self.calls = 0

    def get_pose(self, frame, bbox):
        self.calls += 1
        return None, "unknown"


class FakeFusion:
    def evaluate(self, *_args):
        return 0, {"method": "test"}, True


def detection(class_id, name, bbox, conf=.9, track_id=1):
    return {"class_id": class_id, "class_name": name, "bbox": bbox,
            "conf": conf, "track_id": track_id}


class TestRavenPipeline(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.person = detection(0, "person", [10, 10, 100, 250])
        self.knife = detection(43, "knife", [40, 80, 65, 120], .95, 2)

    def pipeline(self, *, thermal=False, detections=None, quality_detector=None, alert_store=None):
        with patch.dict("os.environ", {"RAVEN_VIDEO_INFERENCE_STRIDE": "5"}):
            return self._pipeline(thermal=thermal, detections=detections,
                                  quality_detector=quality_detector, alert_store=alert_store)

    def _pipeline(self, *, thermal=False, detections=None, quality_detector=None, alert_store=None):
        detections = detections if detections is not None else [self.person, self.knife]
        pipe = RavenPipeline(config_path="configs/models.yaml", enable_thermal=thermal,
            rgb_tracker=FakeTracker(detections),
            thermal_tracker=FakeTracker([self.person]) if thermal else None,
            pose_classifier=FakePose(), fusion=FakeFusion(),
            object_detector=FakeThreatDetector(), quality_detector=quality_detector,
            alert_store=alert_store or Path(tempfile.gettempdir()) / "raven-test-alerts.jsonl")
        pipe.object_detector.model = None
        pipe.person_filter["min_confirmed_hits"] = 1
        pipe.person_filter["min_height_px"] = 1
        pipe.person_filter["min_height_fraction"] = 0
        pipe.person_filter["min_width_height_ratio"] = .01
        return pipe

    def test_person_and_configured_object_are_returned_from_same_frame(self):
        pipe = self.pipeline()
        pipe.object_detector.detect = lambda _frame, observations=None: [{
            "label": "knife", "class_id": 43, "confidence": .95,
            "bbox": [40, 80, 65, 120],
        }]
        result = pipe.process_rgb_frame(self.rgb)
        self.assertTrue(any(str(item.get("track_id")) == "1" for item in result))
        self.assertTrue(any(obj["label"] == "knife" for item in result for obj in item["threat_objects"]))

    def test_motion_telemetry_does_not_create_untrained_action_alert(self):
        moving_person = {**self.person, "velocity": (900.0, 900.0),
                         "acceleration": (900.0, 900.0)}
        with patch.dict("os.environ", {"RAVEN_ACTION_MODEL_WEIGHTS": ""}):
            result = self.pipeline(detections=[moving_person]).process_rgb_frame(self.rgb)
        person = next(row for row in result if str(row.get("track_id")) == "1")
        self.assertEqual(person["anomaly_method"], "motion_heuristic_fallback")
        self.assertFalse(person["anomaly_model_available"])
        self.assertFalse(person["needs_review"])
        self.assertEqual(person["risk_score"], 0.0)
        self.assertEqual(person["alert_status"], "low")

    def test_synthetic_temporal_weights_fall_back_to_labeled_motion_cue(self):
        from src.fusion.temporal_cross_modal import TemporalCrossModalFusion, TemporalCrossModalRuntime

        with patch.dict("os.environ", {"RAVEN_ACTION_MODEL_WEIGHTS": "",
                                       "RAVEN_CONFIDENCE_FUSION_WEIGHTS": ""}):
            pipe = self._pipeline(thermal=True, detections=[self.person])
            model = TemporalCrossModalFusion(feature_dim=16)
            model.training_metadata = {"synthetic": True}
            runtime = TemporalCrossModalRuntime(model)
            runtime.update = unittest.mock.Mock(side_effect=AssertionError("synthetic scores must be gated"))
            pipe.temporal_fusion_runtime = runtime
            pipe.temporal_fusion_checkpoint_synthetic = True
            pipe.temporal_fusion_scoring_eligible = False

            rows = pipe.process_frame_pair(self.rgb, self.rgb)

        person = next(row for row in rows if str(row.get("track_id")) == "1")
        runtime.update.assert_not_called()
        self.assertFalse(person["anomaly_model_available"])
        self.assertEqual(person["signal_label"], "PROTOTYPE_REVIEW_CUE")

    def test_performance_profile_caps_pose_work_per_analyzed_frame(self):
        people = [detection(0, "person", [index * 70, 10, index * 70 + 50, 250],
                            conf=0.95 - index * 0.01, track_id=index + 1)
                  for index in range(6)]
        pipe = self.pipeline(detections=people)
        pipe.rgb_reid.enabled = False
        pipe.set_processing_mode("Performance")
        pipe.process_rgb_frame(self.rgb)
        self.assertEqual(pipe.pose_classifier.calls, 4)

    def test_rgb_only_and_dual_feed_paths(self):
        rgb_result = self.pipeline().process_rgb_frame(self.rgb)
        self.assertTrue(all(item.get("thermal_sensor_available") is False for item in rgb_result))
        paired = self.pipeline(thermal=True).process_frame_pair(self.rgb, self.rgb)
        self.assertIsInstance(paired, list)
        with self.assertRaises(ValueError):
            self.pipeline(thermal=False).process_frame_pair(self.rgb, self.rgb)

    def test_image_signature_corruption_and_decode(self):
        image = Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        decoded = RavenPipeline._decode_image(encoded, "rgb")
        self.assertEqual(decoded.shape[:2], (12, 16))
        for bad in ("not-base64", base64.b64encode(b"not an image").decode()):
            with self.assertRaises(ValueError):
                RavenPipeline._decode_image(bad, "rgb")
        with patch.dict("os.environ", {"RAVEN_MAX_IMAGE_PIXELS": "100"}):
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                RavenPipeline._decode_image(encoded, "rgb")

    def test_enhancement_is_opt_in_and_comparison_is_separate(self):
        frame = np.full((80, 120, 3), 20, dtype=np.uint8)
        enhanced, quality = enhance_frame(frame, upscale_factor=1.25)
        self.assertEqual(enhanced.shape[:2], (100, 150))
        self.assertFalse(quality["detail_restoration_claimed"])
        self.assertIn("low_light", assess_image_quality(frame)["issues"])
        original = [detection(0, "person", [1, 1, 20, 20])]
        comp = compare_detection_sets(original, original)
        self.assertEqual(comp["matched_count"], 1)
        self.assertTrue(comp["enhanced_results_are_uncertain"])
        pipe = self.pipeline(detections=[self.person], quality_detector=FakeTracker([self.person, self.knife]))
        result = pipe.process_rgb_frame(self.rgb, enhanced_frame=self.rgb)[0]
        self.assertIn("enhanced_detection_comparison", result)
        self.assertEqual(result["enhanced_detection_comparison"]["original_detection_count"], 1)

    def test_performance_mode_skips_detector_but_keeps_short_track_state(self):
        pipe = self.pipeline()
        pipe.video_inference_stride = 1
        pipe.set_processing_mode("Performance")
        first = pipe.process_rgb_frame(self.rgb, metadata={"frame_index": 0, "video_timestamp_seconds": 0})
        second = pipe.process_rgb_frame(self.rgb, metadata={"frame_index": 1, "video_timestamp_seconds": .04})
        third = pipe.process_rgb_frame(self.rgb, metadata={"frame_index": 2, "video_timestamp_seconds": .08})
        self.assertEqual(pipe.rgb_tracker.calls, 2)
        first_person = next(row for row in first if row.get("track_id") == "1")
        skipped_person = next(row for row in second if row.get("track_id") == "1")
        recovered_person = next(row for row in third if row.get("track_id") == "1")
        self.assertFalse(first_person["detection_skipped"])
        self.assertTrue(skipped_person["detection_skipped"])
        self.assertEqual(skipped_person["occlusion_state"], "OCCLUDED")
        self.assertEqual(recovered_person["occlusion_state"], "REACQUIRED")


class TestTrackerReset(unittest.TestCase):
    def test_none_tracker_state_is_removed_not_set_to_none(self):
        predictor = SimpleNamespace(trackers=None)
        tracker = SentryTracker(model=SimpleNamespace(predictor=predictor))
        tracker.reset()
        self.assertFalse(hasattr(predictor, "trackers"))
        predictor.trackers = None
        tracker._ensure_tracker_state()
        self.assertFalse(hasattr(predictor, "trackers"))

    def test_reset_resets_tracker_in_place(self):
        tracker_state = SimpleNamespace(reset=lambda: setattr(tracker_state, "was_reset", True))
        predictor = SimpleNamespace(trackers=[tracker_state], vid_path=["old"])
        sentry = SentryTracker(model=SimpleNamespace(predictor=predictor))
        sentry.reset()
        self.assertTrue(tracker_state.was_reset)
        self.assertEqual(predictor.vid_path, [None])
        self.assertIs(predictor.trackers[0], tracker_state)


class TestTrackState(unittest.TestCase):
    def test_cached_pose_is_normalized_relative_to_box_origin(self):
        manager = TrackStateManager()
        det = detection(0, "person", [100, 50, 200, 250])
        manager.update([det], 0.0)
        manager.record_pose("1", [[125, 100, .9]], "standing/walking", .9, 0.0, det["bbox"])
        point = manager.tracks["1"].pose["normalized_keypoints"][0]
        self.assertEqual(point, [.25, .25, .9])

    def test_histories_are_bounded_and_stale_tracks_expire(self):
        manager = TrackStateManager(temporal_window=2, stale_after_seconds=1)
        det = detection(0, "person", [0, 0, 10, 20])
        manager.update([det], 0.0)
        manager.record_pose("1", [[1, 2]], "standing/walking", .8, 0, det["bbox"])
        manager.record_pose("1", [[2, 3]], "crouching", .7, .1, det["bbox"])
        manager.record_pose("1", [[3, 4]], "lying", .6, .2, det["bbox"])
        state = manager.tracks["1"]
        self.assertEqual(state.trajectory[0]["center"], [5.0, 20.0])
        self.assertEqual(state.trajectory.maxlen, 20)
        self.assertEqual(len(state.pose_history), 2)
        self.assertEqual(len(state.behavior_history), 2)
        normalized_point = state.pose_history[-1]["keypoints"][0]
        self.assertEqual(len(normalized_point), 3)
        self.assertTrue(all(0 <= value <= 1 for value in normalized_point))
        self.assertEqual(manager.predict(.5)[0]["occlusion_state"], "OCCLUDED")
        self.assertEqual(manager.predict(2.0), [])
        self.assertIn("1", manager.tracks)
        self.assertEqual(manager.tracks["1"].status, "LOST")
        self.assertEqual(manager.predict(3.0), [])
        self.assertEqual(manager.tracks, {})


class TestStreamingVideo(unittest.TestCase):
    def test_trajectory_toggle_removes_path_from_rendered_frame(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        rows = [{"trajectory": [{"center": [10, 90]}, {"center": [110, 90]}]}]
        shown = _annotate(frame, rows, 0.0, show_trajectories=True)
        hidden = _annotate(frame, rows, 0.0, show_trajectories=False)
        self.assertTrue(np.any(shown[90, 10:111] != hidden[90, 10:111]))
        self.assertTrue(np.array_equal(hidden[90, 10:111], frame[90, 10:111]))

    def test_temp_cleanup_never_deletes_demo_video(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            demo = root / "data" / "demo.mp4"
            demo.parent.mkdir()
            demo.write_bytes(b"video")
            self.assertFalse(remove_owned_temp_file(demo, temp_root=root / "tmp"))
            self.assertTrue(demo.exists())

    def test_temp_cleanup_retries_short_windows_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "upload.mp4"
            path.write_bytes(b"video")
            original_unlink = Path.unlink
            attempts = iter((PermissionError("locked"), None))

            def unlink_with_transient_lock(candidate, **kwargs):
                outcome = next(attempts)
                if outcome:
                    raise outcome
                return original_unlink(candidate, **kwargs)

            with patch("src.video.files.Path.unlink", new=unlink_with_transient_lock):
                self.assertTrue(remove_owned_temp_file(path, temp_root=folder, sleep=lambda _delay: None))
            self.assertFalse(path.exists())

    def test_iterator_and_annotated_export_process_incrementally(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.avi"
            output = Path(folder) / "annotated.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"),
                                     10.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("OpenCV MJPG video writer is unavailable")
            for index in range(4):
                frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            capture = cv2.VideoCapture(str(source))
            metadata = video_metadata(capture)
            capture.release()
            self.assertEqual((metadata.width, metadata.height), (64, 48))
            frames = iter_video_frames(source)
            self.assertEqual([item.index for item in frames], [0, 1, 2, 3])
            fixture = TestRavenPipeline()
            fixture.setUp()
            result = export_annotated_video(source, output, fixture.pipeline(), mode="Accuracy")
            self.assertEqual(result["frames_written"], 4)
            self.assertTrue(output.exists())


class TestAlerts(unittest.TestCase):
    def test_thresholds_and_object_alert_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            alert = AlertGenerator(path, warning_threshold=40, alert_threshold=70,
                                  threat_threshold=.5, cooldown_seconds=0)
            low = {"risk_score": 39, "threat_objects": []}
            warning = {"risk_score": 45, "threat_objects": [], "track_id": "p1"}
            threat = {"risk_score": 10, "threat_objects": [{"label": "knife", "confidence": .8}],
                      "track_id": "p2", "timestamp": "now"}
            events = alert.evaluate([low, warning, threat])
            self.assertEqual(low["alert_status"], "low")
            self.assertEqual(warning["alert_status"], "review")
            self.assertEqual(threat["alert_status"], "alert")
            self.assertEqual(warning["alert_event"]["level"], "REVIEW")
            self.assertEqual(threat["alert_event"]["level"], "ALERT")
            self.assertIn("object candidate crossed", threat["alert_event"]["reason"])
            self.assertEqual(threat["alert_event"]["signal_label"], "OBJECT_DETECTION_CANDIDATE")
            self.assertTrue(threat["alert_event"]["operator_review_required"])
            self.assertFalse(threat["alert_event"]["confirmed_threat"])
            stored = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(item["reason"].startswith("configured object candidate crossed")
                                for item in stored))
            self.assertEqual(len(events), 2)

    def test_threshold_configuration_rejects_invalid_order(self):
        with self.assertRaises(ValueError):
            AlertGenerator("unused.jsonl", warning_threshold=70, alert_threshold=60)


if __name__ == "__main__":
    unittest.main()
