"""Threshold-based alert generation with auditable JSONL event storage."""
import json
import logging
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class AlertGenerator:
    def __init__(self, event_log, warning_threshold=50, alert_threshold=70,
                 threat_threshold=0.35, cooldown_seconds=5.0,
                 trigger_actions=None, clock=time.time):
        self.event_log = Path(event_log)
        self.clock = clock
        self.cooldown_seconds = float(cooldown_seconds)
        self.last_triggered = {}
        self.trigger_actions = list(["notify_operator", "log_event"] if trigger_actions is None else trigger_actions)
        self.set_thresholds(warning_threshold, alert_threshold, threat_threshold)

    def set_thresholds(self, warning_threshold, alert_threshold, threat_threshold=None):
        warning, alert = int(warning_threshold), int(alert_threshold)
        if not 0 <= warning < alert <= 100:
            raise ValueError("thresholds must satisfy 0 <= warning < alert <= 100")
        self.warning_threshold, self.alert_threshold = warning, alert
        if threat_threshold is not None:
            self.threat_threshold = float(threat_threshold)
            if not 0 <= self.threat_threshold <= 1:
                raise ValueError("threat confidence threshold must be between 0 and 1")

    def evaluate(self, results):
        events = []
        for result in results:
            objects = [item for item in result.get("threat_objects", [])
                       if float(item.get("confidence", 0)) >= self.threat_threshold]
            risk = float(result.get("risk_score", 0))
            if objects:
                level = "ALERT"
                reason = "configured object candidate crossed the review threshold; confirm it in the source video"
            elif risk >= self.warning_threshold:
                level = "REVIEW"
                reason = ("Experimental action/anomaly score crossed its review threshold; human review is required."
                          if result.get("model_validation_status") == "synthetic_unvalidated_demo"
                          else "Action/anomaly score crossed its review threshold; human review is required.")
            else:
                result["alert_status"] = "low"
                result["trigger_actions"] = []
                continue
            result["alert_status"] = level.lower()
            reason_key = "configured object class detected" if objects else "action/anomaly review threshold crossed"
            event_key = (result.get("track_id"), level, reason_key)
            wall_time = self.clock()
            video_time = result.get("video_timestamp_seconds")
            cooldown_time = float(video_time) if video_time is not None else wall_time
            if cooldown_time - self.last_triggered.get(event_key, float("-inf")) < self.cooldown_seconds:
                result["trigger_actions"] = []
                continue
            signal_label = ("OBJECT_DETECTION_CANDIDATE" if objects else
                            result.get("signal_label") or
                            ("SYNTHETIC_DEMO" if result.get("model_validation_status") == "synthetic_unvalidated_demo"
                             else "PROTOTYPE_REVIEW_CUE"))
            event = {"timestamp": result.get("timestamp"), "event_time": wall_time,
                     "event_id": f"{result.get('track_id', 'scene')}:{int(cooldown_time * 1000)}",
                     "level": level, "reason": reason,
                     "signal_label": signal_label,
                     "operator_review_required": True,
                     "confirmed_threat": False,
                     "model_validation_status": result.get("model_validation_status"),
                     "demo_label": ("SYNTHETIC_DEMO" if result.get("model_validation_status") == "synthetic_unvalidated_demo" else None),
                     "track_id": result.get("track_id"), "risk_score": risk,
                     "thresholds": {"warning": self.warning_threshold, "alert": self.alert_threshold,
                                    "threat_confidence": self.threat_threshold},
                     "threat_objects": objects, "source": result.get("source"),
                     "actions": self.trigger_actions}
            self.last_triggered[event_key] = cooldown_time
            result["trigger_actions"] = list(self.trigger_actions)
            result["alert_event"] = event
            if "log_event" in self.trigger_actions:
                self._store(event)
            if "notify_operator" in self.trigger_actions:
                events.append(event)
        return events

    def _store(self, event):
        try:
            self.event_log.parent.mkdir(parents=True, exist_ok=True)
            with self.event_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
        except OSError:
            LOGGER.exception("Alert event could not be written to %s", self.event_log)
