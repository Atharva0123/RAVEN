"""Celery tasks for Redis-backed asynchronous image batches."""
from __future__ import annotations

import logging
import json
import os

from celery import Celery

from src.observability import InferenceAudit, configure_json_logging
from src.pipeline import RavenPipeline

LOGGER = logging.getLogger(__name__)
configure_json_logging()
_pipeline = None
_video_pipeline = None
_audit = InferenceAudit()

redis_url = os.getenv("RAVEN_REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "raven",
    broker=redis_url,
    backend=os.getenv("RAVEN_CELERY_RESULT_BACKEND", redis_url),
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=3600,
)


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RavenPipeline()
    return _pipeline


def _get_video_pipeline():
    global _video_pipeline
    if _video_pipeline is None:
        _video_pipeline = RavenPipeline(enable_thermal=False)
    return _video_pipeline


@celery_app.task(name="raven.analyze_batch", bind=True)
def analyze_batch_task(self, payloads):
    """Analyze a bounded batch of base64 RGB/thermal image pairs."""
    if not isinstance(payloads, list) or not 1 <= len(payloads) <= 32:
        raise ValueError("payloads must contain between 1 and 32 image pairs")
    pipeline = _get_pipeline()
    results = []
    total = len(payloads)
    for index, payload in enumerate(payloads):
        self.update_state(state="PROGRESS", meta={"completed": index, "total": total})
        result = pipeline.process_request(payload)
        results.append(result)
        if _audit.backend != "disabled":
            _audit.record(result)
    return {"count": len(results), "results": results}


def _publish_job_status(job_id, status, progress, **details):
    from src.metadata import update_job
    payload = {"type": details.get("event_type", "job"), "job_id": job_id, "status": status,
               "progress": round(float(progress), 4)}
    for key in ("video_id", "completed_frames", "total_frames", "summary", "error", "alert_event"):
        if key in details:
            payload[key] = details[key]
    try:
        update_job(job_id, status, progress, error=details.get("error"),
                   summary=details.get("persisted_result", details.get("summary")))
    except Exception:
        LOGGER.exception("job metadata update failed")
    try:
        from redis import Redis
        publisher = Redis.from_url(redis_url)
        publisher.publish("raven.events", json.dumps(payload))
        publisher.close()
    except Exception:
        LOGGER.exception("job event publish failed")


@celery_app.task(name="raven.process_video", bind=True)
def process_video_task(self, video_record):
    """Analyze an upload and store downloadable RGB/estimated-thermal results."""
    job_id = self.request.id
    _publish_job_status(job_id, "processing", 0, video_id=video_record["video_id"])
    try:
        from src.video.processing import process_uploaded_video

        current_progress = 0.0

        def progress(completed, total):
            nonlocal current_progress
            value = min(.99, completed / total) if total else 0
            current_progress = value
            self.update_state(state="PROGRESS", meta={"completed_frames": completed,
                              "total_frames": total, "progress": value})
            _publish_job_status(job_id, "processing", value, completed_frames=completed,
                                total_frames=total)

        def alert(event):
            _publish_job_status(job_id, "processing", current_progress,
                                event_type="alert", video_id=video_record["video_id"],
                                alert_event=event)

        result = process_uploaded_video(video_record, job_id, pipeline=_get_video_pipeline(),
                                         progress_callback=progress, event_callback=alert)
        summary = result["summary"]
        _publish_job_status(job_id, "completed", 1, summary=summary, persisted_result=result)
        return result
    except Exception as exc:
        _publish_job_status(job_id, "failed", 0, error=str(exc)[:1000])
        raise
