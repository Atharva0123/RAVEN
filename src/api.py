"""Production-oriented FastAPI endpoints for RAVEN image inference."""
from __future__ import annotations

import asyncio
import copy
import hmac
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.inference_queue import AsyncInferenceQueue
from src.observability import InferenceAudit, configure_json_logging
from src.pipeline import RavenPipeline, PROJECT_ROOT

configure_json_logging()
LOGGER = logging.getLogger("raven.api")
MAX_IMAGE_BYTES = int(os.getenv("RAVEN_MAX_IMAGE_BYTES", 8 * 1024 * 1024))
MAX_REQUEST_BYTES = int(os.getenv("RAVEN_MAX_REQUEST_BYTES", 32 * 1024 * 1024))
MAX_VIDEO_REQUEST_BYTES = int(os.getenv("RAVEN_MAX_VIDEO_BYTES", str(2 * 1024 * 1024 * 1024))) + 2 * 1024 * 1024
RATE_LIMIT = os.getenv("RAVEN_RATE_LIMIT", "60/minute")


class AnalyzeRequest(BaseModel):
    rgb: str = Field(description="Base64 image or image data URL")
    thermal: str = Field(description="Base64 image or image data URL")
    zone_context: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("rgb", "thermal")
    def payload_is_bounded(cls, value):
        if not isinstance(value, str) or len(value) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 512:
            raise ValueError("encoded image exceeds the configured payload limit")
        return value


class BatchRequest(BaseModel):
    images: list[AnalyzeRequest]

    @validator("images")
    def batch_is_bounded(cls, value):
        if not 1 <= len(value) <= 32:
            raise ValueError("batch must contain between 1 and 32 image pairs")
        return value


class ThermalPaletteRequest(BaseModel):
    image: str = Field(description="Base64 palette thermal image or data URL")
    minimum_temperature: float
    maximum_temperature: float

    @validator("image")
    def palette_image_is_bounded(cls, value):
        if not isinstance(value, str) or len(value) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 512:
            raise ValueError("encoded image exceeds the configured payload limit")
        return value


_pipeline = None
_temporal_fusion_checkpoint = None
_audit = InferenceAudit()
_limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[RATE_LIMIT],
    storage_uri=os.getenv("RAVEN_RATE_LIMIT_STORAGE_URI", "memory://"),
)
_inference_queue: AsyncInferenceQueue | None = None
_video_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raven-video")
_local_video_jobs_lock = threading.RLock()
_local_job_persist_lock = threading.Lock()
_active_video_fingerprints: dict[tuple[str, str, str], str] = {}
_local_video_pipeline = None


def _local_job_state_path() -> Path:
    configured = Path(os.getenv("RAVEN_LOCAL_JOB_STATE_PATH", str(PROJECT_ROOT / "data" / "local_jobs.json")))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _load_local_video_jobs() -> dict[str, dict[str, Any]]:
    path = _local_job_state_path()
    pending_path = path.with_name(f"{path.stem}.pending{path.suffix}")
    source = path
    try:
        if pending_path.is_file() and (not path.is_file()
                                       or pending_path.stat().st_mtime >= path.stat().st_mtime):
            source = pending_path
    except OSError:
        pass
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
        jobs = loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
    changed = source != path
    for job in jobs.values():
        interrupted = (job.get("status") in {"queued", "processing"}
                       or (job.get("status") == "failed" and
                           str(job.get("error", "")).startswith("Local video processing stopped when the API restarted")))
        if not interrupted:
            continue
        if _recover_local_video_artifacts(job):
            changed = True
            continue
        job["status"] = "failed"
        job["error"] = "Local video processing stopped before output files were saved. Upload the video again."
        job["updated_at"] = time.time()
        changed = True
    if changed:
        _write_local_video_jobs(jobs)
    return jobs


def _recover_local_video_artifacts(job: dict[str, Any]) -> bool:
    """Restore completed status when output files survived a failed history-file replace."""
    job_id, video_id = job.get("job_id"), job.get("video_id")
    if not job_id or not video_id:
        return False
    artifact_dir = PROJECT_ROOT / "data" / "objects" / "annotated" / str(video_id)
    rgb_name = f"{job_id}-rgb.mp4"
    thermal_name = f"{job_id}-estimated-thermal.mp4"
    rgb_path, thermal_path = artifact_dir / rgb_name, artifact_dir / thermal_name
    if not all(path.is_file() and path.stat().st_size > 0 for path in (rgb_path, thermal_path)):
        return False
    # The output files prove export completed; retain only counters previously
    # persisted and mark that detailed per-track results could not be recovered.
    summary = {"recovered_from_saved_artifacts": True,
               "details_note": "Annotated outputs were recovered; per-track details were not persisted."}
    if job.get("completed_frames") is not None:
        summary["processed_frames"] = job["completed_frames"]
    job.update(status="completed", phase="complete", progress=1.0, error=None,
               summary=summary,
               result={"job_id": str(job_id), "video_id": str(video_id), "status": "completed",
                       "summary": summary,
                       "rgb_artifact_key": f"annotated/{video_id}/{rgb_name}",
                       "thermal_artifact_key": f"annotated/{video_id}/{thermal_name}",
                       "rgb_artifact_url": f"/api/v1/jobs/{job_id}/artifacts/rgb",
                       "thermal_artifact_url": f"/api/v1/jobs/{job_id}/artifacts/estimated-thermal"},
               updated_at=time.time())
    return True


def _write_local_video_jobs(jobs: dict[str, dict[str, Any]] | None = None) -> None:
    path = _local_job_state_path()
    temporary = None
    with _local_job_persist_lock:
        try:
            if jobs is None:
                with _local_video_jobs_lock:
                    jobs = {job_id: dict(job) for job_id, job in _local_video_jobs.items()}
            path.parent.mkdir(parents=True, exist_ok=True)
            persisted_jobs = {}
            for job_id, job in jobs.items():
                persisted_job = dict(job)
                summary = job.get("summary")
                if isinstance(summary, dict):
                    persisted_job["summary"] = _compact_summary_for_disk(summary)
                result = job.get("result")
                if isinstance(result, dict):
                    persisted_result = dict(result)
                    if isinstance(result.get("summary"), dict):
                        persisted_result["summary"] = _compact_summary_for_disk(result["summary"])
                    persisted_job["result"] = persisted_result
                persisted_jobs[job_id] = persisted_job
            fd, temporary = tempfile.mkstemp(prefix="raven-jobs-", suffix=".json", dir=path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(persisted_jobs, output, ensure_ascii=False)
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            try:
                os.replace(temporary, path)
            except PermissionError:
                # OneDrive/Windows may briefly deny replacing an indexed JSON file.
                # Preserve the newest complete state as a sidecar for the next start.
                pending_path = path.with_name(f"{path.stem}.pending{path.suffix}")
                try:
                    os.replace(temporary, pending_path)
                except OSError:
                    shutil.copyfile(temporary, pending_path)
                LOGGER.warning("job state target is locked; saved latest state to %s", pending_path)
                temporary = None
            else:
                path.with_name(f"{path.stem}.pending{path.suffix}").unlink(missing_ok=True)
        except (OSError, TypeError, ValueError):
            LOGGER.exception("could not persist local video job state")
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)


def _compact_summary_for_disk(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep useful track evidence while preventing per-frame history from ballooning."""
    compact = dict(summary)
    tracks = summary.get("tracks")
    if isinstance(tracks, list):
        compact_tracks = []
        for track in tracks:
            if not isinstance(track, dict):
                compact_tracks.append(track)
                continue
            compact_track = dict(track)
            for key, limit, count_key in (("identity_timeline", 40, "identity_observation_count"),
                                          ("trajectory", 20, "trajectory_observation_count")):
                values = track.get(key)
                if isinstance(values, list):
                    compact_track[count_key] = max(int(track.get(count_key, 0) or 0), len(values))
                    compact_track[key] = values[-limit:]
            evidence = track.get("evidence")
            if isinstance(evidence, list):
                compact_track["evidence_count"] = max(
                    int(track.get("evidence_count", 0) or 0), len(evidence))
                compact_track["evidence"] = evidence[-20:]
            compact_tracks.append(compact_track)
        compact["tracks"] = compact_tracks
    events = summary.get("events")
    if isinstance(events, list):
        compact["events_retained_count"] = len(events)
        compact["events"] = events[-100:]
    return compact


_local_video_jobs: dict[str, dict[str, Any]] = _load_local_video_jobs()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = RavenPipeline(
            temporal_fusion_model=copy.deepcopy(_temporal_fusion_checkpoint)
            if _temporal_fusion_checkpoint is not None else None)
    return _pipeline


def _process_payload(payload: dict[str, Any]):
    return get_pipeline().process_request(payload)


def _queue():
    global _inference_queue
    if _inference_queue is None:
        _inference_queue = AsyncInferenceQueue(_process_payload)
    return _inference_queue


async def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    expected = os.getenv("RAVEN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="API key authentication is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _temporal_fusion_checkpoint
    temporal_path = Path(os.getenv("RAVEN_TEMPORAL_FUSION_WEIGHTS",
                                  str(PROJECT_ROOT / "data" / "models" / "best_model.pth")))
    if not temporal_path.is_absolute():
        temporal_path = PROJECT_ROOT / temporal_path
    if temporal_path.is_file():
        try:
            from src.fusion.temporal_cross_modal import TemporalCrossModalFusion
            _temporal_fusion_checkpoint = TemporalCrossModalFusion.load(temporal_path)
            LOGGER.info("Temporal fusion checkpoint loaded", extra={
                "synthetic_demo": bool(_temporal_fusion_checkpoint.training_metadata.get("synthetic", False))})
        except Exception:
            LOGGER.exception("Temporal fusion checkpoint could not be loaded")
    await _queue().start()
    yield
    if _inference_queue is not None:
        await _inference_queue.close()


app = FastAPI(
    title="RAVEN Research Prototype API",
    version="2.0.0",
    description=("Human-in-the-loop video review API for the RAVEN multi-modal quality-aware sensor fusion and temporal event-ranking research prototype. "
        "Scores and object detections are review cues, not confirmed threats; synthetic temporal weights are disabled for scoring."),
    lifespan=lifespan,
)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
app.add_middleware(CORSMiddleware,
                   allow_origins=[item.strip() for item in os.getenv("RAVEN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if item.strip()],
                   allow_credentials=False, allow_methods=["GET", "HEAD", "POST", "OPTIONS"],
                   allow_headers=["Content-Type", "Range", "X-API-Key", "Idempotency-Key"],
                   expose_headers=["Accept-Ranges", "Content-Range", "Content-Length", "X-Request-ID"])


@app.post("/api/v1/videos", status_code=201, tags=["videos"])
@_limiter.limit("10/minute")
async def upload_video(request: Request, file: UploadFile = File(...),
                       processing_mode: str = Form(default="Performance"),
                       input_kind: str = Form(default="rgb"),
                       show_trajectories: bool = Form(default=True),
                       idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
                       _api_key: None = Depends(require_api_key)):
    """Store an upload and immediately queue analysis, even in local-only mode."""
    from src.metadata import save_job, save_video_metadata
    from src.storage import store_video
    if processing_mode not in {"Accuracy", "Balanced", "Performance"}:
        raise HTTPException(status_code=422, detail="processing_mode must be Accuracy, Balanced, or Performance")
    if input_kind not in {"rgb", "thermal-palette"}:
        raise HTTPException(status_code=422, detail="input_kind must be rgb or thermal-palette")
    if not os.getenv("RAVEN_REDIS_URL"):
        with _local_video_jobs_lock:
            active = sum(job["status"] in {"queued", "processing"}
                         for job in _local_video_jobs.values())
        if active >= 3:
            raise HTTPException(status_code=429, detail="Video analysis queue is full; wait for the current job")
    try:
        record = await store_video(file)
        record["processing_mode"] = processing_mode
        record["input_kind"] = input_kind
        record["show_trajectories"] = show_trajectories
        fingerprint = (record["sha256"], input_kind, processing_mode, show_trajectories)
        with _local_video_jobs_lock:
            previous_id = _active_video_fingerprints.get(fingerprint)
            previous_job = _local_video_jobs.get(previous_id or "", {})
            if previous_id and (os.getenv("RAVEN_REDIS_URL") or previous_job.get("status") in {"queued", "processing"}):
                raise HTTPException(status_code=409, detail=f"This video is already queued or processing (job {previous_id[:8]})")
            if previous_id:
                _active_video_fingerprints.pop(fingerprint, None)
            active = sum(job["status"] in {"queued", "processing"}
                         for job in _local_video_jobs.values())
            if not os.getenv("RAVEN_REDIS_URL") and active >= 3:
                raise HTTPException(status_code=429, detail="Video analysis queue is full; wait for the current job")
            job_id = str(uuid.uuid4())
            _active_video_fingerprints[fingerprint] = job_id
        record["active_fingerprint"] = fingerprint
        await asyncio.to_thread(save_video_metadata, record)
        await asyncio.to_thread(save_job, job_id, record["video_id"])
        response = {key: record[key] for key in
                    ("video_id", "filename", "size_bytes", "sha256", "content_type")}
        response.update(status="queued", job_id=job_id, job_url=f"/api/v1/jobs/{job_id}")
        try:
            if os.getenv("RAVEN_REDIS_URL"):
                from src.tasks import process_video_task
                process_video_task.apply_async(args=[record], task_id=job_id)
            else:
                _set_local_video_job(job_id, status="queued", progress=0.0, video_id=record["video_id"],
                                     filename=record["filename"], created_at=time.time())
                _video_executor.submit(_run_local_video_job, job_id, record)
        except Exception:
            with _local_video_jobs_lock:
                if _active_video_fingerprints.get(fingerprint) == job_id:
                    _active_video_fingerprints.pop(fingerprint, None)
            raise
        return response
    except HTTPException:
        raise
    except Exception:
        if "fingerprint" in locals() and "job_id" in locals():
            with _local_video_jobs_lock:
                if _active_video_fingerprints.get(fingerprint) == job_id:
                    _active_video_fingerprints.pop(fingerprint, None)
        raise
    finally:
        await file.close()


@app.get("/api/v1/jobs/{job_id}", tags=["jobs"])
@_limiter.limit(RATE_LIMIT)
async def video_job_status(request: Request, job_id: str,
                           _api_key: None = Depends(require_api_key)):
    with _local_video_jobs_lock:
        local = _local_video_jobs.get(job_id)
        if local:
            return _public_local_job(local)
    if not os.getenv("RAVEN_REDIS_URL"):
        raise HTTPException(status_code=404, detail="Video job was not found in this local API session")
    from src.tasks import celery_app
    task = celery_app.AsyncResult(job_id)
    payload = {"job_id": job_id, "status": task.status.lower()}
    if task.info and isinstance(task.info, dict) and task.status == "PROGRESS":
        payload.update(task.info)
    if task.successful():
        payload.update(_public_result_payload(task.result))
    elif task.failed():
        payload["error"] = str(task.result)
    from src.metadata import get_job
    stored = await asyncio.to_thread(get_job, job_id)
    if stored:
        payload.update(status=stored["status"], progress=stored["progress"])
        if stored.get("error"):
            payload["error"] = stored["error"]
        if isinstance(stored.get("summary"), dict):
            payload.update(_public_result_payload(stored["summary"]))
    return payload


@app.get("/api/v1/jobs/{job_id}/artifacts/{kind}", tags=["jobs"])
@_limiter.limit(RATE_LIMIT)
async def video_job_artifact(request: Request, job_id: str, kind: str,
                             _api_key: None = Depends(require_api_key)):
    if kind not in {"rgb", "estimated-thermal"}:
        raise HTTPException(status_code=404, detail="Unknown video artifact")
    key_name = "rgb_artifact_key" if kind == "rgb" else "thermal_artifact_key"
    with _local_video_jobs_lock:
        local = _local_video_jobs.get(job_id)
    job_result = (local or {}).get("result")
    if job_result is None and os.getenv("RAVEN_REDIS_URL"):
        from src.tasks import celery_app
        task = celery_app.AsyncResult(job_id)
        if task.successful() and isinstance(task.result, dict):
            job_result = task.result
        if not job_result:
            from src.metadata import get_job
            stored = await asyncio.to_thread(get_job, job_id)
            if stored and isinstance(stored.get("summary"), dict):
                job_result = stored["summary"]
    if not job_result or not job_result.get(key_name):
        raise HTTPException(status_code=404, detail="The requested result is not ready")
    object_key = job_result[key_name]
    if object_key.startswith("s3://"):
        return await _s3_video_response(request, object_key, kind)
    from src.storage import materialize_object
    try:
        path, temporary = await asyncio.to_thread(materialize_object, object_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Stored video result is unavailable") from exc
    cleanup = BackgroundTask(Path(path).unlink, missing_ok=True) if temporary else None
    filename = f"raven-{kind}.mp4"
    return FileResponse(path, media_type="video/mp4", background=cleanup,
                        headers={"Content-Disposition": f'inline; filename="{filename}"',
                                 "Accept-Ranges": "bytes", "Cache-Control": "private, no-store"})


async def _s3_video_response(request: Request, object_url: str, kind: str):
    import boto3

    bucket_and_key = object_url.removeprefix("s3://")
    bucket, separator, key = bucket_and_key.partition("/")
    if not separator or not bucket or not key:
        raise HTTPException(status_code=404, detail="Stored video result is unavailable")
    client = boto3.client("s3", endpoint_url=os.getenv("RAVEN_S3_ENDPOINT_URL"))
    try:
        info = await asyncio.to_thread(client.head_object, Bucket=bucket, Key=key)
    except Exception as exc:
        LOGGER.warning("video artifact lookup failed", exc_info=True)
        raise HTTPException(status_code=404, detail="Stored video result is unavailable") from exc
    size = int(info["ContentLength"])
    start, end, status_code = 0, size - 1, 200
    range_header = request.headers.get("range")
    if range_header:
        try:
            unit, spec = range_header.split("=", 1)
            if unit != "bytes" or "," in spec:
                raise ValueError
            first, last = spec.split("-", 1)
            if not first:
                suffix_length = int(last)
                if suffix_length < 1:
                    raise ValueError
                start = max(0, size - suffix_length)
            else:
                start = int(first)
                end = min(size - 1, int(last)) if last else size - 1
            if start < 0 or start >= size or end < start:
                raise ValueError
            status_code = 206
        except ValueError:
            return Response(status_code=416, headers={"Accept-Ranges": "bytes",
                           "Content-Range": f"bytes */{size}"})
    length = end - start + 1
    try:
        kwargs = {"Bucket": bucket, "Key": key}
        if status_code == 206:
            kwargs["Range"] = f"bytes={start}-{end}"
        response = await asyncio.to_thread(client.get_object, **kwargs)
    except Exception as exc:
        LOGGER.exception("video artifact read failed")
        raise HTTPException(status_code=502, detail="Stored video result could not be streamed") from exc
    body = response["Body"]

    async def stream():
        try:
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                if chunk:
                    yield chunk
        finally:
            body.close()

    headers = {"Accept-Ranges": "bytes", "Content-Length": str(length),
               "Content-Disposition": f'inline; filename="raven-{kind}.mp4"',
               "Cache-Control": "private, no-store"}
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(stream(), status_code=status_code, media_type="video/mp4", headers=headers)


def _public_result_payload(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    summary = result.get("summary")
    public = {"summary": summary} if isinstance(summary, dict) else {}
    job_id = result.get("job_id")
    if job_id:
        public["result"] = {
            "job_id": job_id,
            "video_id": result.get("video_id"),
            "status": result.get("status", "completed"),
            "summary": summary,
            "rgb_artifact_url": result.get("rgb_artifact_url"),
            "thermal_artifact_url": result.get("thermal_artifact_url"),
        }
    return public


def _public_local_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = {key: job[key] for key in (
        "job_id", "status", "phase", "progress", "video_id", "filename", "created_at", "updated_at",
        "completed_frames", "total_frames", "summary", "error", "latest_alert") if key in job}
    payload.update(_public_result_payload(job.get("result")))
    return payload


def _set_local_video_job(job_id: str, **updates):
    should_persist = False
    with _local_video_jobs_lock:
        job = _local_video_jobs.setdefault(job_id, {"job_id": job_id})
        previous_status = job.get("status")
        job.update(updates)
        job["updated_at"] = time.time()
        # Frame progress is served directly from this in-memory state. Persisting
        # every callback rewrote the full (often multi-megabyte) history file
        # every 25 frames and made long videos appear frozen on local storage.
        status = job.get("status")
        fingerprint = job.get("active_fingerprint")
        if status in {"completed", "failed"} and fingerprint:
            if _active_video_fingerprints.get(tuple(fingerprint)) == job_id:
                _active_video_fingerprints.pop(tuple(fingerprint), None)
        if status in {"completed", "failed"} or status != previous_status:
            finished = [item for item in _local_video_jobs.values()
                        if item.get("status") in {"completed", "failed"}]
            if len(finished) > 200:
                for oldest in sorted(finished, key=lambda item: item.get("updated_at", 0))[:-200]:
                    _local_video_jobs.pop(oldest["job_id"], None)
            should_persist = True
        result = dict(job)
    if should_persist:
        # Disk/OneDrive I/O is slow; keep API status reads independent of the writer.
        _write_local_video_jobs()
    return result


def _run_local_video_job(job_id: str, record: dict):
    global _local_video_pipeline
    _set_local_video_job(job_id, status="processing", phase="loading models", progress=0.0)
    try:
        from src.metadata import update_job
        from src.video.processing import process_uploaded_video
        if _local_video_pipeline is None:
            _local_video_pipeline = RavenPipeline(
                enable_thermal=False,
                temporal_fusion_model=copy.deepcopy(_temporal_fusion_checkpoint)
                if _temporal_fusion_checkpoint is not None else None)

        def progress(completed, total):
            value = min(.99, completed / total) if total else 0.0
            phase = "finalizing result feeds" if total and completed >= total else "analyzing frames"
            _set_local_video_job(job_id, status="processing", phase=phase, progress=value,
                                 completed_frames=completed, total_frames=total)

        def alert(event):
            with _local_video_jobs_lock:
                current = dict(_local_video_jobs.get(job_id, {}))
            _set_local_video_job(job_id, status="processing",
                                 progress=current.get("progress", 0.0),
                                 completed_frames=current.get("completed_frames", 0),
                                 total_frames=current.get("total_frames", 0), latest_alert=event)

        result = process_uploaded_video(record, job_id, pipeline=_local_video_pipeline,
                                        progress_callback=progress, event_callback=alert)
        _set_local_video_job(job_id, status="completed", phase="complete", progress=1.0, summary=result["summary"],
                             result=result, completed_frames=result["summary"]["processed_frames"],
                             total_frames=result["summary"]["processed_frames"])
        try:
            update_job(job_id, "completed", 1.0, summary=result)
        except Exception:
            LOGGER.exception("local video job metadata update failed")
    except Exception as exc:
        LOGGER.exception("local video job failed")
        error = str(exc)[:1000]
        _set_local_video_job(job_id, status="failed", phase="failed", progress=0.0, error=error)
        try:
            from src.metadata import update_job
            update_job(job_id, "failed", 0.0, error=error)
        except Exception:
            LOGGER.exception("local failed-job metadata update failed")


@app.websocket("/api/v1/ws/status")
async def websocket_status(websocket: WebSocket, job_id: str | None = None):
    """Stream lightweight status and event messages from Redis Pub/Sub."""
    allowed_origins = {item.strip() for item in os.getenv(
        "RAVEN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3002,http://127.0.0.1:3002").split(",") if item.strip()}
    origin = websocket.headers.get("origin")
    if origin and origin not in allowed_origins:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    redis_url = os.getenv("RAVEN_REDIS_URL")
    client = pubsub = None
    try:
        await websocket.send_json({"type": "status", "status": "connected", "service": "raven-api"})
        if redis_url:
            from redis.asyncio import Redis
            client = Redis.from_url(redis_url, decode_responses=True)
            pubsub = client.pubsub()
            await pubsub.subscribe("raven.events")
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("data"):
                    try:
                        event = json.loads(message["data"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if job_id and event.get("job_id") == job_id:
                        allowed = {key: event[key] for key in (
                            "type", "job_id", "status", "progress", "completed_frames", "total_frames",
                            "video_id", "summary", "error", "alert_event") if key in event}
                        await websocket.send_json(allowed)
                else:
                    await asyncio.sleep(0.1)
        else:
            last_versions = {}
            sent_alert_ids = set()
            while True:
                with _local_video_jobs_lock:
                    jobs = [dict(_public_local_job(job)) for job in _local_video_jobs.values()
                            if job_id and job.get("job_id") == job_id]
                for job in jobs:
                    version = job.get("updated_at")
                    if version != last_versions.get(job["job_id"]):
                        last_versions[job["job_id"]] = version
                        await websocket.send_json({key: value for key, value in
                                                   {"type": "job", **job}.items() if key != "result"})
                        alert = job.get("latest_alert")
                        if alert:
                            event_id = alert.get("event_id") or json.dumps(alert, sort_keys=True)
                            if event_id not in sent_alert_ids:
                                sent_alert_ids.add(event_id)
                                await websocket.send_json({"type": "alert", "job_id": job["job_id"],
                                                           "alert_event": alert})
                await asyncio.sleep(.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        LOGGER.exception("websocket status stream failed")
    finally:
        if pubsub is not None:
            await pubsub.aclose()
        if client is not None:
            await client.aclose()


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
    length = request.headers.get("content-length")
    request_limit = MAX_VIDEO_REQUEST_BYTES if request.url.path == "/api/v1/videos" else MAX_REQUEST_BYTES
    if length:
        try:
            if int(length) > request_limit:
                return JSONResponse(status_code=413, content={"detail": "request body exceeds the configured size limit"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid Content-Length header"})
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        LOGGER.info("http_request", extra={
            "request_id": request_id,
            "path": request.url.path,
            "status_code": getattr(locals().get("response"), "status_code", 500),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "client_ip": request.client.host if request.client else "unknown",
        })


def _temporal_fusion_health_fields():
    metadata = (getattr(_temporal_fusion_checkpoint, "training_metadata", {}) or {})
    loaded = _temporal_fusion_checkpoint is not None
    synthetic = bool(metadata.get("synthetic", False))
    eligible = bool(loaded and metadata and not synthetic)
    status = ("not_loaded" if not loaded else
              "synthetic_architecture_smoke_test_only" if synthetic else
              "checkpoint_provenance_unknown_scoring_disabled" if not metadata else
              "experimental_weights_not_independently_validated")
    return {"temporal_fusion_loaded": loaded,
            "synthetic_demo_checkpoint": bool(loaded and synthetic),
            "temporal_fusion_eligible_for_scoring": eligible,
            "temporal_fusion_status": status}


@app.get("/health")
def health():
    return {"status": "ready", "service": "raven",
            "models_loaded": (_pipeline is not None or _local_video_pipeline is not None
                              or _temporal_fusion_checkpoint is not None),
            **_temporal_fusion_health_fields(),
            "queue_depth": _inference_queue.queue.qsize() if _inference_queue and _inference_queue.queue else 0}


@app.get("/ready")
async def readiness():
    dependencies = {"api": "ready"}
    dsn = os.getenv("RAVEN_DATABASE_URL")
    if dsn:
        try:
            import psycopg
            await asyncio.to_thread(lambda: psycopg.connect(dsn).close())
            dependencies["postgres"] = "ready"
        except Exception:
            dependencies["postgres"] = "unavailable"
    redis_url = os.getenv("RAVEN_REDIS_URL")
    if redis_url:
        try:
            from redis.asyncio import Redis
            client = Redis.from_url(redis_url)
            await client.ping()
            await client.aclose()
            dependencies["redis"] = "ready"
        except Exception:
            dependencies["redis"] = "unavailable"
        try:
            from src.tasks import celery_app
            workers = await asyncio.to_thread(celery_app.control.ping, timeout=2.0)
            dependencies["worker"] = "ready" if workers else "unavailable"
        except Exception:
            dependencies["worker"] = "unavailable"
    if os.getenv("RAVEN_STORAGE_BACKEND", "local").lower() == "s3":
        try:
            import boto3
            bucket = os.getenv("RAVEN_S3_BUCKET")
            if not bucket:
                raise RuntimeError("RAVEN_S3_BUCKET is not set")
            client = boto3.client("s3", endpoint_url=os.getenv("RAVEN_S3_ENDPOINT_URL"))
            await asyncio.to_thread(client.head_bucket, Bucket=bucket)
            dependencies["object_storage"] = "ready"
        except Exception:
            dependencies["object_storage"] = "unavailable"
    ready = all(value == "ready" for value in dependencies.values())
    with _local_video_jobs_lock:
        queue_depth = sum(job.get("status") in {"queued", "processing"}
                          for job in _local_video_jobs.values())
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ready" if ready else "degraded", "service": "raven",
                                 "models_loaded": (_pipeline is not None or _local_video_pipeline is not None
                                                   or _temporal_fusion_checkpoint is not None),
                                 **_temporal_fusion_health_fields(),
                                 "queue_depth": queue_depth,
                                 "dependencies": dependencies})


@app.post("/predict")
@app.post("/analyze", include_in_schema=False)
@_limiter.limit(RATE_LIMIT)
async def predict(request: Request, payload: AnalyzeRequest, background_tasks: BackgroundTasks,
                  _api_key: None = Depends(require_api_key)):
    try:
        result = await _queue().submit(payload.model_dump() if hasattr(payload, "model_dump") else payload.dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Frame analysis failed") from exc
    if _audit.backend != "disabled":
        background_tasks.add_task(_record_audit, result)
    return result


@app.post("/predict/batch")
@_limiter.limit("10/minute")
async def predict_batch(request: Request, payload: BatchRequest, background_tasks: BackgroundTasks,
                        _api_key: None = Depends(require_api_key)):
    jobs = [item.model_dump() if hasattr(item, "model_dump") else item.dict() for item in payload.images]
    try:
        results = await asyncio.gather(*[_queue().submit(item) for item in jobs])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch analysis failed") from exc
    if _audit.backend != "disabled":
        for result in results:
            background_tasks.add_task(_record_audit, result)
    return {"count": len(results), "results": results}


@app.post("/api/v1/thermal/palette/analyze")
@app.post("/api/v1/thermal/palette/estimate", include_in_schema=False)
@_limiter.limit("20/minute")
async def analyze_thermal_palette(request: Request, payload: ThermalPaletteRequest,
                                 _api_key: None = Depends(require_api_key)):
    """Estimate palette temperatures for a single image with thermaPy."""
    if not math.isfinite(payload.minimum_temperature) or not math.isfinite(payload.maximum_temperature):
        raise HTTPException(status_code=422, detail="temperature bounds must be finite")
    if payload.maximum_temperature <= payload.minimum_temperature:
        raise HTTPException(status_code=422, detail="maximum_temperature must exceed minimum_temperature")
    try:
        frame = RavenPipeline._decode_image(payload.image, "image")
        import tempfile
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as stream:
                temp_path = Path(stream.name)
            if not cv2.imwrite(str(temp_path), frame):
                raise RuntimeError("could not stage thermal palette image")
            from src.thermal.processing import estimate_palette_image
            temperature_map = await asyncio.to_thread(
                estimate_palette_image, str(temp_path), payload.minimum_temperature, payload.maximum_temperature)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("thermal palette analysis failed")
        raise HTTPException(status_code=422, detail=f"Thermal palette analysis failed: {exc}") from exc
    if temperature_map.shape[:2] != frame.shape[:2]:
        raise HTTPException(status_code=502, detail="thermaPy returned unexpected temperature-map dimensions")
    return {"height": int(temperature_map.shape[0]), "width": int(temperature_map.shape[1]),
            "minimum_temperature": float(payload.minimum_temperature),
            "maximum_temperature": float(payload.maximum_temperature),
            "estimated_min": float(np.nanmin(temperature_map)),
            "estimated_max": float(np.nanmax(temperature_map)),
            "temperature_map": temperature_map.astype(float).tolist(),
            "calibrated": False,
            "method": "thermaPy palette estimate; accuracy depends on source palette and supplied range"}


def _record_audit(result):
    try:
        _audit.record(result)
    except Exception:
        LOGGER.exception("inference metadata audit write failed")


@app.post("/jobs", status_code=202)
@_limiter.limit("5/minute")
async def submit_background_batch(request: Request, payload: BatchRequest,
                                  _api_key: None = Depends(require_api_key)):
    redis_url = os.getenv("RAVEN_REDIS_URL")
    if not redis_url:
        raise HTTPException(status_code=503, detail="Background jobs are disabled; configure RAVEN_REDIS_URL")
    from src.tasks import analyze_batch_task
    task = analyze_batch_task.apply_async(args=[[item.model_dump() if hasattr(item, "model_dump") else item.dict()
                                                 for item in payload.images]])
    return {"job_id": task.id, "status": "queued"}


@app.get("/jobs/{job_id}")
@_limiter.limit(RATE_LIMIT)
async def job_status(request: Request, job_id: str, _api_key: None = Depends(require_api_key)):
    if not os.getenv("RAVEN_REDIS_URL"):
        raise HTTPException(status_code=503, detail="Background jobs are disabled; configure RAVEN_REDIS_URL")
    from src.tasks import celery_app
    task = celery_app.AsyncResult(job_id)
    response = {"job_id": job_id, "status": task.status.lower()}
    if task.successful():
        response["result"] = task.result
    elif task.failed():
        response["error"] = str(task.result)
    return response


@app.get("/alerts")
@_limiter.limit(RATE_LIMIT)
async def alerts(request: Request, limit: int = 100, _api_key: None = Depends(require_api_key)):
    if not 1 <= limit <= 1000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
    path = PROJECT_ROOT / "data" / "alerts.jsonl"
    if not path.exists():
        return {"count": 0, "alerts": []}
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Alert log could not be read") from exc
    rows = rows[-limit:]
    return {"count": len(rows), "alerts": rows}
