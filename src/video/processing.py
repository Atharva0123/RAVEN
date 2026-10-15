"""Shared uploaded-video inference and artifact generation for local and Celery jobs."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from src.pipeline import RavenPipeline
from src.storage import materialize_object, store_generated_file
from src.video.export import export_annotated_video


def _transcode_browser_video(path: str) -> None:
    """Convert OpenCV's MP4V output into a broadly playable H.264 MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Browser-compatible video encoding requires FFmpeg. Install the project requirements "
                "or add ffmpeg to PATH."
            ) from exc

    source = Path(path)
    fd, converted = tempfile.mkstemp(prefix=f"{source.stem}-h264-", suffix=".mp4", dir=source.parent)
    os.close(fd)
    try:
        command = [ffmpeg, "-nostdin", "-y", "-i", str(source), "-an", "-c:v", "libx264",
                   "-preset", os.getenv("RAVEN_FFMPEG_PRESET", "ultrafast"),
                   "-crf", os.getenv("RAVEN_FFMPEG_CRF", "26"), "-pix_fmt", "yuv420p",
                   "-tag:v", "avc1", "-movflags", "+faststart", converted]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=int(os.getenv("RAVEN_FFMPEG_TIMEOUT_SECONDS", "1800")))
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "unknown ffmpeg error")[-2000:]
            raise RuntimeError(f"Could not encode a browser-compatible result video: {detail}")
        if not Path(converted).is_file() or Path(converted).stat().st_size == 0:
            raise RuntimeError("FFmpeg completed without producing a result video")
        os.replace(converted, source)
    finally:
        Path(converted).unlink(missing_ok=True)


def process_uploaded_video(record: dict, job_id: str, *, pipeline: RavenPipeline | None = None,
                           progress_callback: Callable[[int, int], None] | None = None,
                           event_callback: Callable[[dict], None] | None = None) -> dict:
    """Analyze one RGB upload and produce RGB annotated plus estimated-thermal feeds."""
    source_path, temporary_source = materialize_object(record["object_key"])
    rgb_fd, rgb_path = tempfile.mkstemp(prefix="raven-rgb-result-", suffix=".mp4")
    thermal_fd, thermal_path = tempfile.mkstemp(prefix="raven-thermal-result-", suffix=".mp4")
    os.close(rgb_fd)
    os.close(thermal_fd)
    mode = record.get("processing_mode") or os.getenv("RAVEN_VIDEO_PROCESSING_MODE", "Performance")
    input_kind = record.get("input_kind", "rgb")
    started = time.perf_counter()
    try:
        pipeline = pipeline or RavenPipeline(enable_thermal=False)
        if hasattr(pipeline, "configure_single_stream"):
            pipeline.configure_single_stream("thermal" if input_kind == "thermal-palette" else "rgb")
        result = export_annotated_video(
            source_path, rgb_path, pipeline, mode=mode,
            progress_callback=progress_callback,
            estimated_thermal_output_path=thermal_path,
            event_callback=event_callback,
            input_kind=input_kind,
            show_trajectories=bool(record.get("show_trajectories", True)),
        )
        _transcode_browser_video(rgb_path)
        _transcode_browser_video(thermal_path)
        rgb_key = f"annotated/{record['video_id']}/{job_id}-rgb.mp4"
        thermal_key = f"annotated/{record['video_id']}/{job_id}-estimated-thermal.mp4"
        rgb_object = store_generated_file(rgb_path, rgb_key)
        thermal_object = store_generated_file(thermal_path, thermal_key)
        elapsed = time.perf_counter() - started
        summary = {
            "processed_frames": result["frames_written"],
            "duration_seconds": round(result["duration_seconds"], 3),
            "processing_fps": round(result["frames_written"] / max(elapsed, .001), 3),
            "max_concurrent_tracks": result["max_concurrent_tracks"],
            "unique_tracks": result["unique_tracks"],
            "unique_tracker_ids": result.get("unique_tracker_ids", result["unique_tracks"]),
            "event_count": result["event_count"],
            "events": result["events"],
            "tracks": result["tracks"],
            "threat_detections": result["threat_detections"],
            "alert_policy": result["alert_policy"],
            "no_alert_explanation": result["no_alert_explanation"],
            "capabilities": result["capabilities"],
            "compute_device": result["compute_device"],
            "inference_seconds": result["inference_seconds"],
            "inference_fps": result["inference_fps"],
            "stage_profile_ms": result["stage_profile_ms"],
            "export_seconds": result["export_seconds"],
            "processing_errors": result["processing_errors"],
            "truncated": result["truncated"],
            "processing_error_samples": result.get("processing_error_samples", []),
            "width": result["width"], "height": result["height"], "fps": result["fps"],
            "processing_mode": mode,
            "available_models": result["available_models"],
            "thermal_note": ("Thermal-palette visualization only; video output does not report numeric temperatures."
                             if input_kind == "thermal-palette"
                             else "Estimated from RGB appearance; not calibrated temperature data."),
            "input_kind": input_kind,
            "show_trajectories": bool(record.get("show_trajectories", True)),
        }
        return {
            "job_id": job_id, "video_id": record["video_id"], "status": "completed",
            "summary": summary,
            "rgb_artifact_key": rgb_object,
            "thermal_artifact_key": thermal_object,
            "rgb_artifact_url": f"/api/v1/jobs/{job_id}/artifacts/rgb",
            "thermal_artifact_url": f"/api/v1/jobs/{job_id}/artifacts/estimated-thermal",
        }
    finally:
        Path(rgb_path).unlink(missing_ok=True)
        Path(thermal_path).unlink(missing_ok=True)
        if temporary_source:
            Path(source_path).unlink(missing_ok=True)
