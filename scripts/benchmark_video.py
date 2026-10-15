"""Stream a local video through Accuracy/Balanced/Performance for timing."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.pipeline import RavenPipeline
from src.video.io import iter_video_frames, VideoReadError


def benchmark(video_path: Path, mode: str, pipeline: RavenPipeline):
    pipeline.set_processing_mode(mode)
    pipeline.reset_video_state()
    pipeline.stage_latency_ema.clear()
    stage_samples = {name: [] for name in
                     ("yolo_botsort", "pose", "reid", "fusion", "total_pipeline")}
    processed_frames = skipped_detections = detection_count = 0
    started = time.perf_counter()
    for item in iter_video_frames(video_path):
        rows = pipeline.process_rgb_frame(item.image, metadata={
            "frame_index": item.index, "video_timestamp_seconds": item.timestamp_seconds})
        processed_frames += 1
        skipped_detections += int(bool(getattr(pipeline, "last_frame_stats", {}).get("detection_skipped")))
        detection_count += len(rows)
        for name, value in pipeline.last_frame_stats.get("stage_latency_ms", {}).items():
            if name in stage_samples:
                stage_samples[name].append(float(value))
    elapsed = time.perf_counter() - started
    return {"mode": mode, "video": str(video_path), "frames_processed": processed_frames,
            "detector_skips": skipped_detections, "result_rows": detection_count,
            "elapsed_seconds": round(elapsed, 2),
            "pipeline_fps": round(processed_frames / max(elapsed, 1e-9), 2),
            "device": pipeline.device_name, "gpu_memory_mb": pipeline.gpu_memory_mb,
            "mean_stage_latency_ms": {name: round(statistics.mean(values), 2) if values else 0.0
                                       for name, values in stage_samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--mode", choices=("Accuracy", "Balanced", "Performance", "all"), default="all")
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    modes = ("Accuracy", "Balanced", "Performance") if args.mode == "all" else (args.mode,)
    try:
        pipeline = RavenPipeline(enable_thermal=False)
        reports = [benchmark(args.video, mode, pipeline) for mode in modes]
    except VideoReadError as exc:
        parser.error(str(exc))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
