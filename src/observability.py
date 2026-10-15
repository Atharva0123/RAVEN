"""JSON logging and lightweight inference metadata persistence."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = "".join(traceback.format_exception(*record.exc_info))
        for key in ("request_id", "path", "status_code", "duration_ms", "client_ip"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_json_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("RAVEN_LOG_LEVEL", "INFO").upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers[:] = [handler]
        logger.propagate = False


class InferenceAudit:
    """Persist detection metadata only; image payloads are never stored."""

    def __init__(self):
        self.backend = os.getenv("RAVEN_AUDIT_BACKEND", "disabled").strip().lower()
        self.sqlite_path = Path(os.getenv("RAVEN_AUDIT_SQLITE_PATH", "data/inference_audit.sqlite3"))
        self.s3_bucket = os.getenv("RAVEN_AUDIT_S3_BUCKET")
        self.s3_prefix = os.getenv("RAVEN_AUDIT_S3_PREFIX", "raven/inference")
        if self.backend not in {"sqlite", "s3", "disabled"}:
            raise ValueError("RAVEN_AUDIT_BACKEND must be sqlite, s3, or disabled")

    @staticmethod
    def _metadata(result):
        rows = result.get("results", []) if isinstance(result, dict) else []
        detections = []
        for row in rows:
            for obj in row.get("threat_objects", []):
                detections.append({"class": obj.get("label"), "confidence": obj.get("confidence")})
            if row.get("track_id") is not None:
                detections.append({"class": "person", "confidence": row.get("rgb_confidence")})
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result_count": len(rows),
            "classes": sorted({str(item["class"]) for item in detections if item.get("class")}),
            "detections": detections,
        }

    def record(self, result):
        if self.backend == "disabled":
            return
        metadata = self._metadata(result)
        if self.backend == "sqlite":
            path = self.sqlite_path if self.sqlite_path.is_absolute() else Path(__file__).resolve().parents[1] / self.sqlite_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path, timeout=10) as connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS inference_metadata (
                    id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, result_count INTEGER NOT NULL,
                    classes_json TEXT NOT NULL, detections_json TEXT NOT NULL)""")
                connection.execute("INSERT INTO inference_metadata VALUES (?, ?, ?, ?, ?)", (
                    str(uuid4()), metadata["timestamp"], metadata["result_count"],
                    json.dumps(metadata["classes"]), json.dumps(metadata["detections"])))
            return
        if not self.s3_bucket:
            raise ValueError("RAVEN_AUDIT_S3_BUCKET is required when S3 audit logging is enabled")
        import boto3
        key = f"{self.s3_prefix.rstrip('/')}/{metadata['timestamp'][:10]}/{uuid4()}.json"
        boto3.client("s3", endpoint_url=os.getenv("RAVEN_S3_ENDPOINT_URL") or None).put_object(
            Bucket=self.s3_bucket, Key=key,
            Body=json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json")
