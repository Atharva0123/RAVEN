"""Streaming local or S3-compatible object storage for uploaded media."""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_VIDEO_BYTES = int(os.getenv("RAVEN_MAX_VIDEO_BYTES", str(2 * 1024 * 1024 * 1024)))
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
CONTENT_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime",
                 ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}


def _validate_video_signature(header: bytes, suffix: str) -> None:
    is_mp4 = len(header) >= 12 and header[4:8] == b"ftyp"
    is_avi = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    is_mkv = header.startswith(bytes.fromhex("1a45dfa3"))
    matches_extension = ((suffix in {".mp4", ".mov"} and is_mp4)
                         or (suffix == ".avi" and is_avi)
                         or (suffix == ".mkv" and is_mkv))
    if not matches_extension:
        raise HTTPException(status_code=415, detail="Unsupported or invalid video file signature")


def _persist_object(local_path: str, object_key: str, content_type: str) -> str:
    backend = os.getenv("RAVEN_STORAGE_BACKEND", "local").lower()
    if backend == "local":
        root = Path(os.getenv("RAVEN_STORAGE_LOCAL_ROOT", str(PROJECT_ROOT / "data" / "objects"))).resolve()
        target = (root / object_key).resolve()
        if os.path.commonpath((str(root), str(target))) != str(root):
            raise ValueError("invalid storage key")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(local_path, target)
        return object_key
    if backend == "s3":
        import boto3
        bucket = os.getenv("RAVEN_S3_BUCKET")
        if not bucket:
            raise RuntimeError("RAVEN_S3_BUCKET is required for S3 storage")
        client = boto3.client("s3", endpoint_url=os.getenv("RAVEN_S3_ENDPOINT_URL"))
        client.upload_file(local_path, bucket, object_key, ExtraArgs={"ContentType": content_type})
        return f"s3://{bucket}/{object_key}"
    raise RuntimeError("RAVEN_STORAGE_BACKEND must be 'local' or 's3'")


def store_generated_file(local_path: str, object_key: str,
                         content_type: str = "video/mp4") -> str:
    """Persist a generated artifact from the background video worker."""
    return _persist_object(local_path, object_key, content_type)


def materialize_object(object_key: str) -> tuple[str, bool]:
    """Resolve local object storage or stage S3 data into a temporary file."""
    if object_key.startswith("s3://"):
        import boto3
        _, _, bucket_and_key = object_key.partition("s3://")
        bucket, _, key = bucket_and_key.partition("/")
        fd, path = tempfile.mkstemp(prefix="raven-source-")
        os.close(fd)
        boto3.client("s3", endpoint_url=os.getenv("RAVEN_S3_ENDPOINT_URL")).download_file(bucket, key, path)
        return path, True
    # Shared container root is explicit in compose; local mode defaults to the
    # repository data directory for API and worker processes alike.
    root = Path(os.getenv("RAVEN_STORAGE_LOCAL_ROOT", str(PROJECT_ROOT / "data" / "objects"))).resolve()
    target = (root / object_key).resolve()
    if os.path.commonpath((str(root), str(target))) != str(root) or not target.is_file():
        raise FileNotFoundError("stored video object is unavailable")
    return str(target), False


async def store_video(upload: UploadFile) -> dict:
    original = Path(upload.filename or "video.mp4").name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, AVI, or MKV video")
    video_id = str(uuid.uuid4())
    object_key = f"raw/{video_id}{suffix}"
    digest = hashlib.sha256()
    size = 0
    fd, temp_name = tempfile.mkstemp(prefix="raven-upload-", suffix=suffix)
    os.close(fd)
    try:
        with open(temp_name, "wb") as target:
            header = await upload.read(32)
            _validate_video_signature(header, suffix)
            size = len(header)
            if size > MAX_VIDEO_BYTES:
                raise HTTPException(status_code=413, detail="Video exceeds RAVEN_MAX_VIDEO_BYTES")
            digest.update(header)
            target.write(header)
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds RAVEN_MAX_VIDEO_BYTES")
                digest.update(chunk)
                target.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty")
        # Use the extension-to-type mapping only after validating file magic.
        mime = CONTENT_TYPES[suffix]
        stored_key = await asyncio.to_thread(_persist_object, temp_name, object_key, mime)
        return {"video_id": video_id, "filename": original[:255], "object_key": stored_key,
                "size_bytes": size, "sha256": digest.hexdigest(), "content_type": mime}
    finally:
        Path(temp_name).unlink(missing_ok=True)

