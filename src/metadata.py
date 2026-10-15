"""PostgreSQL metadata access; database persistence is optional in local dev."""
from __future__ import annotations

import os


def save_video_metadata(record: dict) -> bool:
    dsn = os.getenv("RAVEN_DATABASE_URL")
    if not dsn:
        return False
    import psycopg
    from psycopg.types.json import Jsonb
    with psycopg.connect(dsn) as connection:
        connection.execute(
            "INSERT INTO videos (id, original_filename, object_key, size_bytes, sha256, content_type, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (record["video_id"], record["filename"], record["object_key"], record["size_bytes"],
             record["sha256"], record["content_type"],
             Jsonb({"source": "upload", "processing_mode": record.get("processing_mode", "Performance")})),
        )
    return True


def save_job(job_id: str, video_id: str) -> None:
    dsn = os.getenv("RAVEN_DATABASE_URL")
    if not dsn:
        return
    import psycopg
    with psycopg.connect(dsn) as connection:
        connection.execute("INSERT INTO processing_jobs (id, video_id, status) VALUES (%s, %s, 'queued')",
                           (job_id, video_id))


def update_job(job_id: str, status: str, progress: float, *, error: str | None = None,
               summary: dict | None = None) -> None:
    dsn = os.getenv("RAVEN_DATABASE_URL")
    if not dsn:
        return
    import psycopg
    from psycopg.types.json import Jsonb
    with psycopg.connect(dsn) as connection:
        connection.execute("UPDATE processing_jobs SET status=%s, progress=%s, error=%s, summary=COALESCE(%s, summary), updated_at=now() WHERE id=%s",
                           (status, progress, error, Jsonb(summary) if summary is not None else None, job_id))


def get_job(job_id: str) -> dict | None:
    """Load durable job state and artifact references from PostgreSQL, when configured."""
    dsn = os.getenv("RAVEN_DATABASE_URL")
    if not dsn:
        return None
    import psycopg
    with psycopg.connect(dsn) as connection:
        row = connection.execute(
            "SELECT id, video_id, status, progress, error, summary FROM processing_jobs WHERE id=%s",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return {"job_id": str(row[0]), "video_id": str(row[1]) if row[1] else None,
            "status": row[2], "progress": float(row[3]), "error": row[4], "summary": row[5]}
