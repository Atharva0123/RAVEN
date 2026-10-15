import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.api as raven_api
from src.api import app


class TestVideoUploadAPI(unittest.TestCase):
    def test_upload_streams_to_local_object_storage_with_random_key(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "RAVEN_API_KEY": "unit-test-key",
            "RAVEN_STORAGE_BACKEND": "local",
            "RAVEN_STORAGE_LOCAL_ROOT": folder,
            "RAVEN_LOCAL_JOB_STATE_PATH": str(Path(folder) / "jobs.json"),
            "RAVEN_REDIS_URL": "",
            "RAVEN_DATABASE_URL": "",
        }):
            content = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00raven-test"
            with patch.object(raven_api._video_executor, "submit") as submit:
                with TestClient(app) as client:
                    response = client.post("/api/v1/videos", headers={"X-API-Key": "unit-test-key"},
                        files={"file": ("../../camera clip.mp4", content, "text/plain")})
            self.assertEqual(response.status_code, 201, response.text)
            record = response.json()
            self.assertEqual(record["status"], "queued")
            self.assertEqual(record["filename"], "camera clip.mp4")
            self.assertEqual(record["content_type"], "video/mp4")
            self.assertNotIn("object_key", record)
            self.assertTrue(record["job_url"].endswith(record["job_id"]))
            stored = list((Path(folder) / "raw").glob("*.mp4"))
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].read_bytes(), content)
            submit.assert_called_once()

    def test_upload_rejects_bad_signature_and_missing_key(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "RAVEN_API_KEY": "unit-test-key", "RAVEN_STORAGE_LOCAL_ROOT": folder,
            "RAVEN_STORAGE_BACKEND": "local", "RAVEN_REDIS_URL": "", "RAVEN_DATABASE_URL": "",
            "RAVEN_LOCAL_JOB_STATE_PATH": str(Path(folder) / "jobs.json"),
        }):
            with TestClient(app) as client:
                denied = client.post("/api/v1/videos", files={"file": ("bad.mp4", b"not a video", "video/mp4")})
                invalid = client.post("/api/v1/videos", headers={"X-API-Key": "unit-test-key"},
                    files={"file": ("bad.mp4", b"not a video signature", "video/mp4")})
            self.assertEqual(denied.status_code, 401)
            self.assertEqual(invalid.status_code, 415)

    def test_job_status_hides_object_keys_and_artifacts_support_ranges(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "RAVEN_API_KEY": "unit-test-key", "RAVEN_STORAGE_LOCAL_ROOT": folder,
            "RAVEN_STORAGE_BACKEND": "local", "RAVEN_REDIS_URL": "", "RAVEN_DATABASE_URL": "",
            "RAVEN_LOCAL_JOB_STATE_PATH": str(Path(folder) / "jobs.json"),
        }):
            relative = "annotated/video/job-rgb.mp4"
            artifact = Path(folder) / relative
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"0123456789")
            job_id = "range-test-job"
            raven_api._set_local_video_job(job_id, status="completed", progress=1,
                result={"job_id": job_id, "status": "completed", "summary": {"processed_frames": 3},
                        "rgb_artifact_key": relative, "rgb_artifact_url": f"/api/v1/jobs/{job_id}/artifacts/rgb"})
            try:
                with TestClient(app) as client:
                    status = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "unit-test-key"})
                    video = client.get(f"/api/v1/jobs/{job_id}/artifacts/rgb",
                                       headers={"X-API-Key": "unit-test-key", "Range": "bytes=2-5"})
                self.assertEqual(status.status_code, 200, status.text)
                self.assertEqual(status.json()["result"]["summary"]["processed_frames"], 3)
                self.assertNotIn("rgb_artifact_key", status.text)
                self.assertEqual(video.status_code, 206, video.text)
                self.assertEqual(video.content, b"2345")
                self.assertEqual(video.headers.get("content-range"), "bytes 2-5/10")
            finally:
                with raven_api._local_video_jobs_lock:
                    raven_api._local_video_jobs.pop(job_id, None)

    def test_s3_artifact_streams_only_the_requested_range(self):
        class FakeBody:
            def iter_chunks(self, chunk_size):
                yield b"2345"

            def close(self):
                pass

        class FakeS3:
            def head_object(self, **_kwargs):
                return {"ContentLength": 10}

            def get_object(self, **kwargs):
                self.requested_range = kwargs.get("Range")
                return {"Body": FakeBody()}

        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
            "RAVEN_API_KEY": "unit-test-key", "RAVEN_REDIS_URL": "", "RAVEN_DATABASE_URL": "",
            "RAVEN_LOCAL_JOB_STATE_PATH": str(Path(folder) / "jobs.json"),
        }):
            job_id = "s3-range-test"
            s3 = FakeS3()
            raven_api._set_local_video_job(job_id, status="completed", progress=1,
                result={"job_id": job_id, "status": "completed",
                        "rgb_artifact_key": "s3://raven-test/annotated/video/job-rgb.mp4"})
            try:
                with patch("boto3.client", return_value=s3):
                    with TestClient(app) as client:
                        response = client.get(f"/api/v1/jobs/{job_id}/artifacts/rgb",
                            headers={"X-API-Key": "unit-test-key", "Range": "bytes=2-5"})
                self.assertEqual(response.status_code, 206, response.text)
                self.assertEqual(response.content, b"2345")
                self.assertEqual(s3.requested_range, "bytes=2-5")
                self.assertEqual(response.headers.get("content-range"), "bytes 2-5/10")
            finally:
                with raven_api._local_video_jobs_lock:
                    raven_api._local_video_jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
