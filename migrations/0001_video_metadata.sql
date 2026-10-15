CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), email text NOT NULL UNIQUE,
  password_hash text NOT NULL, role text NOT NULL DEFAULT 'operator', created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS cameras (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), name text NOT NULL,
  credential_secret_ref text, enabled boolean NOT NULL DEFAULT true,
  config jsonb NOT NULL DEFAULT '{}'::jsonb, health_status text NOT NULL DEFAULT 'unknown',
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS videos (
  id uuid PRIMARY KEY, original_filename text NOT NULL, object_key text NOT NULL,
  size_bytes bigint NOT NULL CHECK (size_bytes > 0), sha256 char(64) NOT NULL,
  content_type text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS processing_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(), video_id uuid REFERENCES videos(id),
  camera_id uuid REFERENCES cameras(id), status text NOT NULL CHECK (status IN ('queued','processing','completed','failed','cancelled')),
  progress double precision NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
  error text, config jsonb NOT NULL DEFAULT '{}'::jsonb, summary jsonb,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS tracks (
  id bigserial PRIMARY KEY, camera_id uuid REFERENCES cameras(id), job_id uuid REFERENCES processing_jobs(id),
  external_track_id text NOT NULL, first_seen timestamptz NOT NULL, last_seen timestamptz NOT NULL,
  state text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS detections (
  id bigserial PRIMARY KEY, camera_id uuid REFERENCES cameras(id), job_id uuid REFERENCES processing_jobs(id),
  track_id bigint REFERENCES tracks(id), observed_at timestamptz NOT NULL, class_name text NOT NULL,
  confidence real NOT NULL, bbox jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id bigserial PRIMARY KEY, camera_id uuid REFERENCES cameras(id), job_id uuid REFERENCES processing_jobs(id),
  track_id bigint REFERENCES tracks(id), event_type text NOT NULL, occurred_at timestamptz NOT NULL,
  details jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS anomaly_results (
  id bigserial PRIMARY KEY, camera_id uuid REFERENCES cameras(id), job_id uuid REFERENCES processing_jobs(id),
  observed_at timestamptz NOT NULL, score real, artifact_key text, details jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE TABLE IF NOT EXISTS system_metrics (
  id bigserial PRIMARY KEY, camera_id uuid REFERENCES cameras(id), job_id uuid REFERENCES processing_jobs(id),
  observed_at timestamptz NOT NULL DEFAULT now(), metrics jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
  id bigserial PRIMARY KEY, name text NOT NULL, version text NOT NULL, checksum char(64),
  config jsonb NOT NULL DEFAULT '{}'::jsonb, deployment_status text NOT NULL DEFAULT 'registered',
  UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS audit_logs (
  id bigserial PRIMARY KEY, actor_id uuid REFERENCES users(id), action text NOT NULL,
  occurred_at timestamptz NOT NULL DEFAULT now(), details jsonb NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON processing_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_videos_sha256 ON videos(sha256);
CREATE INDEX IF NOT EXISTS idx_events_camera_time ON events(camera_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events(event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks(camera_id);
CREATE INDEX IF NOT EXISTS idx_detections_track_time ON detections(track_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_camera_time ON system_metrics(camera_id, observed_at DESC);
