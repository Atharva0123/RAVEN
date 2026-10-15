"""Run the Next.js console and local FastAPI service for development.

Use `python scripts/run_demo.py --legacy-streamlit` for the temporary research UI.
PostgreSQL, Redis, and S3-compatible services are started with Docker Compose.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_api_key() -> str:
    """Load the shared local API key from the environment or ignored .env file."""
    configured = os.getenv("RAVEN_API_KEY")
    if configured:
        return configured
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "RAVEN_API_KEY":
                configured = value.strip().strip("\"'")
                if configured and not configured.startswith("change-me"):
                    return configured
    return secrets.token_urlsafe(32)


def _project_runtime_settings() -> dict[str, str]:
    """Read local RAVEN settings from .env without replacing shell overrides."""
    settings = {}
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return settings
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key.startswith("RAVEN_") and key not in os.environ:
            settings[key] = value.strip().strip("\"'")
    return settings


def _legacy_streamlit() -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py",
                            "--server.address", "127.0.0.1", "--server.port", "8501"], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-streamlit", action="store_true",
                        help="run the old Streamlit research dashboard")
    parser.add_argument("--api-port", type=int, default=None,
                        help="bind FastAPI to a chosen local port (defaults to an available port)")
    parser.add_argument("--frontend-port", type=int, default=None,
                        help="bind Next.js to a chosen local port (defaults to an available port)")
    args = parser.parse_args()
    if args.legacy_streamlit:
        return _legacy_streamlit()

    frontend_dir = ROOT / "frontend"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm or not (frontend_dir / "node_modules").is_dir():
        print("Install Node.js dependencies first: cd frontend; npm ci", file=sys.stderr)
        return 2

    api_port = args.api_port or _free_port()
    frontend_port = args.frontend_port or _free_port()
    if not 1 <= api_port <= 65535 or not 1 <= frontend_port <= 65535:
        parser.error("ports must be between 1 and 65535")
    if api_port == frontend_port:
        parser.error("API and frontend ports must be different")
    if args.frontend_port is None:
        while frontend_port == api_port:
            frontend_port = _free_port()
    while frontend_port == api_port:
        frontend_port = _free_port()
    api_key = _local_api_key()
    env = os.environ.copy()
    env.update(_project_runtime_settings())
    env.update({
        "RAVEN_API_KEY": api_key,
        "RAVEN_STORAGE_BACKEND": os.getenv("RAVEN_STORAGE_BACKEND", "local"),
        "RAVEN_STORAGE_LOCAL_ROOT": os.getenv("RAVEN_STORAGE_LOCAL_ROOT", str(ROOT / "data" / "objects")),
        "RAVEN_CORS_ORIGINS": f"http://localhost:{frontend_port},http://127.0.0.1:{frontend_port}",
        "RAVEN_API_URL": f"http://127.0.0.1:{api_port}",
        "RAVEN_BROWSER_API_URL": f"http://127.0.0.1:{api_port}",
    })
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api:app",
                            "--host", "127.0.0.1", "--port", str(api_port)], cwd=ROOT, env=env)
    frontend = None
    try:
        ready_url = f"http://127.0.0.1:{api_port}/ready"
        for _ in range(100):
            if api.poll() is not None:
                raise RuntimeError("FastAPI exited before becoming ready")
            try:
                with urlopen(ready_url, timeout=1):
                    break
            except (OSError, URLError):
                time.sleep(.2)
        else:
            raise RuntimeError("FastAPI did not become ready within 20 seconds")
        frontend = subprocess.Popen([npm, "run", "dev", "--", "--port", str(frontend_port)],
                                    cwd=frontend_dir, env=env)
        print(f"RAVEN Next.js: http://localhost:{frontend_port}", flush=True)
        print(f"FastAPI docs:  http://127.0.0.1:{api_port}/docs", flush=True)
        print("Local mode uses disk storage; PostgreSQL, Redis jobs, and MinIO require Docker Compose.", flush=True)
        frontend.wait()
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if frontend is not None and frontend.poll() is None:
            frontend.terminate()
            frontend.wait(timeout=10)
        if api.poll() is None:
            api.terminate()
            api.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
