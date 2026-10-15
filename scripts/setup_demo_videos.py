"""Download simulation clips and record checksums in data/demo/manifest.json.

Run from raven/: python scripts/setup_demo_videos.py. The videos remain outside
Git. Checksums are computed from the downloaded bytes and checked on later runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = {
    "normal_activity": ("https://motchallenge.net/sequenceVideos/MOT17-02-raw.mp4", "People walking around a large square"),
    "event_activity": ("https://motchallenge.net/sequenceVideos/MOT17-04-raw.mp4", "Pedestrian street at night, elevated viewpoint"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = ROOT / "data" / "demo" / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    for name, (url, description) in CLIPS.items():
        target_dir = ROOT / "data" / "demo" / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "demo.mp4"
        existing_hash = sha256(target) if target.is_file() else None
        if existing_hash and manifest.get(name, {}).get("sha256") == existing_hash:
            print(f"Verified {target.relative_to(ROOT)}")
            continue
        fd, temporary_name = tempfile.mkstemp(prefix="raven-demo-", suffix=".mp4", dir=target_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RAVEN-demo-setup/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            checksum = sha256(temporary)
            temporary.replace(target)
            manifest[name] = {"file": target.relative_to(ROOT).as_posix(), "url": url,
                              "description": description, "sha256": checksum}
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(f"Installed {target.relative_to(ROOT)} (SHA-256 {checksum})")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
