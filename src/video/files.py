"""Ownership-safe cleanup for app-managed temporary files."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def remove_owned_temp_file(path, *, temp_root=None, retries=5, sleep=time.sleep) -> bool:
    """Remove a file only if it is below the system temp root.

    Non-temporary inputs (including bundled demo videos) are never deleted.
    Brief Windows decoder locks are retried, then left for later cleanup.
    """
    if not path:
        return False
    candidate = Path(path).resolve()
    root = Path(temp_root or tempfile.gettempdir()).resolve()
    try:
        if os.path.commonpath((str(candidate), str(root))) != str(root):
            return False
    except ValueError:
        return False
    for attempt in range(max(1, retries)):
        try:
            candidate.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt + 1 == retries:
                return False
            sleep(0.05 * (attempt + 1))
