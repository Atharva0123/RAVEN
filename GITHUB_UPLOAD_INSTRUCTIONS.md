# GitHub publication notes

The project folder is prepared for a GitHub repository, but it has no earlier Git history or remote URL. A new local `main` branch is initialized. The managed workspace does not allow the agent to write Git's index, so the files are not staged or committed here.

## Included in the curated project

Source code, configuration, the current `RAVEN.ipynb`, protocol documentation, the small app-side checkpoint, and the clean-run CSV/summary files are eligible for upload.

The `.gitignore` keeps credentials, LLVIP images/annotations, local recordings and jobs, generated caches, demo videos, model weights, the manuscript working copies, rendered paper drafts, and historical code out of Git. The current manuscript is kept locally under `docs/manuscripts/` until its audit items are resolved; see `docs/PAPER_AUDIT.md`. LLVIP must be obtained from its official source and used under its terms. Large YOLO/CLIP weights download separately. Optional sample clips can be installed locally with `python scripts/setup_demo_videos.py`.

## Publish from a normal writable checkout

1. Sign in to GitHub and create a repository, or choose an existing repository. Decide whether it should be public or private.
2. In a normal local checkout where Git can write `.git`, review the files and publish:

```powershell
git add --all
git diff --cached --stat
git commit -m "Prepare RAVEN project and LLVIP notebook"
git remote add origin <repository-url>
git push -u origin main
```

Do not force-push over existing history. If using an existing repository with commits, clone it first and copy the prepared project files into that checkout before committing.

The project currently has no license file. Decide a license with the project authors before presenting the repository as reusable open-source software.
