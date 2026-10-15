# Script inventory

## Active scripts

- `run_demo.py` launches the local FastAPI and Next.js demo. It uses local disk storage by default; optional PostgreSQL, Redis and MinIO services are started with Docker Compose.
- `setup_demo_videos.py` downloads sample demonstration clips into ignored runtime storage.
- `train_and_evaluate_raven.py` is a separate app-side S-DEC training/evaluation prototype. Its outputs are not the corrected Colab paper benchmark.
- `train_thermal_person.py` and `verify_llvip.py` support local thermal model preparation and dataset checks.
- `benchmark_video.py` measures throughput for a selected local configuration.

The current LLVIP research workflow is the project-root [`RAVEN.ipynb`](../RAVEN.ipynb). The historical `archived/` scripts are not required for the app or research workflow and are excluded from the curated GitHub upload; they remain in the local project folder for recovery/reference.
