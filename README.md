# RAVEN

RAVEN is a research prototype for arbitration between visible-light and thermal pedestrian detections. The project has two separate run paths: a local application demo and an LLVIP research workflow. The research workflow is the editable [RAVEN.ipynb](RAVEN.ipynb). See the [reproducibility guide](docs/REPRODUCIBILITY.md) for dataset handling, metric definition, run metadata, artifact gaps, and pre-submission checks, and the [architecture diagram](docs/figures/raven_architecture.md) for the implemented research path. The current manuscript working copy is stored locally under `docs/manuscripts/` and is excluded from GitHub until its audit items are resolved; see the [paper audit checklist](docs/PAPER_AUDIT.md). The project uses LLVIP only; it does not establish safety validation, calibrated risk probabilities, or performance on another dataset.

## Run the local application

The full local web demo runs a FastAPI service and the Next.js interface. It uses local disk storage by default; PostgreSQL, Redis and MinIO are optional for a larger deployment.

Requirements: Python, Node.js with npm, and (for optional services) Docker Desktop.

```powershell
# From the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Push-Location frontend
npm ci
Pop-Location
python scripts/run_demo.py
```

Open the local URL printed by the launcher. Press `Ctrl+C` to stop it. The launcher creates a temporary local API key if `RAVEN_API_KEY` is not already configured. For the older Streamlit dashboard, run `python scripts/run_demo.py --legacy-streamlit` instead.

For the Docker Compose deployment, copy `.env.example` to `.env`, replace every example password/key with private values, then run:

```powershell
docker compose up --build
```

Never commit `.env`. The checked-in `.env.example` is only a template. The model weights are downloaded by Ultralytics when needed; they are intentionally excluded from GitHub. The app-side S-DEC checkpoint at `data/models/confidence_fusion.pt` is included as a small project artifact, but its presence does not mean it was validated by the paper benchmark.

## Run the LLVIP research workflow

Open [RAVEN.ipynb](RAVEN.ipynb) in Google Colab and run the cells from top to bottom. Select a GPU runtime for practical run time, mount Google Drive, and provide the LLVIP ZIP or extracted folder when prompted. The notebook uses the official train/test split, checks image-stem overlap, and writes the run archive and tables under `/content/raven_results/`.

The saved benchmark uses frozen YOLOv8n detectors for both visible and thermal inputs and evaluates all 3,463 official LLVIP test pairs. The notebook supports optional detector pairings, but only the `v8n_v8n` pairing has saved results in this version. Do not report results for another pairing until its complete workflow has been run and its `pairing_summary.json` is saved.

### Measured LLVIP results in the notebook

Full-feature RAVEN reached mean mAP@50 0.7083 ± 0.0029 across three risk-head seeds, compared with 0.6961 for late-average fusion. The paired image-bootstrap 95% interval was +0.0065 to +0.0187. The confidence-plus-match ablation was slightly higher at 0.7088 ± 0.0019, so the experiment does not establish that geometry improves results.

In the clean test, weighted-box fusion (WBF) scored 0.7210 mAP@50, above RAVEN. Across the tested simulated one-sensor degradation conditions, RAVEN exceeded late-average fusion but remained below WBF. Coverage stayed near 100%, so useful abstention was not demonstrated. Image-level bootstrap intervals do not account for scene clustering or all training uncertainty. Include false positives, misses, coverage and ablations with mAP; do not describe the risk estimates as calibrated probabilities or the system as safety-validated.

The clean-run CSVs and run summary are in [output/RAVEN_LLVIP_2026-10-15/](output/RAVEN_LLVIP_2026-10-15/). Re-run Colab and download the generated degradation tables and pairing summary to archive those results with the clean-run artifacts. Review [docs/PAPER_AUDIT.md](docs/PAPER_AUDIT.md) before using the manuscript results. The LLVIP images and annotations are not included; obtain them from the official dataset source and follow its terms.

## Contributions and limitations

RAVEN is a scoped research prototype, not a new detector architecture or a safety-validated system. Its research contribution is a specific annotation-derived, post-detection action-risk formulation and a controlled comparison of arbitration policies over frozen RGB and thermal detectors. The saved clean benchmark does **not** show that the learned head is best: WBF scores 0.7210, above RAVEN at 0.7083 ± 0.0029, and the confidence-plus-match ablation at 0.7088 ± 0.0019 is marginally above the full model. The study uses LLVIP only; independent-dataset generalization is untested. Sensor degradation experiments are synthetic. Risk scores are not calibrated probabilities, near-100% coverage does not demonstrate useful abstention, and the system has not been validated for real-world safety or deployment.

The checked-in clean results are reported notebook outputs; the repository audit identifies missing degradation exports and incomplete baseline rows in one benchmark CSV. Follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [docs/PAPER_AUDIT.md](docs/PAPER_AUDIT.md) before relying on the results or submitting the manuscript. Do not interpret application/dashboard outputs as benchmark measurements.

## Project layout

```text
RAVEN_PROJECT_BACKUP/
├── RAVEN.ipynb                 # Current LLVIP research notebook
├── app.py                      # Streamlit compatibility entry point
├── configs/                    # Local application configuration
├── dashboard/                  # Streamlit interface
├── frontend/                   # Next.js interface
├── src/                         # FastAPI and application modules
├── scripts/                    # Launch, setup and evaluation utilities
├── docs/                        # Protocol, reproducibility guide, architecture, paper audit
├── data/models/                 # Small local S-DEC checkpoint; other weights are ignored
├── data/local_assets/           # Local-only sample frames; not needed by the app
└── output/RAVEN_LLVIP_2026-10-15/ # Clean-run CSVs and summary
```

The research notebook uses YOLOv8n on both streams for its saved benchmark. The local application defaults in `configs/models.yaml` use YOLO26-L for RGB detection and pose, and YOLOv8n for thermal detection. These are separate configurations and results.

## Contributors

Based on the supplied project-epic allocation; contributors should verify before submission: Aaradhya More — project management ,literature review,research work and Novelty analysis; Atharva Khobrekar — LLVIP preparation and core implementation; Zain Shaikh — evaluation and experiments; Pranjal Nirmal — paper structure, documentation and presentation; Keerthana T — faculty guidance.

