# Reproducibility guide

This guide describes the checked-in LLVIP research workflow. It deliberately distinguishes the research notebook from the local web application; application outputs and dashboard screenshots are not benchmark evidence.

## Scope and status

- Dataset: Low-Light Visible-Infrared Paired (LLVIP), using its official train/test split.
- Saved detector pairing: frozen COCO-pretrained YOLOv8n on RGB and frozen COCO-pretrained YOLOv8n on thermal.
- Saved test set: 3,463 official LLVIP test pairs.
- Risk head: lightweight post-detection head trained from annotation-derived action-risk targets. The notebook audit reports 506 candidate examples from 150 training pairs, split into 431 training and 75 validation candidates.
- Reported seeds: 42, 123, and 2025.
- Scope: LLVIP only. No KAIST evaluation or independent-dataset generalization is established.

The existing repository audit identifies the primary metric as **custom all-point AP@50 under the notebook's stated one-to-one matching rule**. Do not describe it simply as standard Pascal VOC mAP without changing the evaluator and recomputing every method. The matching procedure selects the best still-unused ground-truth box, which differs from strict VOC matching behavior. Use the same evaluator for all compared arbitration methods and disclose the rule.

## Dataset acquisition and layout

The LLVIP dataset is not redistributed here. Obtain it from the official LLVIP project/source and follow its terms. Do not commit dataset images, annotations, private credentials, or large detector weights.

The notebook accepts an LLVIP ZIP or extracted directory in Colab after Google Drive is mounted. Use the official train/test split and retain the notebook's image-stem overlap check. Keep paired visible and infrared images correctly matched. Before running, inspect the notebook's dataset-discovery cells and adapt only the input path if your directory layout differs; do not silently alter the split.

## Environment

The repository's root `requirements.txt` is for the local application and is **not** a complete research-notebook lock file. The notebook uses Colab and PyTorch/Ultralytics. Record the actual runtime before a full run:

- Python, platform, CUDA runtime, GPU model, and PyTorch version
- Ultralytics version and detector checkpoint identifiers/hashes
- NumPy, OpenCV, Pillow, and notebook dependency versions
- Git commit SHA, dataset source/version, and official split file hashes
- All seeds, confidence/IoU thresholds, image size, matching rule, and bootstrap settings

Do not label an environment as locked or exact until these values are captured from the successful run. A fresh Colab runtime can resolve different package versions.

## Run the notebook

1. Open [`RAVEN.ipynb`](../RAVEN.ipynb) in Google Colab.
2. Select a GPU runtime where available and mount Google Drive when prompted.
3. Provide the LLVIP archive or extracted directory.
4. Run cells from top to bottom without skipping detector, candidate, target, evaluation, or export stages.
5. Save the complete notebook and all generated outputs from the same run.
6. Review warnings and confirm the evaluated test-pair count is 3,463 and that no train/test image-stem overlap is detected.

The notebook writes its result archive under `/content/raven_results/` in Colab. Copy generated outputs into a versioned folder under `output/` only after checking that each file was produced by the run being documented.

## Existing clean benchmark

The compact archive at [`output/RAVEN_LLVIP_2026-10-15_v8n_v8n/`](../output/RAVEN_LLVIP_2026-10-15_v8n_v8n/) contains the `v8n_v8n` run summary, pairing metadata, clean benchmark, aggregate and per-run degradation tables, and paired bootstrap intervals. Its aggregate table contains the clean WBF and three-seed RAVEN rows. The older `output/RAVEN_LLVIP_2026-10-15/` directory is a separate seed-42 run (RAVEN AP50 0.7056); keep the folders distinct. The compact archive excludes weights, checkpoints, per-image records/error analysis, and training logs, so it supports inspection of reported tables but is not a complete rerun package.

| Method | Reported custom AP@50 / mAP@50 |
|---|---:|
| Late-average fusion | 0.6961 |
| RAVEN full model | 0.7083 ± 0.0029 |
| Confidence-plus-match ablation | 0.7088 ± 0.0019 |
| Weighted-box fusion (WBF) | 0.7210 |

The standard deviations for the RAVEN and confidence-plus-match rows are across three risk-head seeds; see `degradation_per_run.csv` for the seed-level records. The archived `degradation_bootstrap.csv` gives a clean paired image-bootstrap 95% interval of +0.0065 to +0.0183 for RAVEN versus late-average fusion. This is an image-level interval, not a scene/clip-clustered interval, and it does not account for all training uncertainty.

Interpret these results conservatively: WBF is higher on the clean benchmark; confidence-plus-match is slightly higher than the full model; these results do not establish that the learned head improves over a simple confidence rule. WBF also fuses and can move boxes, while arbitration methods select from a shared candidate/action set, so this is not a perfectly like-for-like operation.

## Required output archive

For the archived `v8n_v8n` run, the compact output folder contains the following selected files:

- `README.md` describing the run and the limits of the compact selection
- `run_summary.json` with configuration, versions, hashes, benchmark metrics, and evaluation metadata
- `pairing_summary.json` for the evaluated `v8n_v8n` detector pairing
- `table8_benchmark.csv` for the compact clean benchmark table
- `degradation_summary.csv` and `degradation_per_run.csv` for aggregate and seed-level condition results
- `degradation_bootstrap.csv` for paired image-bootstrap intervals
- `degradation_map50.png` for the degradation plot

The compact benchmark CSV has four rows; the aggregate degradation table additionally contains max-confidence, WBF, and the RAVEN ablations. Do not treat the older seed-42 folder as part of this run. Checkpoint files, serialized per-image records, image-level error analysis, and training logs were left out of the compact archive; the notebook and source code remain necessary for a fresh rerun. Do not create placeholder or synthetic outputs.

## Bootstrap and degradation experiments

Use the notebook's implemented bootstrap procedure and record the seed and resampling unit in the exported configuration. The archived clean comparison is +0.0065 to +0.0183 in `degradation_bootstrap.csv`; align the abstract and conclusion with this exported interval before submission. The image bootstrap does not account for scene/clip clustering or all training uncertainty.

The one-sensor degradations are synthetic perturbations, not physical sensor-failure trials. Report each perturbation and its parameters. They do not establish robustness to real darkness, glare, weather, hardware failure, or deployment conditions. If abstention coverage remains near 100%, do not claim meaningful selective risk reduction.

## Reproducing tables and figures

Generate tables and plots only from archived run outputs, and preserve the source CSV/JSON alongside each rendered artifact. Record the notebook cell or script and commit SHA used to generate each item. Do not use local dashboard examples as benchmark results. Do not report KAIST results or cross-dataset generalization unless an independent evaluation is run and archived.

## Pre-submission checklist

- [ ] Notebook and manuscript use the same title and method description.
- [ ] Metric name explicitly states custom all-point AP@50 and its matching rule, unless all results are recomputed with a standard evaluator.
- [ ] All baseline rows in the paper match the exported benchmark CSV.
- [ ] Clean-data and degradation bootstrap intervals match the archived CSV/JSON.
- [ ] Degradation and pairing artifacts are actually present and correspond to the reported run.
- [ ] Runtime, versions, detector checkpoints, thresholds, seeds, split and bootstrap settings are recorded.
- [ ] WBF's higher clean score, ablation results, near-100% coverage, and LLVIP-only scope are disclosed.
- [ ] Manuscript equations, tables, captions, ethics statement, author details and data/code links have been manually checked.
