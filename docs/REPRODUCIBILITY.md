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

The checked-in README and audit report the following saved results. These numbers are transcribed as existing reported results, not independently rerun in this documentation change.

| Method | Reported custom AP@50 / mAP@50 |
|---|---:|
| Late-average fusion | 0.6961 |
| RAVEN full model | 0.7083 ± 0.0029 |
| Confidence-plus-match ablation | 0.7088 ± 0.0019 |
| Weighted-box fusion (WBF) | 0.7210 |

The standard deviation for the RAVEN and confidence-plus-match rows is reported across three risk-head seeds. The audit records a paired image-bootstrap 95% interval of +0.0065 to +0.0187 for the RAVEN versus late-average comparison. This is an image-level interval, not a scene/clip-clustered interval, and it does not account for all training uncertainty.

Interpret these results conservatively: WBF is higher on the clean benchmark; confidence-plus-match is slightly higher than the full model; these results do not establish that the learned head improves over a simple confidence rule. WBF also fuses and can move boxes, while arbitration methods select from a shared candidate/action set, so this is not a perfectly like-for-like operation.

## Required output archive

Archive the exact files emitted by a completed run. The repository audit noted that the current clean-run folder does not contain all outputs described by the paper/notebook. In particular, do not claim the following files are archived until they are present and inspected:

- `run_summary.json`
- `table8_benchmark.csv` with all methods discussed in the paper
- Clean benchmark CSVs for each reported method/seed
- `degradation_summary.csv`
- `degradation_per_run.csv`
- `degradation_bootstrap.csv`
- `pairing_summary.json`, only for detector pairings actually evaluated
- A run configuration containing seed list, checkpoint IDs, thresholds, split, metric/matching rule, bootstrap settings, package versions, and timestamp

The audit found no checked-in `degradation_summary.csv`, `degradation_per_run.csv`, `degradation_bootstrap.csv`, or `pairing_summary.json`; it also found that `table8_benchmark.csv` omits max-confidence and WBF references discussed in the paper. Regenerate and verify these artifacts before treating them as available. Do not create placeholder CSVs or synthetic summaries to make the archive appear complete.

## Bootstrap and degradation experiments

Use the notebook's implemented bootstrap procedure and record the seed and resampling unit in the exported configuration. Resolve the documented discrepancy between the paper's Table 12 interval and the notebook's saved/displayed degradation-bootstrap bounds by exporting the exact bootstrap table from the same run or revising the manuscript to match it.

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
