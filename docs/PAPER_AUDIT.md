# RAVEN paper audit

Reviewed: `docs/manuscripts/RAVEN_final_updated_v2.docx` against the current root notebook, `RAVEN.ipynb`, and archived run artifacts under `output/`, including the compact `v8n_v8n` bundle.

## Assessment

The paper is substantially more careful than an earlier promotional draft. It limits the scope to LLVIP, explains that YOLOv8n detectors are frozen, separates the dashboard examples from measured results, reports error counts and ablations, and explicitly says that the risk scores are not calibrated probabilities and that RAVEN is not safety-validated. Its strongest contribution is a reproducible comparison of a small post-detection arbitration head with simple policies, including a useful negative result: the learned head is about equal to max-confidence and below weighted boxes fusion (WBF).

The method is incremental rather than a strong new detector or a clear state-of-the-art result. Describe its novelty as a specific annotation-derived, post-detection action-risk formulation and controlled LLVIP study. Do not claim first-of-kind, general superiority, safety, calibration, or cross-dataset generalization.

## Checked against the notebook and run files

- The notebook and paper use LLVIP only, with two frozen COCO-pretrained YOLOv8n detectors, one for RGB and one for thermal. KAIST is cited as background, not used as an evaluation dataset.
- The reported sample sizes agree: 506 head-training candidates from 150 LLVIP training pairs, split into 431 training and 75 validation candidates; the official test split has 3,463 pairs.
- The paper's rounded clean benchmark and degradation mAP values agree with the executed notebook summaries. The 867-parameter head, seeds 42/123/2025, three ablations, and threshold coverage near 100% also agree.
- The paper correctly explains that WBF is a different reference: it fuses raw detections and can move boxes, whereas late-average fusion and RAVEN arbitrate over the shared candidate/action set. Keep that distinction wherever the comparison is discussed.

## Resolve before submission

1. **Use one clean-data confidence interval throughout.** The compact archive’s `degradation_bootstrap.csv` gives +0.0065 to +0.0183 for clean RAVEN versus late-average fusion, matching Table 12 when rounded. The README abstract/conclusion summary previously gave +0.0065 to +0.0187; align those sections and every manuscript reference with the exported interval before submission. This image-level bootstrap does not account for scene clustering or all training uncertainty.
2. **Complete the data/code availability statement.** It still contains `[Public repository link to be added.]`; add the real public repository or Colab link only after it works. The compact `output/RAVEN_LLVIP_2026-10-15_v8n_v8n/` archive now includes `degradation_summary.csv`, `degradation_per_run.csv`, `degradation_bootstrap.csv`, and `pairing_summary.json`. The four-row `table8_benchmark.csv` is a compact clean table; the degradation summary additionally provides max-confidence, WBF, and ablation rows. Keep the older seed-42 archive separate. The compact archive omits model weights, checkpoints, per-image records/error analysis, and training logs, so do not describe it as a complete rerun package.
3. **Make the evaluation metric name precise.** Section 4.2 states that matching selects the best still-unused ground-truth box, unlike strict Pascal VOC matching, where the highest-overlap annotation is considered first and a duplicate match counts as a false positive. The paper discloses this difference, which is good, but “VOC-style mAP@50” can still be mistaken for the standard metric. Either use a verified standard evaluator and recompute every method, or label the existing result explicitly as custom all-point AP@50 under the stated one-to-one matching rule. Keep the same evaluator for all compared arbitration methods.
4. **Align the notebook and paper title.** The paper is titled “RAVEN: Lightweight Post-Detection Risk-Based Arbitration for Frozen RGB–Thermal Pedestrian Detectors”; the notebook heading still says “Post-Hoc Absolute Risk Minimization and Selective Governance…”. Use one title and make the notebook identify this exact manuscript as its companion.
5. **Expand dataset and detector names on first use.** The abstract calls LLVIP the “LLVIP visible-infrared dataset” without spelling out *Low-Light Visible-Infrared Paired (LLVIP)*, and YOLOv8n appears before its family name is explained. Expand those at first use and keep the short forms consistent. **Verify equations and figures visually.** The DOCX contains 21 native Word math objects and six embedded images, so the formulas are represented as editable equation objects rather than merely flattened text. I could not complete the render-and-inspect pass because LibreOffice (`soffice.exe`) is unavailable in this environment. Open the DOCX in Word or render it in an environment with LibreOffice, then inspect all pages, especially equations (1)–(21), table numbering, and figure captions, before submission.
6. **Finish author and institutional checks.** Confirm the spelling and preferred full name for “Keerthana T,” all email addresses, contribution statements, and the ethics sentence marked for author confirmation. These are author/institution checks, not facts I can infer from the notebook.

## Novelty and claims

The defensible novelty is narrow: per-candidate selection among joint, RGB-only, and thermal-only outputs from frozen paired detectors, using annotation-derived action-risk targets and optional abstention. Related multispectral fusion and risk-based sensor selection are cited, so frame this as a scoped implementation and controlled study. The results do not show that learning improves over a max-confidence rule; the confidence-plus-match ablation is slightly higher than the full model, and WBF is higher than all selection policies. The paper already states this honestly; keep that conclusion prominent.

The five single-sensor degradations are synthetic and do not establish robustness to real glare, darkness, or thermal failure. The training sample excludes empty frames, all seeds share the same sampled images, the validation set is small, and the bootstrap does not group by scene/clip. Retain these as limitations. No additional dataset is needed to make these LLVIP-only claims clearer; independent-dataset generalization must remain explicitly untested.

## Project file placement

- Current research code and run instructions: root `RAVEN.ipynb` and `README.md`.
- Current paper working copy: `docs/manuscripts/RAVEN_final_updated_v2.docx`.
- Superseded draft: `docs/manuscripts/archive/RAVEN_revised_from_colab.docx`.
- Archived source artifacts: `output/RAVEN_LLVIP_2026-10-15/` (separate seed-42 run) and `output/RAVEN_LLVIP_2026-10-15_v8n_v8n/` (compact results bundle).
- Curated GitHub upload: `github_upload/RAVEN/` and `github_upload/RAVEN_GitHub_ready.zip`. The manuscript remains outside this upload until the checks above are resolved; the dataset, private `.env`, and large detector weights remain excluded.
