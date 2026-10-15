# RAVEN paper audit

Reviewed: `docs/manuscripts/RAVEN_final_updated_v2.docx` against the current root notebook, `RAVEN.ipynb`, and local run artifacts under `output/RAVEN_LLVIP_2026-10-15/`.

## Assessment

The paper is substantially more careful than an earlier promotional draft. It limits the scope to LLVIP, explains that YOLOv8n detectors are frozen, separates the dashboard examples from measured results, reports error counts and ablations, and explicitly says that the risk scores are not calibrated probabilities and that RAVEN is not safety-validated. Its strongest contribution is a reproducible comparison of a small post-detection arbitration head with simple policies, including a useful negative result: the learned head is about equal to max-confidence and below weighted boxes fusion (WBF).

The method is incremental rather than a strong new detector or a clear state-of-the-art result. Describe its novelty as a specific annotation-derived, post-detection action-risk formulation and controlled LLVIP study. Do not claim first-of-kind, general superiority, safety, calibration, or cross-dataset generalization.

## Checked against the notebook and run files

- The notebook and paper use LLVIP only, with two frozen COCO-pretrained YOLOv8n detectors, one for RGB and one for thermal. KAIST is cited as background, not used as an evaluation dataset.
- The reported sample sizes agree: 506 head-training candidates from 150 LLVIP training pairs, split into 431 training and 75 validation candidates; the official test split has 3,463 pairs.
- The paper's rounded clean benchmark and degradation mAP values agree with the executed notebook summaries. The 867-parameter head, seeds 42/123/2025, three ablations, and threshold coverage near 100% also agree.
- The paper correctly explains that WBF is a different reference: it fuses raw detections and can move boxes, whereas late-average fusion and RAVEN arbitrate over the shared candidate/action set. Keep that distinction wherever the comparison is discussed.

## Resolve before submission

1. **Use one clean-data confidence interval throughout.** The abstract and conclusion give +0.0065 to +0.0187; this agrees when rounded with the saved `robust_paired_bootstrap.image_bootstrap_ci95` in `run_summary.json`. Table 12 gives a separate interval (+0.0065 to +0.0183), which the text attributes to a different resampling stream. The notebook’s displayed degradation-bootstrap bounds differ slightly from Table 12, and the local bundle has no `degradation_bootstrap.csv` to identify the exact run behind the table. Preserve the abstract interval with its matching summary, and regenerate/export the Table 12 bootstrap CSV (or update Table 12 to match that exported run) before submission.
2. **Complete the data/code availability statement.** It still contains `[Public repository link to be added.]`. Add the real public repository or Colab link only after it works. The local result bundle currently has clean-run tables and summaries but no `degradation_summary.csv`, `degradation_per_run.csv`, `degradation_bootstrap.csv`, or `pairing_summary.json`. Its `table8_benchmark.csv` also omits the max-confidence and WBF references shown in the paper. The notebook prints these results and writes the files during execution, but archive the generated files alongside the clean-run CSVs before claiming that the shared run outputs contain them. Otherwise narrow the availability wording to the artifacts actually shared.
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
- Clean-run source artifacts: `output/RAVEN_LLVIP_2026-10-15/`.
- Curated GitHub upload: `github_upload/RAVEN/` and `github_upload/RAVEN_GitHub_ready.zip`. The manuscript remains outside this upload until the checks above are resolved; the dataset, private `.env`, and large detector weights remain excluded.