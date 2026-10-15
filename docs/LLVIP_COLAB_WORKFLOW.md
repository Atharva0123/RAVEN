# LLVIP Colab workflow

## Notebook and run configuration

Use the project-root `RAVEN.ipynb`, the single maintained research notebook. Open it in Google Colab, choose a GPU runtime when available, and run cells in order. It installs its Python dependencies and downloads the detector weights from Ultralytics on first use.

The saved benchmark run selects `v8n_v8n` (YOLOv8n for visible and thermal) and uses all 3,463 official LLVIP test pairs. The notebook also defines `y26l_v8n` and `y26l_y26l`; `y26n_y26n` is optional and commented out. Only `v8n_v8n` has results saved in this notebook. The pairing-comparison cells need separately completed `pairing_summary.json` files before they can compare different pairings.

## Run steps

1. Upload `RAVEN.ipynb` to Colab and connect Drive when prompted.
2. Provide an extracted LLVIP directory or the LLVIP ZIP. Expected structure: `visible/{train,test}`, `infrared/{train,test}`, and VOC XML annotations.
3. Keep `CFG.num_test_pairs = None` for the full official test set. A limited subset is only a dry run.
4. Run the notebook from the top. Keep all three configured seeds and the listed ablations; do not select the best seed.
5. The notebook writes to `/content/raven_results/<pairing>/`. Download and retain the run summary, metrics, training logs, error analysis, checkpoints and degradation outputs together.
6. If running another detector pairing, set `RAVEN_PAIRING` before starting and run its entire workflow. Save its summary in the shared output folder. Do not compare pairings evaluated on different test counts or dry-run subsets.

## Evaluation and reporting

Training targets assign loss 1 to unavailable actions, and inference masks those same unavailable actions. Late-average fusion and RAVEN share detector outputs, candidate generation, association, filtering and evaluation; arbitration policy is the intended difference. The multi-seed analysis retrains the risk head while reusing the fixed candidate bank and image split.

The degradation section applies deterministic simulated darkness, blur, glare and thermal low-contrast changes to one stream at a time. It retains thresholds chosen on clean validation data and reports max-confidence and WBF alongside late-average fusion and RAVEN. These perturbations are not real sensor failures or field tests. The bootstrap resamples images and does not account for temporal/scene clustering.

The saved clean result is full-feature RAVEN mAP@50 0.7083 ± 0.0029 versus 0.6961 late-average fusion; confidence-plus-match is 0.7088 ± 0.0019. WBF reaches 0.7210 on clean LLVIP and remains stronger than RAVEN across the saved degradation comparisons. Coverage near 100% means useful abstention was not demonstrated. The model outputs are bounded risk estimates, not calibrated probabilities. This is an LLVIP-only research result, not safety validation.
