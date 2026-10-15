# Fusion and decision governance

## Active post-hoc risk governance

`PostHocRiskHead` in [`confidence_head.py`](confidence_head.py) is the S-DEC Config B head used by the paired visible/thermal pipeline. It runs after the frozen detectors and has **867 trainable parameters** (`8 -> 32 -> 16 -> 3`, with ReLU hidden layers and sigmoid risk outputs). The checkpoint path is `data/models/confidence_fusion.pt`; the dashboard reloads the pipeline when this checkpoint changes.

The runtime feature contract, in order, is:

```text
[c_rgb, c_ir, abs(c_rgb - c_ir), box_iou, center_dist,
 scale_ratio, aspect_diff, matched_flag]
```

`center_dist` is the normalized center displacement, `scale_ratio` is the signed log area ratio, and `aspect_diff` is the absolute aspect-ratio difference. A missing modality has a zero confidence and a `matched_flag` of zero. The supplied bare checkpoint was trained with `matched_flag` in position four; `PostHocRiskHead.forward()` permutes the canonical feature vector to that checkpoint order before inference. Keep this adapter in place unless the head is retrained and the checkpoint schema is versioned.

The head estimates absolute task risk for joint, RGB, and thermal actions. The lowest estimated risk is selected unless all three risks exceed `tau = 0.45`, in which case the action is `Abstain / Human Review`.

| Action name | Operator label | Dashboard color |
|---|---|---|
| `joint_consensus` | Joint Consensus | Green (`#00FF00`) |
| `rgb_vetoed_thermal_dominant` | Thermal Dominant / RGB Veto | Cyan (`#00E5FF`) |
| `ir_vetoed_rgb_dominant` | RGB Dominant | Red (`#FF0000`) |
| `abstain` | Abstain / Human Review | Amber (`#FFB300`) |

The detector backbones remain frozen. Governance runs only when an independent thermal detection is available and associated with the visible candidate; RGB-derived pseudo-thermal imagery is not an independent sensor and does not qualify for cross-modal governance.

## Checkpoint and training provenance

The installed `data/models/confidence_fusion.pt` is the dashboard runtime artifact and is not the dated three-seed Colab research checkpoint. Its provenance and normalization are described by its adjacent runtime metadata; do not assume its metrics describe the corrected paper run. A checkpoint with a different architecture or parameter ordering must not be loaded as Config B.

The executed corrected S-DEC research notebook is at the project root as `RAVEN.ipynb`; its dated metrics, logs, per-image errors, and research checkpoints are archived under `output/RAVEN_LLVIP_2026-10-15/`. These artifacts document the Colab research run and are not automatically loaded by the dashboard. The active application checkpoint and its recorded metrics remain separate and were not replaced by the research checkpoint. See `scripts/README.md` for the distinction between the paper experiment and app implementation.

## Legacy confidence calibration

`ConfidenceFusionHead` remains available as a compatibility implementation for older two-input scalar checkpoints and existing callers. It is not the active S-DEC risk head. The old quality-aware/scalar fusion experiments are retired from the primary governance path; `quality_fusion.pt` is not the S-DEC checkpoint. The runtime may fall back to rule fusion if no compatible S-DEC checkpoint is available, and health metadata reports whether the risk head actually loaded.

`consensus.py`, `learned_consensus.py`, and `temporal_cross_modal.py` contain distinct rule-based or experimental components. Their scores are not interchangeable with the S-DEC action-risk triplet and should not be reported as S-DEC results without separate validation.



