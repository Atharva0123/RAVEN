# RAVEN research status and algorithm layers

RAVEN is a human-reviewed visible/thermal video research prototype. The attached paper's main experiment is a two-confidence-input MLP for person-detection consensus. It reports a 12-unit head trained with L-BFGS, 800 LLVIP training pairs and 200 held-out pairs, synthetic blur/darkness/glare corruptions, image-level bootstrap intervals, and a negative result for global frame-quality cues.

## Paper-aligned fusion

`src/fusion/confidence_head.py` implements the paper's two-input architecture. `scripts/prepare_confidence_fusion_data.py` defaults to frozen YOLOv8 Nano detectors, retains source-image IDs, groups image rows, includes annotated empty-person negatives, and writes a run manifest. `scripts/train_confidence_fusion.py` trains the two-input head with L-BFGS by default.

The previous six-input quality-aware implementation and its derived feature/checkpoint artifacts have been removed because they do not match the paper's method. The paper's metrics have not yet been regenerated from this checkout. Synthetic corruptions, held-out glare evaluation, image-level bootstrap intervals, and exact optimizer settings remain necessary for numerical reproduction.

## Other prototype components

- YOLO detectors, BoT-SORT tracking, pose estimation, appearance descriptors, FastAPI, and Next.js support the review application. Tracking is qualitative in the paper; do not claim tracking benchmark accuracy.
- TemporalCrossModalFusion is an architectural proof of concept with synthetic smoke-test weights, gated from scoring.
- RGB-derived pseudo-thermal imagery is a display transform, not an independent sensor or temperature measurement.
- Object detections are review candidates, not confirmed threats. Events require human review; the system does not infer intent.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\prepare_confidence_fusion_data.py --train-pairs 800 --validation-pairs 200 --output data\paper_experiment --seed 42
.\.venv\Scripts\python.exe scripts\train_confidence_fusion.py data\paper_experiment\train.csv data\paper_experiment\validation.csv data\models\confidence_fusion.pt --epochs 300 --seed 42 --optimizer lbfgs
```

This is a starting run, not the paper's exact numerical reproduction: it does not yet apply the paper's corruption schedule or calculate bootstrap intervals. Keep splits and uncertainty analysis at image level, and document the scene-leakage caveat until scene IDs are verified. LLVIP pedestrian labels do not support action/anomaly validation. Follow dataset and checkpoint terms before distribution.
