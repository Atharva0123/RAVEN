# Thermal inputs, tracking, and model-evaluation limits

RAVEN is a research prototype and feasibility study. This note documents the current thermal-input boundary and the data needed to evaluate the system; it does not claim deployment validation.

## Sensor inputs and thermal model

The checked-in model configuration uses YOLOv8 Nano for thermal-palette person detection, with RGB and pose checkpoints configured separately in `configs/models.yaml`. A thermal-palette image is not necessarily calibrated radiometric data. Unless its camera and calibration metadata support temperature measurement, treat its colors as image values only.

RGB uploads can produce an estimated thermal-style video for display. `src/thermal/processing.py` derives that image from RGB luminance, so it contains no independent sensor evidence and no measured temperature. It is not provided to quality-aware or temporal cross-modal fusion. The generic streaming helper also excludes this derived view from fusion calculations.

LLVIP provides aligned visible/infrared image pairs and person annotations that can support pedestrian and detection-consensus studies. It does not provide action or anomaly labels. Its terms apply to the data and derived artifacts; review the current [official project page](https://bupt-ai-cz.github.io/LLVIP/), [paper](https://arxiv.org/abs/2108.10831), and [terms](https://github.com/bupt-ai-cz/LLVIP/blob/main/Term%20of%20Use%20and%20License.md) before use or distribution. Preserve the official scene split and do not use the reported calibrator pilot as action/threat validation.

## Tracking and Re-ID

BoT-SORT settings in `configs/botsort_raven.yaml` retain lost tracks for 90 frames with a matching threshold of 0.8. RGB Re-ID uses a coarse HSV plus Sobel-gradient descriptor; thermal Re-ID uses luminance layout and gradient features. These are lightweight appearance heuristics, not a learned person re-identification model and not biometric identification. Same-looking clothing, small crops, occlusion, viewpoint changes, and low resolution can split or confuse IDs. Report identity switches and track metrics on labeled, camera-separated data before making accuracy claims.

Trajectory overlays use track boxes and are a visualization of estimated movement. They are not evidence of intent or event class.

## Action and temporal event ranking

`data/models/best_model.pth` is marked synthetic in its metadata and is disabled for scoring. It validates architecture/checkpoint/data flow only. `data/models/action_anomaly.pt` and its metrics are also marked synthetic and are disabled for model-driven review scores. With no eligible, non-synthetic action checkpoint, RAVEN may display `PROTOTYPE_REVIEW_CUE` from a transparent motion heuristic. That cue is neither a trained action label nor a threat probability.

UCF-Crime contains video-level anomaly categories, including shooting, but its broad video labels do not identify each person or give precise actor-level event intervals. It can support a carefully scoped weakly supervised video-ranking experiment if its current access/use terms permit it. For person-level or precise temporal claims, annotate the event intervals and actors, split by source video, and hold out a separate test set. Do not treat generated motion examples or an unlabeled clip as training or validation labels.

## Review policy

An object detection above the configured threshold is tagged `OBJECT_DETECTION_CANDIDATE`; it may receive workflow level `ALERT`, but it is not a confirmed threat. API and event outputs require operator review and state `confirmed_threat: false`. Model/heuristic anomaly signals are tagged `PROTOTYPE_REVIEW_CUE`; synthetic scores must be identified `SYNTHETIC_DEMO` if they are shown for a separate software demonstration. Every event must be checked against the source video by a human.
