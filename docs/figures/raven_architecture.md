# RAVEN architecture

This diagram is a documentation-level summary of the current LLVIP research workflow. It is intentionally not a claim that every component is trained jointly.

```mermaid
flowchart LR
    A[Paired LLVIP RGB image] --> B[Frozen YOLOv8n RGB detector]
    T[Paired LLVIP thermal image] --> C[Frozen YOLOv8n thermal detector]
    B --> D[Candidate association and matching]
    C --> D
    D --> E[Candidate/action features]
    L[LLVIP annotations] --> F[Annotation-derived action-risk targets]
    E --> G[Lightweight risk head]
    F --> G
    G --> H{Selection policy}
    D --> H
    H --> I[Joint / RGB-only / thermal-only]
    H --> J[Optional abstention]
    I --> K[LLVIP test evaluation]
    J --> K
    K --> M[Custom all-point AP@50, errors, coverage, bootstrap]
```

## Component boundaries

- **Inputs:** paired visible-light RGB and thermal images from LLVIP.
- **Detectors:** two frozen COCO-pretrained YOLOv8n detectors; one per modality.
- **Association:** candidate detections are matched to create the candidate/action set.
- **Supervision:** annotation-derived action-risk targets train a lightweight head; this is not end-to-end detector training.
- **Policy:** select joint, RGB-only, or thermal-only output, with optional abstention.
- **Evaluation:** official LLVIP test split, using the notebook's custom all-point AP@50 matching rule plus reported error/coverage and bootstrap analyses.

The clean benchmark reports WBF above RAVEN and the confidence-plus-match ablation marginally above the full model. The architecture diagram does not imply state-of-the-art performance, calibrated risk probabilities, safety validation, or generalization beyond LLVIP.
