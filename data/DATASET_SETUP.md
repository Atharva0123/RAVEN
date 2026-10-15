# Dataset Setup for RAVEN

The pipeline relies on several datasets for full training and evaluation. As per the instructions, download and place them here. The code will handle their absence gracefully by using placeholders or default models if the data is not yet available.

## 1. LLVIP (RGB+Thermal Person Detection)
- **URL**: https://bupt-ai-cz.github.io/LLVIP/
- **License**: Non-commercial academic. Requires short signup form.
- **Action**: Download and place the images in `data/raw/LLVIP/`.
- **Citation**: Please cite the LLVIP paper when downloading and using this dataset.

## 2. LLVIP-Pose
- **URL**: https://github.com/MickaelCormier/llvip-pose
- **Action**: Clone the repository or download the JSON annotations. Place them in `data/raw/LLVIP-Pose/`.
- **Citation**: Cite both LLVIP and LLVIP-Pose papers.

## 3. Market-1501
- **Purpose**: Used for Re-ID Rank-1/mAP validation (optional).
- **Action**: Place in `data/raw/Market-1501/`.

## 4. Self-Recorded Clips
- **Purpose**: Ground truth for posture/threat classification.
- **Action**: Record 15-20 short clips of team members performing:
  - Walking
  - Crawling
  - Climbing
  - Running
- **Placement**: Place these videos in `data/recorded_clips/`. These will be used to train the fusion/posture classifiers.
