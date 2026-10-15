# RAVEN — How to Run for Faculty Demo

> Everything you need is on the pen drive at `E:\RAVEN_PROJECT_BACKUP`
> No retraining. No internet required (except first-time setup).

---

## Before Faculty Day — Do This Once at Home

### Step 1 — Copy from pen drive to laptop
Open PowerShell and run:
```powershell
robocopy "E:\RAVEN_PROJECT_BACKUP" "C:\RAVEN" /E
```
Wait for it to finish (~2 min).

### Step 2 — Copy the .env file
```powershell
Copy-Item "E:\RAVEN_PROJECT_BACKUP\.env" "C:\RAVEN\.env" -Force
```

### Step 3 — Install Python (if not already installed)
Download Python 3.11 from https://python.org → tick **"Add to PATH"** during install.

### Step 4 — Install dependencies
```powershell
cd C:\RAVEN
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
> ⚠️ This takes 5–10 minutes the first time. Do it at home, not in front of faculty.

### Step 5 — Test run once
```powershell
cd C:\RAVEN
.venv\Scripts\activate
streamlit run dashboard/app.py
```
Browser should open at `http://localhost:8501`. If it works → you're ready. ✅

---

## On Faculty Day — What to Do

### 1. Open PowerShell
Press `Win + R` → type `powershell` → Enter

### 2. Run these 3 commands
```powershell
cd C:\RAVEN
.venv\Scripts\activate
streamlit run dashboard/app.py
```

### 3. Browser opens automatically
Go to `http://localhost:8501` if it doesn't open by itself.

### 4. Show the demo
- Upload or select one of the demo videos from `C:\RAVEN\demo_videos\`
- Best ones to show:
  - `demo_scene_08_busy_intersection.mp4` — headlight glare scenario
  - `demo_scene_01_alley_lowlight.mp4` — low-light scenario
- The dashboard shows RGB + thermal side by side with RAVEN's decision

### 5. Show the paper
Open this PDF directly:
```
C:\RAVEN\output\pdf\RAVEN_Faculty_Format_Research_Report.pdf
```
Point faculty to:
- **Table 6** — model architecture + training details
- **Table 7** — benchmark results (mAP, Miss Rate, FPPI)
- **Table 8** — bootstrap confidence intervals

---

## Key Numbers to Say (If Asked)

| What | Value |
|---|---|
| Full-feature RAVEN mAP@50 | **0.7083 ± 0.0029** (3 seeds) |
| Late Average Fusion mAP@50 | 0.6961 |
| Confidence + match ablation | 0.7088 ± 0.0019 (slightly higher) |
| Paired image-bootstrap ΔmAP@50 | +0.0122; 95% interval [+0.0065, +0.0187] |
| Test pairs | 3,463 (full LLVIP test set) |
| S-DEC parameters | **867** |
| Colab CPU head-only latency | **0.049 ms mean** |
| Training data | 506 candidates (431 train / 75 validation), reused across seeds |

These are corrected research-run metrics, not dashboard-checkpoint results. The confidence-plus-match ablation is slightly ahead, so do not claim the geometry features caused the gain. Coverage is almost 100%; active abstention was not demonstrated.
---

## If Something Goes Wrong

| Problem | Fix |
|---|---|
| `streamlit not found` | Run `.venv\Scripts\activate` first |
| Browser doesn't open | Manually go to `http://localhost:8501` |
| Port already in use | Run `streamlit run dashboard/app.py --server.port 8502` |
| Missing module error | Run `pip install -r requirements.txt` again |
| YOLO model not loading | All `.pt` files are in `C:\RAVEN\` subfolders — already included |

---

## What You Do NOT Need to Do
- ❌ Run the Colab notebook — training is done, checkpoint is saved
- ❌ Download LLVIP — not needed for the demo
- ❌ Retrain anything — `confidence_fusion.pt` is already in the backup
- ❌ Internet connection for the demo — everything runs locally

