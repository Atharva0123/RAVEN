"""Fine-tune a YOLOv8 thermal person detector on a prepared LLVIP infrared split.

Expected Ultralytics dataset YAML has one class (person) and points to the
official train/validation image and label folders. See docs/THERMAL_AND_TRACKING.md.
The original LLVIP download requires accepting its non-commercial research terms.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Ultralytics dataset YAML for IR images")
    parser.add_argument("--base", default="yolov8n.pt", help="YOLOv8 pretrained initialization")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "models" / "thermal_person_yolov8.pt")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset YAML does not exist: {args.data}")
    if args.epochs < 1 or args.batch < 1:
        parser.error("--epochs and --batch must be positive")

    import yaml
    config = yaml.safe_load(args.data.read_text(encoding="utf-8")) or {}
    names = config.get("names", {})
    values = list(names.values()) if isinstance(names, dict) else list(names)
    if config.get("nc") not in (None, 1) or values and [str(x).lower() for x in values] != ["person"]:
        parser.error("Thermal person fine-tuning expects exactly one class named 'person' (class 0)")
    if not config.get("train") or not config.get("val"):
        parser.error("Dataset YAML must define both train and val image paths")

    # Avoid oversubscribing RAM on local CPU-only Windows workstations.
    if args.device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    from ultralytics import YOLO
    model = YOLO(args.base)
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs,
                         imgsz=args.imgsz, batch=args.batch, device=device,
                         workers=args.workers, patience=max(8, args.epochs // 5),
                         project=str(ROOT / "runs" / "thermal"), name="llvip_yolov8",
                         exist_ok=True, cache=False, amp=device != "cpu", plots=False)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training ended without best.pt at {best}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.output)
    print(f"best_checkpoint={args.output.resolve()}")
    print("Set RAVEN_THERMAL_MODEL to this checkpoint before launching the API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
