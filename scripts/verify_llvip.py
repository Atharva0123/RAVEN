"""Audit an existing LLVIP release without modifying its raw files."""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/raw/LLVIP")
    root = parser.parse_args().root
    annotation_dir = root / "Annotations"
    annotation_ids = {path.stem for path in annotation_dir.glob("*.xml")}
    if not annotation_ids:
        raise SystemExit(f"No LLVIP VOC annotations found under {annotation_dir}")
    total_pairs = 0
    missing_visible, missing_thermal, missing_annotations = [], [], []
    for split in ("train", "test"):
        visible_dir, thermal_dir = root / "visible" / split, root / "infrared" / split
        visible = {path.stem for path in visible_dir.glob("*.jpg")}
        thermal = {path.stem for path in thermal_dir.glob("*.jpg")}
        pair_ids = visible & thermal
        total_pairs += len(pair_ids)
        missing_visible.extend(sorted(thermal - visible))
        missing_thermal.extend(sorted(visible - thermal))
        missing_annotations.extend(sorted(pair_ids - annotation_ids))
        print(f"{split}: visible={len(visible)} infrared={len(thermal)} paired={len(pair_ids)}")
    sample_id = min(annotation_ids)
    tree = ET.parse(annotation_dir / f"{sample_id}.xml")
    sample_person_boxes = sum(1 for item in tree.findall("object")
                              if (item.findtext("name") or "").strip().lower() == "person")
    print(f"annotations={len(annotation_ids)} sample_person_boxes[{sample_id}]={sample_person_boxes} "
          f"paired_images={total_pairs}")
    if missing_visible or missing_thermal or missing_annotations:
        raise SystemExit(f"Dataset audit failed: missing visible={len(missing_visible)}, "
                         f"thermal={len(missing_thermal)}, annotations={len(missing_annotations)}")
    print("LLVIP files are paired and annotated. Labels are pedestrian boxes, not anomaly labels.")


if __name__ == "__main__":
    main()
