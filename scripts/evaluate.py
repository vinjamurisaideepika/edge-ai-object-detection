"""Measure detection accuracy (mAP) of every exported model on COCO128.

COCO128 = the first 128 images of COCO train2017 (downloaded automatically,
~7 MB). It is small, so treat the numbers as a *relative* comparison
between FP32/INT8 and 640/320, not as official COCO scores.

    python scripts/evaluate.py models/*.onnx
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--data", default="coco128.yaml")
    p.add_argument("--output", default="results/accuracy.csv")
    args = p.parse_args()

    from ultralytics import YOLO

    rows = []
    for m in args.models:
        imgsz = int(re.search(r"_(\d{3})_", Path(m).name).group(1)) if re.search(r"_(\d{3})_", m) else 640
        metrics = YOLO(m, task="detect").val(data=args.data, imgsz=imgsz, batch=1, rect=False,
                                             plots=False, verbose=False)
        row = {"model": Path(m).name, "imgsz": imgsz,
               "mAP50": round(float(metrics.box.map50), 4), "mAP50_95": round(float(metrics.box.map), 4)}
        rows.append(row)
        print(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
