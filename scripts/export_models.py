"""Export and optimise YOLO11n for edge inference.

Produces, in models/:
  yolo11n_640_fp32.onnx   baseline, 640x640 input
  yolo11n_320_fp32.onnx   4x fewer pixels  -> faster, less accurate on small objects
  yolo11n_640_int8.onnx   static INT8 quantization (QDQ), calibrated on real images
  yolo11n_320_int8.onnx
  yolo11n_*_fp32.tflite / *_int8.tflite   (only with --tflite, needs litert-torch)

Run on your Mac (not in the Docker container):
    pip install -r requirements-export.txt
    python scripts/export_models.py            # ONNX only
    python scripts/export_models.py --tflite   # ONNX + TFLite

INT8 calibration uses the 8 COCO8 images (downloaded automatically), which
are NOT part of COCO128, the set used by evaluate.py. Calibrating on your
test images would make the INT8 accuracy look better than it really is.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edgevision.capture import FrameSource  # noqa: E402
from edgevision.preprocess import preprocess  # noqa: E402


def export_onnx(weights: str, imgsz: int, out: Path) -> Path:
    from ultralytics import YOLO

    path = YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False, opset=17)
    shutil.move(path, out)
    print(f"[onnx] {out.name}: {out.stat().st_size / 1e6:.1f} MB")
    return out


class ImageCalibrationReader:
    """Feeds preprocessed real images to the quantizer so it can measure the
    real range of every activation (needed to choose INT8 scales)."""

    def __init__(self, input_name: str, imgsz: int, calib_dir: str, max_images: int):
        frames = list(FrameSource(calib_dir))[:max_images]
        self.batches = iter([{input_name: preprocess(f, imgsz)[0]} for f in frames])
        print(f"[int8] calibrating with {len(frames)} images")

    def get_next(self):
        return next(self.batches, None)


def quantize_int8(fp32: Path, imgsz: int, calib_dir: str, max_images: int, out: Path) -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre))  # shape inference + graph clean-up
    input_name = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    reader = ImageCalibrationReader(input_name, imgsz, calib_dir, max_images)
    CalibrationDataReader.register(type(reader))
    quantize_static(
        str(pre), str(out), reader,
        quant_format=QuantFormat.QDQ,      # standard format, runs everywhere
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,                  # better accuracy for conv weights
        # The final detection head mixes box coordinates (0..640) with
        # class scores (0..1); keeping it in float protects accuracy.
        nodes_to_exclude=_head_nodes(pre),
    )
    pre.unlink(missing_ok=True)
    print(f"[int8] {out.name}: {out.stat().st_size / 1e6:.1f} MB")
    return out


def _head_nodes(model_path: Path) -> list[str]:
    """Names of the nodes after the last 'model.23' block (the decode head)."""
    import onnx

    nodes = onnx.load(str(model_path)).graph.node
    return [n.name for n in nodes if "/model.23/" in n.name and ("dfl" in n.name.lower() or
            any(k in n.op_type for k in ("Sigmoid", "Concat", "Split", "Softmax", "Sub", "Add", "Div", "Mul")))]


def export_tflite(weights: str, imgsz: int, models: Path) -> None:
    """TFLite export through Ultralytics (uses Google's litert-torch converter).
    INT8 calibration uses COCO8, so it never sees the COCO128 test images."""
    from ultralytics import YOLO

    for int8 in (False, True):
        path = Path(YOLO(weights).export(format="tflite", imgsz=imgsz, int8=int8, data="coco8.yaml"))
        out = models / f"yolo11n_{imgsz}_{'int8' if int8 else 'fp32'}.tflite"
        shutil.move(path, out)
        print(f"[tflite] {out.name}: {out.stat().st_size / 1e6:.1f} MB")


def prepare_calibration_dir(name: str) -> str:
    """'coco8' -> download COCO8 and copy its 8 images into ./calib/"""
    if name != "coco8":
        return name
    from ultralytics.data.utils import check_det_dataset

    root = Path(check_det_dataset("coco8.yaml")["path"])
    calib = ROOT / "calib"
    calib.mkdir(exist_ok=True)
    for img in root.glob("images/*/*.jpg"):
        shutil.copy(img, calib / img.name)
    return str(calib)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 320])
    p.add_argument("--calib", default="coco8", help="'coco8' or a folder of calibration images")
    p.add_argument("--calib-images", type=int, default=100)
    p.add_argument("--tflite", action="store_true", help="also export TFLite (needs litert-torch)")
    args = p.parse_args()

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    args.calib = prepare_calibration_dir(args.calib)
    for imgsz in args.sizes:
        fp32 = export_onnx(args.weights, imgsz, models / f"yolo11n_{imgsz}_fp32.onnx")
        quantize_int8(fp32, imgsz, args.calib, args.calib_images, models / f"yolo11n_{imgsz}_int8.onnx")
        if args.tflite:
            export_tflite(args.weights, imgsz, models)


if __name__ == "__main__":
    main()
