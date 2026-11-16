# Edge AI Object Detection on an Embedded Linux Platform

[![tests](https://github.com/GITHUB_USER/edge-ai-object-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/GITHUB_USER/edge-ai-object-detection/actions)

An end-to-end, **real-time object detection pipeline** for resource-constrained
**ARM64 Linux** devices, built from modular components:

```
 capture ──> preprocess ──> inference backend ──> postprocess ──> visualize
 (camera,     (letterbox,     ONNX Runtime         (decode YOLO,    (boxes,
  video,       RGB, NCHW/      LiteRT / TFLite      class-aware      FPS
  images)      NHWC tensor)    PyTorch (reference)  NMS, unscale)    overlay)
```

The model is **YOLO11n** (2.6 M parameters), exported and optimised into
several variants to measure the **trade-offs between model size, inference
latency and detection accuracy**:

| Variant | Optimisation |
|---|---|
| `yolo11n_640_fp32` | baseline, 640×640 input |
| `yolo11n_320_fp32` | 4× fewer input pixels |
| `yolo11n_*_int8.onnx` | static INT8 quantization (QDQ, per-channel), calibrated on COCO8; detection head kept in float |
| `yolo11n_*_int8.tflite` | INT8 TFLite via Google's LiteRT converter |

## Target platform

The pipeline runs in an **ARM64 (aarch64) Debian container**, the same CPU
architecture and OS family as a Raspberry Pi 4/5, with Docker CPU and memory
limits to emulate a small board (`pi4`: 4 cores / 4 GB, `tiny`: 1 core / 1 GB).
On an Apple Silicon Mac the container runs natively, without emulation.
Absolute timings are faster than on a real Pi (the cores are faster), so
the comparison **between variants** is what matters.

## Results

<!-- Paste results/REPORT.md here after running the benchmark, and add the chart: -->
<!-- ![trade-offs](results/tradeoff.png) -->

Accuracy measured with `scripts/evaluate.py` on COCO128 (it does not depend on the hardware):

| Model | mAP50-95 |
|---|---:|
| 640 FP32 (ONNX) | 0.505 |
| 640 INT8 (ONNX) | 0.490 |
| 320 FP32 (ONNX / TFLite) | 0.386 |
| 320 INT8 (TFLite) | 0.377 |

Key observations (fill in with your own numbers):
- INT8 cuts model size by ~3.3× for ~1.5 mAP points.
- Halving the input resolution is the biggest speed lever, but costs ~12 mAP points (small objects disappear).
- Whether INT8 is faster depends on the CPU and runtime: compare ONNX Runtime vs LiteRT on ARM64.

## Quick start

```bash
# 1. on the Mac: export models (needs the heavier tools)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-export.txt
python scripts/export_models.py --tflite
python scripts/evaluate.py models/*.onnx models/*.tflite

# 2. inside the ARM64 "edge device" container
docker compose build
docker compose run --rm pi4  detect --model models/yolo11n_320_int8.tflite --source samples
docker compose run --rm pi4  benchmark --models models/*.onnx models/*.tflite --source samples
docker compose run --rm tiny benchmark --models models/*.onnx models/*.tflite --source samples --output results/benchmark_tiny.csv

# 3. merge results into a table + chart
python scripts/make_report.py
```

Live webcam (natively on the Mac, Docker on macOS cannot access cameras):
`python -m edgevision detect --model models/yolo11n_320_int8.onnx --source 0 --show`

## Project layout

```
src/edgevision/   capture.py  preprocess.py  backends.py  postprocess.py  pipeline.py  visualize.py  cli.py
scripts/          export_models.py  evaluate.py  make_report.py
docker/           Dockerfile (python:3.11-slim, linux/arm64)
tests/            unit tests for letterbox, IoU, NMS and YOLO decoding
```

The pipeline's output was verified to match the Ultralytics reference
implementation pixel for pixel on the same ONNX model.

*Model weights: YOLO11n by Ultralytics (AGPL-3.0). They are downloaded by the
export script and are not stored in this repository.*

## What I learned
<!-- 3-5 bullets in your own words -->
