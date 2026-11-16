# Reference results (cloud x86_64 CPU, 2 threads) - for comparison only

Your own numbers from the ARM64 Docker container on the Mac go in results/. Accuracy (mAP) does not depend on the hardware, so your mAP values should match these closely; latency will differ.

| Model | Runtime | Size (MB) | Input | Inference (ms) | p95 (ms) | End-to-end FPS | mAP50-95 (COCO128) |
|---|---|---:|---:|---:|---:|---:|---:|
| yolo11n_320_int8.tflite | tflite | 3.2 | 320 | 7.8 | 8.5 | 107.4 | 0.375 |
| yolo11n_320_int8.onnx | onnxruntime | 3.1 | 320 | 12.1 | 15.4 | 73.0 | 0.392 |
| yolo11n_320_fp32.onnx | onnxruntime | 10.6 | 320 | 14.3 | 19.6 | 62.6 | 0.386 |
| yolo11n_320_fp32.tflite | tflite | 10.7 | 320 | 15.0 | 17.5 | 60.1 | 0.386 |
| yolo11n_640_fp32.onnx | onnxruntime | 10.7 | 640 | 49.4 | 57.3 | 18.2 | 0.505 |
| yolo11n_640_int8.onnx | onnxruntime | 3.3 | 640 | 51.6 | 57.7 | 17.5 | 0.490 |
