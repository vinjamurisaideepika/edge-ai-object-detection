# Edge AI Object Detection - context for Claude Code

Python package `edgevision` (src/edgevision): capture -> preprocess -> inference
backend (ONNX Runtime / LiteRT-TFLite / PyTorch) -> postprocess (decode + NMS) -> visualize.
Runs in an ARM64 Docker container with CPU/RAM limits to emulate an edge board.

## Commands
- Tests: `python -m pytest -q`
- Export models: `python scripts/export_models.py [--tflite]`
- Accuracy: `python scripts/evaluate.py models/*.onnx models/*.tflite`
- Speed: `docker compose run --rm pi4 benchmark --models models/*.onnx models/*.tflite --source samples`
- Report: `python scripts/make_report.py`

## Conventions
- Runtime code must only need `requirements.txt` (no ultralytics/torch on the device).
- Results in `results/` (CSV, REPORT.md, tradeoff.png) ARE committed; model files are NOT.

## Git rules for Claude Code
- Commit messages: Conventional Commits, `type(scope): summary` + short bullet body.
- One logical change per commit. Never commit build output, `Debug/`, `.DS_Store`,
  model weights (`*.pt`, `*.onnx`, `*.tflite`) or anything secret.
- Never rewrite history, never `git push --force`, never change git config.
- If the checks in `ship.sh` fail, stop and explain the failure instead of committing.
