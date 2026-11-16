"""Inference backends: one small class per runtime, all with the same API.

    backend = load_backend("models/yolo11n.onnx")
    raw = backend.infer(tensor)       # tensor from preprocess()

Supported:
  * .onnx    -> ONNX Runtime (CPU)
  * .tflite  -> LiteRT / TensorFlow Lite (ai-edge-litert, tflite-runtime or tensorflow)
  * .pt      -> PyTorch through Ultralytics (reference / baseline only)

Each backend reports `layout` ("NCHW" or "NHWC") and `input_size`, so the
pipeline can preprocess correctly without knowing which runtime it uses.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Backend(ABC):
    name: str = "base"
    layout: str = "NCHW"
    input_size: int = 640

    def __init__(self, model_path: str | Path, threads: int | None = None):
        self.model_path = Path(model_path)
        self.threads = threads or os.cpu_count() or 1

    @abstractmethod
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run the network. Returns the raw output in input-pixel units."""

    @property
    def model_size_mb(self) -> float:
        return self.model_path.stat().st_size / 1e6


class OnnxBackend(Backend):
    name = "onnxruntime"
    layout = "NCHW"

    def __init__(self, model_path, threads=None):
        super().__init__(model_path, threads)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640

    def infer(self, tensor):
        return self.session.run(None, {self.input_name: tensor})[0]


def _load_tflite_interpreter():
    """Find whichever TFLite runtime is installed (newest name first)."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # Google's current package
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # older Raspberry Pi package
        return Interpreter
    except ImportError:
        pass
    import tensorflow as tf  # full TensorFlow as a last resort

    return tf.lite.Interpreter


class TFLiteBackend(Backend):
    name = "tflite"
    layout = "NHWC"  # classic TF models; detected from the input shape below

    def __init__(self, model_path, threads=None):
        super().__init__(model_path, threads)
        Interpreter = _load_tflite_interpreter()
        self.interp = Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        shape = self.inp["shape"]  # (1, 3, S, S) from PyTorch-based exporters, (1, S, S, 3) from TF
        if shape[1] == 3:
            self.layout, self.input_size = "NCHW", int(shape[2])
        else:
            self.layout, self.input_size = "NHWC", int(shape[1])

    def infer(self, tensor):
        x = tensor
        # Fully-integer models take int8/uint8 input: quantize with the
        # model's own scale and zero point.
        if self.inp["dtype"] in (np.int8, np.uint8):
            scale, zero = self.inp["quantization"]
            x = np.clip(np.round(tensor / scale + zero), np.iinfo(self.inp["dtype"]).min,
                        np.iinfo(self.inp["dtype"]).max).astype(self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] in (np.int8, np.uint8):
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        y = np.asarray(y, dtype=np.float32).copy()
        # Ultralytics TFLite exports give box coordinates normalised to 0..1.
        y_boxes = y[:, :4, :] if y.shape[1] < y.shape[2] else y[:, :, :4]
        if y_boxes.max() <= 2.0:
            y_boxes *= self.input_size
        return y


class TorchBackend(Backend):
    """Reference implementation using the original PyTorch weights."""

    name = "pytorch"
    layout = "NCHW"

    def __init__(self, model_path, threads=None, input_size: int = 640):
        super().__init__(model_path, threads)
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(self.threads)
        self.torch = torch
        self.model = YOLO(str(self.model_path)).model.eval()
        self.input_size = input_size

    def infer(self, tensor):
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(tensor))
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out.cpu().numpy()


def load_backend(model_path: str | Path, threads: int | None = None) -> Backend:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return OnnxBackend(model_path, threads)
    if suffix == ".tflite":
        return TFLiteBackend(model_path, threads)
    if suffix == ".pt":
        return TorchBackend(model_path, threads)
    raise ValueError(f"unsupported model format: {suffix}")
