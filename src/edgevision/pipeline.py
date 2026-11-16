"""The end-to-end pipeline, with a timer around every stage.

Timing each stage separately matters on edge devices: often the
"slow model" is actually slow preprocessing or a slow camera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backends import Backend
from .postprocess import Detection, decode_yolo
from .preprocess import preprocess


@dataclass
class StageTimes:
    preprocess_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    postprocess_ms: list[float] = field(default_factory=list)

    def total_ms(self) -> np.ndarray:
        return np.array(self.preprocess_ms) + np.array(self.inference_ms) + np.array(self.postprocess_ms)

    def summary(self) -> dict[str, float]:
        def stats(prefix: str, values) -> dict[str, float]:
            v = np.asarray(values, dtype=np.float64)
            return {f"{prefix}_mean_ms": float(v.mean()), f"{prefix}_p50_ms": float(np.percentile(v, 50)),
                    f"{prefix}_p95_ms": float(np.percentile(v, 95))}

        total = self.total_ms()
        out = {}
        out.update(stats("pre", self.preprocess_ms))
        out.update(stats("infer", self.inference_ms))
        out.update(stats("post", self.postprocess_ms))
        out.update(stats("total", total))
        out["fps"] = float(1000.0 / total.mean())
        return out


class DetectionPipeline:
    def __init__(self, backend: Backend, conf: float = 0.25, iou: float = 0.45):
        self.backend = backend
        self.conf = conf
        self.iou = iou
        self.times = StageTimes()

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        tensor, info = preprocess(frame, self.backend.input_size, self.backend.layout)
        t1 = time.perf_counter()
        raw = self.backend.infer(tensor)
        t2 = time.perf_counter()
        dets = decode_yolo(raw, info, self.conf, self.iou)
        t3 = time.perf_counter()

        self.times.preprocess_ms.append((t1 - t0) * 1e3)
        self.times.inference_ms.append((t2 - t1) * 1e3)
        self.times.postprocess_ms.append((t3 - t2) * 1e3)
        return dets

    def reset_times(self) -> None:
        self.times = StageTimes()
