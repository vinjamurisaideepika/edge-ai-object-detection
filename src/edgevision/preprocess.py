"""Preprocessing: turn a camera frame into the tensor the network expects.

YOLO models want a square RGB image (e.g. 640x640), float values in 0..1.
Simply stretching the frame would distort objects, so we "letterbox":
resize keeping the aspect ratio, then pad the borders with grey.
We remember the scale and padding so boxes can be mapped back later.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float  # resize factor applied to the original frame
    pad_x: float  # left padding in pixels (in network input space)
    pad_y: float  # top padding
    orig_w: int
    orig_h: int


def letterbox(frame: np.ndarray, size: int, color: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize `frame` into a size x size canvas without distortion."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, LetterboxInfo(scale, left, top, w, h)


def preprocess(frame_bgr: np.ndarray, size: int, layout: str = "NCHW") -> tuple[np.ndarray, LetterboxInfo]:
    """BGR uint8 frame -> float32 tensor of shape (1,3,S,S) or (1,S,S,3)."""
    img, info = letterbox(frame_bgr, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if layout == "NCHW":  # PyTorch / ONNX
        img = img.transpose(2, 0, 1)
    elif layout != "NHWC":  # TensorFlow Lite
        raise ValueError(f"unknown layout {layout}")
    return np.ascontiguousarray(img[None]), info
