"""Draw detections and a small performance overlay on a frame."""

from __future__ import annotations

import cv2
import numpy as np

from .labels import class_name
from .postprocess import Detection


def _color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id)  # stable colour per class
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def draw(frame: np.ndarray, dets: list[Detection], fps: float | None = None) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        c = _color(d.class_id)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, c, 2)
        label = f"{class_name(d.class_id)} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), c, -1)
        cv2.putText(out, label, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} FPS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out
