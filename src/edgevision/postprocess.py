"""Post-processing: raw YOLO output -> list of detections in frame pixels.

A YOLO11/YOLOv8 detector outputs one tensor of shape (1, 4 + C, N):
for each of N = 8400 candidate boxes (at 640x640 input), the box centre,
width and height followed by one score per class (C = 80 for COCO).

Steps:
  1. keep candidates whose best class score >= conf_threshold
  2. convert (cx, cy, w, h) -> (x1, y1, x2, y2)
  3. non-maximum suppression (NMS): drop boxes that overlap a better box
     of the same class by more than iou_threshold
  4. undo the letterbox so boxes match the original frame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import LetterboxInfo


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection-over-union between one box and many boxes (x1,y1,x2,y2)."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS. Returns indices of the boxes to keep, best first."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        overlaps = iou(boxes[best], boxes[order[1:]])
        order = order[1:][overlaps <= iou_threshold]
    return keep


def decode_yolo(
    raw: np.ndarray,
    info: LetterboxInfo,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> list[Detection]:
    pred = np.asarray(raw, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:  # (84, 8400) -> (8400, 84)
        pred = pred.T

    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(pred)), class_ids]
    mask = scores >= conf_threshold
    if not mask.any():
        return []
    pred, scores, class_ids = pred[mask], scores[mask], class_ids[mask]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    # Class-aware NMS in one call: shift each class far apart so boxes of
    # different classes can never overlap.
    offset = class_ids[:, None].astype(np.float32) * 10_000.0
    keep = nms(boxes + offset, scores, iou_threshold)[:max_detections]

    dets: list[Detection] = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        # undo letterbox: remove padding, then undo scaling
        x1 = (x1 - info.pad_x) / info.scale
        x2 = (x2 - info.pad_x) / info.scale
        y1 = (y1 - info.pad_y) / info.scale
        y2 = (y2 - info.pad_y) / info.scale
        dets.append(
            Detection(
                float(np.clip(x1, 0, info.orig_w)),
                float(np.clip(y1, 0, info.orig_h)),
                float(np.clip(x2, 0, info.orig_w)),
                float(np.clip(y2, 0, info.orig_h)),
                float(scores[i]),
                int(class_ids[i]),
            )
        )
    return dets
