"""Unit tests for the pre/post-processing stages (no model file needed).

Run:  pytest -q
"""

import numpy as np
import pytest

from edgevision.postprocess import decode_yolo, iou, nms
from edgevision.preprocess import LetterboxInfo, letterbox, preprocess


def test_letterbox_keeps_aspect_ratio_and_pads():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)  # landscape 4:3
    out, info = letterbox(frame, 640)
    assert out.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.0)
    assert info.pad_x == 0 and info.pad_y == 80          # 80 px grey bars top and bottom
    assert (out[0, 0] == 114).all() and (out[100, 100] == 0).all()


def test_preprocess_layouts_and_range():
    frame = np.full((100, 200, 3), 255, dtype=np.uint8)
    nchw, _ = preprocess(frame, 320, "NCHW")
    nhwc, _ = preprocess(frame, 320, "NHWC")
    assert nchw.shape == (1, 3, 320, 320) and nhwc.shape == (1, 320, 320, 3)
    assert nchw.dtype == np.float32 and 0.0 <= nchw.min() and nchw.max() <= 1.0


def test_iou():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    np.testing.assert_allclose(iou(a, boxes), [1.0, 1 / 3, 0.0], atol=1e-6)


def test_nms_removes_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def _fake_yolo_output(boxes_cxcywh, class_ids, scores, num_classes=80, n=100):
    """Build a (1, 4+C, N) tensor like YOLO11 produces."""
    raw = np.zeros((1, 4 + num_classes, n), dtype=np.float32)
    for i, (b, c, s) in enumerate(zip(boxes_cxcywh, class_ids, scores)):
        raw[0, :4, i] = b
        raw[0, 4 + c, i] = s
    return raw


def test_decode_maps_boxes_back_to_original_frame():
    # Original frame 1280x960 -> letterboxed into 640: scale 0.5, pad_y 80
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=80, orig_w=1280, orig_h=960)
    raw = _fake_yolo_output([[320, 320, 100, 50]], [0], [0.9])
    (d,) = decode_yolo(raw, info)
    assert d.class_id == 0 and d.score == pytest.approx(0.9)
    # x: (320 +/- 50) / 0.5 ; y: (320 -/+ 25 - 80) / 0.5
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((540, 430, 740, 530))


def test_decode_threshold_and_class_aware_nms():
    info = LetterboxInfo(1.0, 0, 0, 640, 640)
    raw = _fake_yolo_output(
        [[100, 100, 50, 50], [102, 102, 50, 50], [100, 100, 50, 50], [400, 400, 20, 20]],
        [0, 0, 16, 2],          # person, person (duplicate), dog (same place), car
        [0.9, 0.8, 0.7, 0.1],   # the car is below the 0.25 threshold
    )
    dets = decode_yolo(raw, info, conf_threshold=0.25, iou_threshold=0.45)
    assert sorted(d.class_id for d in dets) == [0, 16]  # duplicate person removed, dog kept
