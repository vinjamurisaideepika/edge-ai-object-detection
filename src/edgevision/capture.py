"""Image acquisition: webcam, video file, single image or a folder of images.

Every source is exposed the same way: iterate over it to get BGR frames
(numpy arrays of shape HxWx3, uint8), exactly what OpenCV produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FrameSource:
    """Open `source` and yield frames.

    source can be:
      * "0", "1", ...   -> webcam index (works natively on macOS / a Pi,
                           not inside Docker on a Mac)
      * path/video.mp4  -> video file
      * path/image.jpg  -> single image
      * path/folder/    -> every image in the folder, sorted by name
    """

    def __init__(self, source: str | int, loop_images: int = 1):
        self.source = str(source)
        self.loop_images = max(1, loop_images)  # repeat stills (useful for benchmarks)

    def _is_camera(self) -> bool:
        return self.source.isdigit()

    def __iter__(self) -> Iterator[np.ndarray]:
        path = Path(self.source)
        if self._is_camera() or (path.is_file() and path.suffix.lower() not in IMAGE_SUFFIXES):
            yield from self._iter_video()
        elif path.is_file():
            yield from self._iter_images([path])
        elif path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if not files:
                raise FileNotFoundError(f"no images found in {path}")
            yield from self._iter_images(files)
        else:
            raise FileNotFoundError(f"source not found: {self.source}")

    def _iter_video(self) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(int(self.source) if self._is_camera() else self.source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source {self.source}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    def _iter_images(self, files: list[Path]) -> Iterator[np.ndarray]:
        for _ in range(self.loop_images):
            for f in files:
                img = cv2.imread(str(f))
                if img is None:
                    raise RuntimeError(f"could not read image {f}")
                yield img
