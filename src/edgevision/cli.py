"""Command-line interface.

  python -m edgevision detect    --model models/yolo11n.onnx --source samples/
  python -m edgevision benchmark --models models/*.onnx models/*.tflite --source samples/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2

from .backends import load_backend
from .capture import FrameSource
from .labels import class_name
from .pipeline import DetectionPipeline
from .visualize import draw


def cmd_detect(args: argparse.Namespace) -> None:
    backend = load_backend(args.model, args.threads)
    pipe = DetectionPipeline(backend, args.conf, args.iou)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} backend={backend.name} input={backend.input_size} threads={backend.threads}")

    writer = None
    t_start = time.perf_counter()
    for i, frame in enumerate(FrameSource(args.source)):
        dets = pipe(frame)
        fps = 1000.0 / pipe.times.total_ms()[-1]
        names = ", ".join(f"{class_name(d.class_id)} {d.score:.2f}" for d in dets[:6])
        print(f"frame {i:4d}: {len(dets):2d} objects | {pipe.times.total_ms()[-1]:6.1f} ms | {names}")

        vis = draw(frame, dets, fps)
        if args.show:
            cv2.imshow("edgevision", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.save_video:
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "output.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            writer.write(vis)
        else:
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), vis)
        if args.max_frames and i + 1 >= args.max_frames:
            break

    if writer is not None:
        writer.release()
    s = pipe.times.summary()
    print(f"\n{len(pipe.times.inference_ms)} frames in {time.perf_counter() - t_start:.1f} s")
    print(f"pre {s['pre_mean_ms']:.1f} ms | inference {s['infer_mean_ms']:.1f} ms | "
          f"post {s['post_mean_ms']:.1f} ms | total {s['total_mean_ms']:.1f} ms = {s['fps']:.1f} FPS")
    print(f"annotated output in {out_dir}/")


def environment() -> dict[str, str]:
    """Where did this benchmark run? Results mean nothing without this."""
    cpu_quota = "unlimited"
    try:  # cgroup v2 CPU limit, set by `docker run --cpus=...`
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpu_quota = f"{int(quota) / int(period):.2f} CPUs"
    except (OSError, ValueError):
        pass
    return {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
            "cpu_count": str(os.cpu_count()), "cpu_quota": cpu_quota}


def cmd_benchmark(args: argparse.Namespace) -> None:
    frames = list(FrameSource(args.source))
    env = environment()
    print("environment:", json.dumps(env))
    rows = []
    for model in args.models:
        backend = load_backend(model, args.threads)
        pipe = DetectionPipeline(backend, args.conf, args.iou)
        for f in frames[: args.warmup]:  # warm-up: caches, lazy allocations
            pipe(f)
        pipe.reset_times()
        n_dets = 0
        for i in range(args.runs):
            n_dets += len(pipe(frames[i % len(frames)]))
        s = pipe.times.summary()
        row = {"model": Path(model).name, "backend": backend.name, "size_mb": round(backend.model_size_mb, 2),
               "input": backend.input_size, "threads": backend.threads,
               **{k: round(v, 2) for k, v in s.items()}, "avg_detections": round(n_dets / args.runs, 2), **env}
        rows.append(row)
        print(f"{row['model']:<32} {row['size_mb']:>6.2f} MB  infer {row['infer_mean_ms']:7.1f} ms "
              f"(p95 {row['infer_p95_ms']:7.1f})  total {row['total_mean_ms']:7.1f} ms  {row['fps']:6.1f} FPS")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nresults written to {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="edgevision", description="Edge object detection pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", default="samples", help="webcam index, video, image or folder")
    common.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    common.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    common.add_argument("--threads", type=int, default=None, help="CPU threads for inference")

    d = sub.add_parser("detect", parents=[common], help="run detection and save annotated frames")
    d.add_argument("--model", required=True)
    d.add_argument("--save-dir", default="results/detect")
    d.add_argument("--save-video", action="store_true", help="write output.mp4 instead of JPEGs")
    d.add_argument("--show", action="store_true", help="open a preview window (not in Docker)")
    d.add_argument("--max-frames", type=int, default=0)
    d.set_defaults(func=cmd_detect)

    b = sub.add_parser("benchmark", parents=[common], help="compare models: size, latency, FPS")
    b.add_argument("--models", nargs="+", required=True)
    b.add_argument("--runs", type=int, default=50)
    b.add_argument("--warmup", type=int, default=5)
    b.add_argument("--output", default="results/benchmark.csv")
    b.set_defaults(func=cmd_benchmark)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
