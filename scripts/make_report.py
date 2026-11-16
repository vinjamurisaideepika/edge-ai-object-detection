"""Merge results/benchmark.csv + results/accuracy.csv into
results/REPORT.md (a table to paste into the README) and
results/tradeoff.png (latency vs accuracy, marker size = model size).

    python scripts/make_report.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(name: str) -> dict[str, dict]:
    with (RESULTS / name).open() as fh:
        return {row["model"]: row for row in csv.DictReader(fh)}


def main() -> None:
    bench, acc = read("benchmark.csv"), read("accuracy.csv")
    rows = []
    for model, b in bench.items():
        a = acc.get(model, {})
        rows.append({
            "model": model, "runtime": b["backend"], "size_mb": float(b["size_mb"]),
            "input": int(b["input"]), "infer_ms": float(b["infer_mean_ms"]),
            "p95_ms": float(b["infer_p95_ms"]), "fps": float(b["fps"]),
            "map50_95": float(a["mAP50_95"]) if a else None,
            "env": f"{b['machine']}, {b['cpu_quota']}",
        })
    rows.sort(key=lambda r: r["infer_ms"])

    lines = [
        f"Environment: {rows[0]['env']}\n",
        "| Model | Runtime | Size (MB) | Input | Inference (ms) | p95 (ms) | End-to-end FPS | mAP50-95 (COCO128) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        m = f"{r['map50_95']:.3f}" if r["map50_95"] is not None else "-"
        lines.append(f"| {r['model']} | {r['runtime']} | {r['size_mb']:.1f} | {r['input']} | "
                     f"{r['infer_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps']:.1f} | {m} |")
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip install matplotlib for the chart)")
        return

    pts = [r for r in rows if r["map50_95"] is not None]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in pts:
        ax.scatter(r["infer_ms"], r["map50_95"], s=r["size_mb"] * 40, alpha=0.6)
        ax.annotate(r["model"].replace("yolo11n_", ""), (r["infer_ms"], r["map50_95"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Inference latency (ms, lower is better)")
    ax.set_ylabel("mAP50-95 on COCO128 (higher is better)")
    ax.set_title(f"YOLO11n edge trade-offs - {rows[0]['env']}\n(marker size = model size)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "tradeoff.png", dpi=150)
    print(f"chart: {RESULTS / 'tradeoff.png'}")


if __name__ == "__main__":
    main()
