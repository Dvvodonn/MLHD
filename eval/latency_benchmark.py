#!/usr/bin/env python
"""
Benchmark YOLOv5s vs MLHD inference latency on training_5 validation images.

Hardcoded inputs:
- Images: data/processed/training_5/images/val
- MLHD checkpoint: checkpoints/best.pt (13x13 backbone)
- YOLOv5 weights: yolov5s.pt

Outputs are written to outputs/latency_benchmark:
- latency_metrics.json summarizing per-model stats
- latency_boxplot.png with per-image latency distribution
- latency_histogram.png with overlapping histograms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data/processed/training_5/images/val"
MLHD_CKPT = ROOT / "checkpoints" / "best.pt"
YOLO_WEIGHTS = ROOT / "yolov5s.pt"
OUTPUT_DIR = ROOT / "outputs" / "latency_benchmark"
MLHD_BACKBONE = "13x13"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

sys.path.insert(0, str(ROOT))

from datasets.transforms import letterbox_image  # noqa: E402
from models.backbone import backbone_output_grid, backbone_input_resolution  # noqa: E402
from models.yolo_like import Model  # noqa: E402
from utils.batch_compare_predictions import _load_yolov5 as load_yolov5  # noqa: E402
from utils.device import describe_device, get_best_device  # noqa: E402


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _load_mlhd_model(device: torch.device) -> tuple[Model, int, int]:
    grid = backbone_output_grid(MLHD_BACKBONE)
    img_size = backbone_input_resolution(MLHD_BACKBONE)

    model = Model(S=grid, backbone_name=MLHD_BACKBONE)
    ckpt = torch.load(str(MLHD_CKPT), map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, grid, img_size


def _mlhd_forward(model: Model, device: torch.device, img_size: int, img_path: Path) -> bool:
    img_tensor, _ = letterbox_image(str(img_path), target_size=(img_size, img_size))
    batch = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(batch)
    return True


def _yolo_forward(model, img_path: Path) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    with torch.no_grad():
        _ = model(img[..., ::-1])
    return True


def _prepare_images(max_images: int | None) -> List[Path]:
    images = sorted(
        p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found under {IMAGES_DIR}")
    if max_images:
        images = images[:max_images]
    return images


def _benchmark_mlhd(model: Model, device: torch.device, img_size: int, images: Sequence[Path]) -> List[float]:
    latencies = []
    for img_path in images:
        start = perf_counter()
        _mlhd_forward(model, device, img_size, img_path)
        _sync_device(device)
        latencies.append((perf_counter() - start) * 1000.0)
    return latencies


def _benchmark_yolo(model, device: torch.device, images: Sequence[Path]) -> List[float]:
    latencies = []
    for img_path in images:
        start = perf_counter()
        ok = _yolo_forward(model, img_path)
        if not ok:
            continue
        _sync_device(device)
        latencies.append((perf_counter() - start) * 1000.0)
    return latencies


def _warmup(name: str, run_fn, images: Sequence[Path], warmup_runs: int, device: torch.device) -> None:
    if warmup_runs <= 0:
        return
    for idx in range(min(warmup_runs, len(images))):
        run_fn(images[idx])
    _sync_device(device)
    print(f"Warmup finished for {name}")


def _summarize(latencies: Sequence[float]) -> Dict[str, float]:
    arr = np.array(latencies, dtype=np.float64)
    total_time_s = float(arr.sum() / 1000.0) if arr.size else 0.0
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "median_ms": float(np.median(arr)) if arr.size else 0.0,
        "p90_ms": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "min_ms": float(arr.min()) if arr.size else 0.0,
        "max_ms": float(arr.max()) if arr.size else 0.0,
        "fps_from_mean": float(1000.0 / arr.mean()) if arr.size and arr.mean() > 0 else 0.0,
        "total_time_s": total_time_s,
    }


def _print_summary(metrics: Dict[str, float], label: str) -> None:
    print(f"\n{label} latency (ms unless noted):")
    print(f"  count:       {metrics['count']}")
    print(f"  mean:        {metrics['mean_ms']:.2f}")
    print(f"  median:      {metrics['median_ms']:.2f}")
    print(f"  p90:         {metrics['p90_ms']:.2f}")
    print(f"  p95:         {metrics['p95_ms']:.2f}")
    print(f"  min / max:   {metrics['min_ms']:.2f} / {metrics['max_ms']:.2f}")
    print(f"  fps (mean):  {metrics['fps_from_mean']:.2f}")
    print(f"  total time:  {metrics['total_time_s']:.2f}s")


def _plot_box(latencies: Dict[str, List[float]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    data = [latencies["MLHD"], latencies["YOLOv5s"]]
    plt.boxplot(
        data,
        labels=["MLHD (best.pt)", "YOLOv5s"],
        showmeans=True,
        meanline=True,
    )
    plt.ylabel("Per-image latency (ms)")
    plt.title("Inference latency comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_hist(latencies: Dict[str, List[float]], out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    bins = 30
    for label, values in latencies.items():
        plt.hist(values, bins=bins, alpha=0.55, label=label, density=True)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Density")
    plt.title("Latency distribution on training_5/val")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description="Latency benchmark for MLHD vs YOLOv5s on training_5/val.")
    ap.add_argument("--max-images", type=int, default=None, help="Optional limit for quick runs.")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup inferences per model.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = get_best_device()
    print(f"Using device: {describe_device(device)}")

    images = _prepare_images(args.max_images)
    print(f"Benchmarking on {len(images)} images from {IMAGES_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MLHD checkpoint...")
    mlhd_model, grid, img_size = _load_mlhd_model(device)
    print(f"MLHD grid {grid} | input {img_size}x{img_size}")

    print("Loading YOLOv5s weights...")
    yolo_model = load_yolov5(str(YOLO_WEIGHTS), device)

    if args.warmup:
        _warmup("MLHD", lambda p: _mlhd_forward(mlhd_model, device, img_size, p), images, args.warmup, device)
        _warmup("YOLOv5s", lambda p: _yolo_forward(yolo_model, p), images, args.warmup, device)

    print("Running MLHD latency benchmark...")
    mlhd_lat = _benchmark_mlhd(mlhd_model, device, img_size, images)
    print("Running YOLOv5s latency benchmark...")
    yolo_lat = _benchmark_yolo(yolo_model, device, images)

    metrics = {
        "MLHD": _summarize(mlhd_lat),
        "YOLOv5s": _summarize(yolo_lat),
        "config": {
            "images_dir": str(IMAGES_DIR),
            "mlhd_checkpoint": str(MLHD_CKPT),
            "yolo_weights": str(YOLO_WEIGHTS),
            "device": describe_device(device),
            "num_images": len(images),
            "warmup_runs": args.warmup,
        },
    }

    _print_summary(metrics["MLHD"], "MLHD (best.pt)")
    _print_summary(metrics["YOLOv5s"], "YOLOv5s")

    metrics_path = OUTPUT_DIR / "latency_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    plot_box = OUTPUT_DIR / "latency_boxplot.png"
    plot_hist = OUTPUT_DIR / "latency_histogram.png"
    _plot_box({"MLHD": mlhd_lat, "YOLOv5s": yolo_lat}, plot_box)
    _plot_hist({"MLHD": mlhd_lat, "YOLOv5s": yolo_lat}, plot_hist)
    print(f"Saved plots:\n- {plot_box}\n- {plot_hist}")

    print("Done.")


if __name__ == "__main__":
    main()
