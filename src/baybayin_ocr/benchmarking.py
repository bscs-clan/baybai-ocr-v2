"""Batch-1 latency benchmark for the OCR model and image-to-text pipeline."""
from pathlib import Path
import statistics
import time
from typing import Dict, Iterable, List, Optional
import cv2
import numpy as np
import torch

from .inference import load_checkpoint, predict_array
from .preprocessing import prepare_image


def _summarize(times_ms: List[float]) -> dict:
    if not times_ms:
        raise ValueError("No timing samples collected.")
    mean_ms = statistics.fmean(times_ms)
    median_ms = statistics.median(times_ms)
    p95_ms = float(np.percentile(np.asarray(times_ms, dtype=np.float64), 95))
    return {
        "iterations": len(times_ms),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "mean_fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        "p95_equivalent_fps": 1000.0 / p95_ms if p95_ms > 0 else float("inf"),
    }


def benchmark_checkpoint(checkpoint_path: str | Path, image_path: str | Path,
                         device: str = "cpu", warmup: int = 20, iterations: int = 100,
                         threads: Optional[int] = 1) -> dict:
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be >= 1 and warmup >= 0")
    if device == "cpu" and threads:
        torch.set_num_threads(int(threads))
    model, vocab, input_config, checkpoint = load_checkpoint(checkpoint_path, device=device)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    tensor, input_length = prepare_image(image, input_config["image_height"], input_config["image_width"])
    tensor = tensor.to(device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        model_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(tensor)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            model_times.append((time.perf_counter() - start) * 1000.0)

        for _ in range(min(warmup, 5)):
            predict_array(model, vocab, image, input_config, device=device)
        pipeline_times = []
        last_text, last_confidence = "", 0.0
        for _ in range(iterations):
            start = time.perf_counter()
            last_text, last_confidence = predict_array(model, vocab, image, input_config, device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            pipeline_times.append((time.perf_counter() - start) * 1000.0)

    return {
        "device": device,
        "cpu_threads": torch.get_num_threads() if device == "cpu" else None,
        "image_path": str(image_path),
        "input_shape": list(tensor.shape),
        "valid_timesteps": int(input_length),
        "parameters": model.count_parameters(),
        "model_only": _summarize(model_times),
        "preprocess_model_decode": _summarize(pipeline_times),
        "example_prediction": last_text,
        "example_confidence": last_confidence,
    }


def format_benchmark_report(report: dict) -> str:
    def section(title, stats):
        return (
            f"{title}\n"
            f"  Mean latency: {stats['mean_ms']:.2f} ms  | estimated serial FPS: {stats['mean_fps']:.1f}\n"
            f"  Median: {stats['median_ms']:.2f} ms | p95: {stats['p95_ms']:.2f} ms "
            f"(p95-equivalent FPS: {stats['p95_equivalent_fps']:.1f})\n"
            f"  Min/max: {stats['min_ms']:.2f}/{stats['max_ms']:.2f} ms"
        )
    return "\n".join([
        f"Device: {report['device']} (CPU threads: {report['cpu_threads']})",
        f"Input: {report['input_shape']} | parameters: {report['parameters']:,} | valid time steps: {report['valid_timesteps']}",
        section("MODEL ONLY (preprocessed tensor already in memory)", report["model_only"]),
        section("PREPROCESS + MODEL + CTC GREEDY DECODE (camera image already in memory)", report["preprocess_model_decode"]),
        f"Sample prediction: {report['example_prediction']!r} (token confidence {report['example_confidence']:.3f})",
        "NOTE: This excludes camera capture, ROI/text detection, rendering, and the actual phone runtime.",
    ])
