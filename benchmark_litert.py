#!/usr/bin/env python3
"""Host-side benchmark of an exported .tflite artifact through litert-torch."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from baybayin_ocr.inference import load_checkpoint
from baybayin_ocr.preprocessing import prepare_image
from baybayin_ocr.decoding import greedy_decode_with_confidence
import cv2


def as_numpy(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, torch.Tensor):
        return output.detach().cpu().numpy()
    return np.asarray(output)


def stats(times_ms):
    avg = statistics.fmean(times_ms)
    med = statistics.median(times_ms)
    p95 = float(np.percentile(np.asarray(times_ms, dtype=np.float64), 95))
    return {"iterations": len(times_ms), "mean_ms": avg, "median_ms": med, "p95_ms": p95,
            "min_ms": min(times_ms), "max_ms": max(times_ms),
            "mean_fps": 1000.0 / avg, "p95_equivalent_fps": 1000.0 / p95}


def main():
    parser = argparse.ArgumentParser(description="Benchmark .tflite on the host LiteRT runtime; phone benchmarks are still required.")
    parser.add_argument("--model", default="checkpoints/baybayin_mobile.tflite")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Checkpoint supplies vocabulary and preprocessing dimensions")
    parser.add_argument("--image", required=True, help="Representative single-line ROI")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", default="runs/litert_benchmark.json")
    args = parser.parse_args()

    try:
        import litert_torch
    except ImportError as exc:
        raise SystemExit("Install the LiteRT converter/runtime dependencies from requirements.txt") from exc

    exported = Path(args.model)
    if not exported.is_file():
        raise FileNotFoundError(exported)
    model, vocabulary, config, _ = load_checkpoint(args.checkpoint, device="cpu")
    del model
    image = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(args.image)
    tensor, input_length = prepare_image(image, config["image_height"], config["image_width"])
    edge_model = litert_torch.load(str(exported))

    def infer():
        output = as_numpy(edge_model(tensor))
        logits = torch.from_numpy(output[0, :input_length])
        return greedy_decode_with_confidence(logits, vocabulary)

    for _ in range(args.warmup):
        infer()
    model_times = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        edge_model(tensor)
        model_times.append((time.perf_counter() - started) * 1000.0)
    pipeline_times = []
    last_text, last_confidence = "", 0.0
    for _ in range(args.iterations):
        started = time.perf_counter()
        last_text, last_confidence = infer()
        pipeline_times.append((time.perf_counter() - started) * 1000.0)
    report = {
        "runtime": "litert_torch host runtime",
        "model": str(exported),
        "model_size_mib": exported.stat().st_size / (1024 * 1024),
        "input_shape": list(tensor.shape),
        "valid_timesteps": input_length,
        "model_only": stats(model_times),
        "model_plus_decode": stats(pipeline_times),
        "example_prediction": last_text,
        "example_confidence": last_confidence,
        "note": "Host-side LiteRT timing is not a substitute for Android/iOS delegate benchmarks.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
