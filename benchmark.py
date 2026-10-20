#!/usr/bin/env python3
"""CLI entry point for batch-1 OCR latency/FPS benchmarks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from baybayin_ocr.benchmarking import benchmark_checkpoint, format_benchmark_report


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR model-only and preprocessing+model+decode latency.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--image", required=True, help="Representative text-line ROI image")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="CPU inference threads; benchmark on target device too")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", default="runs/benchmark.json")
    args = parser.parse_args()

    report = benchmark_checkpoint(args.checkpoint, args.image, device=args.device,
                                  warmup=args.warmup, iterations=args.iterations, threads=args.threads)
    print(format_benchmark_report(report))
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {out_path}")


if __name__ == "__main__":
    main()
