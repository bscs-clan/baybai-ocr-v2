#!/usr/bin/env python3
"""CLI entry point for Baybayin OCR training. Training mechanics live in src/baybayin_ocr/training.py."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from baybayin_ocr.config import OCRConfig
from baybayin_ocr.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train mobile-oriented Baybayin CTC OCR.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root containing images/ and labels.txt")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-width", type=int, default=320, help="Fixed model input width; keep fixed for export")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience; 0 disables it")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-channels", type=int, default=128)
    parser.add_argument("--temporal-blocks", type=int, default=4)
    parser.add_argument("--log-every-n-batches", type=int, default=10)
    parser.add_argument("--metrics-jsonl", default=None, help="Append machine-readable training metrics as JSON Lines")
    parser.add_argument("--no-live-plot", action="store_true", help="Disable the separate live graph window")
    parser.add_argument("--resume", default=None, help="Resume from a full training checkpoint")
    parser.add_argument("--no-augment", action="store_true", help="Disable mild camera-like training augmentation")
    return parser.parse_args()


def main():
    args = parse_args()
    config = OCRConfig(
        image_height=48,
        image_width=args.image_width,
        feature_channels=args.feature_channels,
        temporal_blocks=args.temporal_blocks,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        augment=not args.no_augment,
        log_every_n_batches=args.log_every_n_batches,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_jsonl).resolve() if args.metrics_jsonl else output_dir / "training_metrics.jsonl"
    if not args.metrics_jsonl:
        metrics_path.write_text("", encoding="utf-8")
    done_path = output_dir / "training.done"
    try:
        done_path.unlink()
    except FileNotFoundError:
        pass

    plot_process = None
    if not args.no_live_plot:
        env = os.environ.copy()
        source_path = str(ROOT / "src")
        env["PYTHONPATH"] = source_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        plot_process = subprocess.Popen([
            sys.executable, "-m", "baybayin_ocr.live_plot",
            "--metrics", str(metrics_path), "--done", str(done_path),
            "--output-dir", str(output_dir),
        ], cwd=ROOT, env=env)
        print("Live training graph window starting; final PNG graphs will be saved in", output_dir, flush=True)

    try:
        result = train_model(config, args.dataset, output_dir, metrics_path, args.resume)
    finally:
        done_path.write_text("done\n", encoding="utf-8")
    print("Training result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
