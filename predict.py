#!/usr/bin/env python3
"""Single-image OCR CLI, separate from training and benchmarking."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from baybayin_ocr.inference import load_checkpoint, predict_image


def main():
    parser = argparse.ArgumentParser(description="Recognize Baybayin text from a single text-line ROI image.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    model, vocabulary, input_config, _ = load_checkpoint(args.checkpoint, device=args.device)
    text, confidence = predict_image(model, vocabulary, args.image, input_config, device=args.device)
    print(f"Predicted text: {text}")
    print(f"Mean emitted-token probability (uncalibrated): {confidence:.4f}")


if __name__ == "__main__":
    main()
