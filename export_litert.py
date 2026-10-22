#!/usr/bin/env python3
"""Experimental direct PyTorch -> LiteRT/TFLite export (optional dependencies)."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from baybayin_ocr.inference import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export the mobile Baybayin model to .tflite using litert-torch.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="checkpoints/baybayin_mobile.tflite")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import litert_torch
    except ImportError as exc:
        raise SystemExit(
            f"Missing litert-torch in {sys.executable}. Install dependencies into this interpreter with:\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
            "LiteRT conversion is recommended on Linux with Python 3.11."
        ) from exc

    model, vocab, config, _ = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    example = (torch.zeros(1, 1, config["image_height"], config["image_width"], dtype=torch.float32),)
    with torch.inference_mode():
        reference = model(*example).cpu().numpy()
        edge_model = litert_torch.convert(model, example)
        converted = edge_model(*example)
        if isinstance(converted, (tuple, list)):
            converted = converted[0]
        if isinstance(converted, torch.Tensor):
            converted = converted.detach().cpu().numpy()
        converted = np.asarray(converted)
        if converted.shape != reference.shape:
            raise RuntimeError(f"Output shape mismatch: PyTorch {reference.shape} vs LiteRT {converted.shape}")
        abs_error = float(np.max(np.abs(reference - converted)))
        close = bool(np.allclose(reference, converted, atol=args.atol, rtol=args.rtol))
        print(f"Conversion output max absolute error: {abs_error:.8g}; allclose={close}")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        edge_model.export(str(output))
        vocabulary_path = output.with_suffix(".vocab.json")
        vocab.save(vocabulary_path)
    print(f"Exported: {output.resolve()} ({output.stat().st_size / (1024 * 1024):.2f} MiB)")
    print(f"Exported vocabulary: {vocabulary_path.resolve()}")
    print("Conversion/export success is not a mobile latency guarantee; benchmark the exported artifact on target phones.")
    if not close:
        raise SystemExit("Converted output failed numerical smoke check; do not deploy without investigating.")


if __name__ == "__main__":
    main()
