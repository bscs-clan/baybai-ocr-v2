"""Checkpoint loading and single-ROI OCR inference."""
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import torch

from .config import OCRConfig
from .decoding import greedy_decode_with_confidence
from .model import MobileBaybayinOCR
from .preprocessing import prepare_image
from .vocabulary import BaybayinVocabulary


def _safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
    required = {"model_state_dict", "model_config", "vocabulary"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    config = OCRConfig.from_dict(checkpoint.get("training_config", {}))
    model_config = checkpoint["model_config"]
    height = int(model_config.get("image_height", config.image_height))
    width = int(model_config.get("image_width", config.image_width))
    vocabulary = BaybayinVocabulary.from_dict(checkpoint["vocabulary"])
    model = MobileBaybayinOCR(
        num_classes=vocabulary.num_classes,
        feature_channels=int(model_config.get("feature_channels", config.feature_channels)),
        temporal_blocks=int(model_config.get("temporal_blocks", config.temporal_blocks)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, vocabulary, {"image_height": height, "image_width": width, **model_config}, checkpoint


def predict_array(model: MobileBaybayinOCR, vocabulary: BaybayinVocabulary,
                  image, input_config: Dict[str, int], device: str = "cpu") -> Tuple[str, float]:
    """Predict on one already-captured camera frame/crop; no camera I/O is included."""
    tensor, input_length = prepare_image(image, input_config["image_height"], input_config["image_width"])
    tensor = tensor.to(device)
    with torch.inference_mode():
        logits = model(tensor)[0, :input_length]
    return greedy_decode_with_confidence(logits, vocabulary)


def predict_image(model, vocabulary, image_path: str | Path,
                  input_config: Dict[str, int], device: str = "cpu") -> Tuple[str, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return predict_array(model, vocabulary, image, input_config, device)
