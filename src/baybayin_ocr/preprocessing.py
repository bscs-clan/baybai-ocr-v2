"""OpenCV-based preprocessing shared by train and inference."""
from typing import Tuple
import cv2
import numpy as np
import torch


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def prepare_image(image: np.ndarray, image_height: int = 48, image_width: int = 320) -> Tuple[torch.Tensor, int]:
    """Fit an image into a white fixed-size canvas while preserving aspect ratio.

    Returns normalized NCHW tensor (1, 1, H, W) and the unpadded CTC input
    length. Input is assumed to be a single text-line/ROI, not a full camera frame.
    """
    gray = _to_grayscale(image)
    orig_h, orig_w = gray.shape[:2]
    if orig_h < 1 or orig_w < 1:
        raise ValueError("Image has an empty dimension.")
    scale = min(image_height / float(orig_h), image_width / float(orig_w))
    resized_h = max(1, min(image_height, int(round(orig_h * scale))))
    resized_w = max(1, min(image_width, int(round(orig_w * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (resized_w, resized_h), interpolation=interpolation)

    canvas = np.full((image_height, image_width), 255, dtype=np.uint8)
    y0 = (image_height - resized_h) // 2
    # Left align keeps time-step calculation simple and avoids shifting leading text.
    canvas[y0:y0 + resized_h, :resized_w] = resized
    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.unsqueeze(0).unsqueeze(0).contiguous()
    # First convolution has stride 2 and padding 1 -> output width is ceil(W/2).
    input_length = (resized_w + 1) // 2
    return tensor, input_length


def augment_gray(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild camera-like augmentation; avoids changing the character labels."""
    out = image.copy()
    h, w = out.shape[:2]
    angle = float(rng.uniform(-2.5, 2.5))
    scale = float(rng.uniform(0.92, 1.08))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += float(rng.uniform(-0.02, 0.02)) * w
    matrix[1, 2] += float(rng.uniform(-0.04, 0.04)) * h
    out = cv2.warpAffine(out, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    gain = float(rng.uniform(0.85, 1.15))
    bias = float(rng.uniform(-10.0, 10.0))
    out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        sigma = float(rng.uniform(1.0, 5.0))
        noise = rng.normal(0.0, sigma, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out
