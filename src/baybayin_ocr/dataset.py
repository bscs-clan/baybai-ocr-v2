from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import unicodedata
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import _to_grayscale, augment_gray, prepare_image
from .vocabulary import BaybayinVocabulary


def read_label_rows(dataset_root: str | Path) -> List[Tuple[str, str]]:
    root = Path(dataset_root)
    labels_file = root / "labels.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_file}")
    rows: List[Tuple[str, str]] = []
    with labels_file.open("r", encoding="utf-8-sig") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"{labels_file}:{line_num}: expected filename<TAB>text")
            filename, text = line.split("\t", 1)
            filename = filename.strip()
            text = unicodedata.normalize("NFC", text)
            if not filename:
                raise ValueError(f"{labels_file}:{line_num}: empty filename")
            image_path = root / "images" / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"{labels_file}:{line_num}: image not found: {image_path}")
            rows.append((filename, text))
    if not rows:
        raise ValueError(f"No labeled samples found in {labels_file}")
    return rows


class BaybayinDataset(Dataset):
    def __init__(self, dataset_root: str | Path, rows: Sequence[Tuple[str, str]],
                 vocabulary: BaybayinVocabulary, image_height: int = 48,
                 image_width: int = 320, augment: bool = False, seed: int = 42):
        self.root = Path(dataset_root)
        self.rows = list(rows)
        self.vocabulary = vocabulary
        self.image_height = image_height
        self.image_width = image_width
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> Dict[str, object]:
        filename, text = self.rows[index]
        path = self.root / "images" / filename
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"OpenCV could not read image: {path}")
        gray = _to_grayscale(image)
        if self.augment:
            # Deterministic per (seed, epoch, index) despite data loader worker ordering.
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 97)
            gray = augment_gray(gray, rng)
        tensor, input_length = prepare_image(gray, self.image_height, self.image_width)
        target = torch.tensor(self.vocabulary.encode(text), dtype=torch.long)
        return {
            "image": tensor.squeeze(0),
            "target": target,
            "target_length": len(target),
            "input_length": input_length,
            "text": text,
            "filename": filename,
        }


def collate_ocr_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    flat_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    return {
        "images": images,
        "targets": flat_targets,
        "target_lengths": torch.tensor([item["target_length"] for item in batch], dtype=torch.long),
        "input_lengths": torch.tensor([item["input_length"] for item in batch], dtype=torch.long),
        "texts": [str(item["text"]) for item in batch],
        "filenames": [str(item["filename"]) for item in batch],
    }
