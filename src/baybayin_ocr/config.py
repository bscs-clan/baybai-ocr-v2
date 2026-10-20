from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class OCRConfig:
    """Configuration shared by training, inference, benchmarking and export."""

    image_height: int = 48
    image_width: int = 320
    feature_channels: int = 128
    temporal_blocks: int = 4
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    val_split: float = 0.15
    num_workers: int = 0
    patience: int = 15
    seed: int = 42
    device: str = "auto"
    augment: bool = True
    log_every_n_batches: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "OCRConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in values.items() if k in allowed})

    def resolved_device(self) -> str:
        import torch
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        if self.image_height != 48:
            raise ValueError("The current mobile model expects image_height=48 (fixed height projection).")
        if self.image_width < 64 or self.image_width % 2 != 0:
            raise ValueError("image_width must be an even number >= 64 for a stable fixed-shape export.")
        if self.feature_channels < 16:
            raise ValueError("feature_channels must be >= 16.")
        if self.temporal_blocks < 1:
            raise ValueError("temporal_blocks must be >= 1.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0, 1).")


def project_root_from_file(file: str) -> Path:
    """Resolve repository root when called from a package module."""
    return Path(file).resolve().parents[2]
