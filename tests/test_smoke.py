import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baybayin_ocr.decoding import greedy_decode_with_confidence
from baybayin_ocr.metrics import corpus_cer, exact_match_accuracy
from baybayin_ocr.model import MobileBaybayinOCR
from baybayin_ocr.preprocessing import prepare_image
from baybayin_ocr.vocabulary import BaybayinVocabulary
import numpy as np


def test_model_shapes():
    model = MobileBaybayinOCR(num_classes=8, feature_channels=32, temporal_blocks=2).eval()
    with torch.inference_mode():
        y = model(torch.zeros(2, 1, 48, 320))
    assert tuple(y.shape) == (2, 160, 8)


def test_vocab_and_decoder():
    vocab = BaybayinVocabulary.from_texts(["ᜀᜃ ᜄ"])
    assert vocab.encode("ᜀᜃ ᜄ")
    logits = torch.full((4, vocab.num_classes), -5.0)
    char = vocab.char_to_idx["ᜀ"]
    logits[0, char] = 5
    logits[1, char] = 5
    logits[2, 0] = 5
    logits[3, char] = 5
    text, conf = greedy_decode_with_confidence(logits, vocab)
    assert text == "ᜀᜀ"
    assert conf > 0.9


def test_preprocess_shape_and_timing_length():
    x, length = prepare_image(np.full((60, 120, 3), 255, dtype=np.uint8), 48, 320)
    assert tuple(x.shape) == (1, 1, 48, 320)
    assert length == 48


def test_metrics():
    assert corpus_cer(["abc"], ["adc"]) == 1 / 3
    assert exact_match_accuracy(["a", "b"], ["a", "x"]) == 0.5


def test_config_save_every_epoch():
    from baybayin_ocr.config import OCRConfig
    cfg = OCRConfig()
    assert cfg.save_every_epoch is True
    d = cfg.to_dict()
    assert d["save_every_epoch"] is True
    restored = OCRConfig.from_dict(d)
    assert restored.save_every_epoch is True


def test_training_loss_plot_per_epoch(tmp_path):
    from baybayin_ocr.live_plot import _build_plots
    import matplotlib.pyplot as plt

    sample_records = [
        {"kind": "batch", "epoch": 1, "batch": 1, "train_loss": 3.5, "global_step": 1},
        {"kind": "batch", "epoch": 1, "batch": 2, "train_loss": 3.0, "global_step": 2},
        {"kind": "epoch", "epoch": 1, "train_loss": 3.25, "val_loss": 2.8, "val_cer": 0.5, "word_accuracy": 0.4},
        {"kind": "batch", "epoch": 2, "batch": 1, "train_loss": 2.0, "global_step": 3},
        {"kind": "epoch", "epoch": 2, "train_loss": 2.1, "val_loss": 1.9, "val_cer": 0.3, "word_accuracy": 0.6},
    ]

    figs = _build_plots(sample_records, tmp_path, interactive=True)
    fig_loss = figs[0][0]
    ax = fig_loss.axes[0]
    assert ax.get_xlabel() == "Epoch"
    assert ax.get_title() == "Training loss"
    line = ax.lines[0]
    x_data, y_data = line.get_data()
    # Ensure x_data is the epoch numbers [1, 2], not the batch steps [1, 2, 3]
    assert list(x_data) == [1, 2]
    assert list(y_data) == [3.25, 2.1]
    for fig, _ in figs:
        plt.close(fig)


def test_train_model_saves_every_epoch(tmp_path):
    import cv2
    from baybayin_ocr.config import OCRConfig
    from baybayin_ocr.training import train_model

    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    for i in range(1, 5):
        img = np.full((48, 100, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i}.png"), img)

    labels_file = dataset_dir / "labels.txt"
    labels_file.write_text("img_1.png\tᜀ\nimg_2.png\tᜃ\nimg_3.png\tᜀ\nimg_4.png\tᜃ\n", encoding="utf-8")

    out_dir = tmp_path / "checkpoints"
    metrics_file = out_dir / "metrics.jsonl"
    cfg = OCRConfig(
        image_height=48,
        image_width=64,
        feature_channels=16,
        temporal_blocks=1,
        batch_size=2,
        epochs=2,
        val_split=0.5,
        save_every_epoch=True,
        device="cpu",
    )
    result = train_model(cfg, dataset_root=dataset_dir, output_dir=out_dir, metrics_jsonl=metrics_file)
    assert (out_dir / "epoch_1.pt").is_file()
    assert (out_dir / "epoch_2.pt").is_file()
    assert (out_dir / "last_model.pt").is_file()
    assert (out_dir / "best_model.pt").is_file()

    cp1 = torch.load(out_dir / "epoch_1.pt", weights_only=False)
    cp2 = torch.load(out_dir / "epoch_2.pt", weights_only=False)
    assert cp1["epoch"] == 1
    assert cp2["epoch"] == 2


def test_stop_and_resume_training(tmp_path):
    import cv2
    from baybayin_ocr.config import OCRConfig
    from baybayin_ocr.training import train_model

    dataset_dir = tmp_path / "dataset"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    for i in range(1, 5):
        img = np.full((48, 100, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"img_{i}.png"), img)

    labels_file = dataset_dir / "labels.txt"
    labels_file.write_text("img_1.png\tᜀ\nimg_2.png\tᜃ\nimg_3.png\tᜀ\nimg_4.png\tᜃ\n", encoding="utf-8")

    out_dir = tmp_path / "checkpoints"
    out_dir.mkdir(parents=True)
    # Write stop signal so training stops after epoch 1
    stop_file = out_dir / "training.stop"

    cfg = OCRConfig(
        image_height=48,
        image_width=64,
        feature_channels=16,
        temporal_blocks=1,
        batch_size=2,
        epochs=3,
        val_split=0.5,
        save_every_epoch=True,
        device="cpu",
    )

    # Pre-write stop file to stop promptly
    stop_file.write_text("stop\n", encoding="utf-8")
    result = train_model(cfg, dataset_root=dataset_dir, output_dir=out_dir)
    assert result["stopped_early"] is True
    assert (out_dir / "last_model.pt").is_file()
    assert not stop_file.exists()  # Ensure stop file was cleaned up

    # Now resume training from the saved checkpoint to complete up to epoch 2
    cfg.epochs = 2
    resume_result = train_model(cfg, dataset_root=dataset_dir, output_dir=out_dir, resume=out_dir / "last_model.pt")
    assert (out_dir / "epoch_2.pt").is_file()
    cp = torch.load(out_dir / "last_model.pt", weights_only=False)
    assert cp["epoch"] == 2


def test_find_resume_checkpoint(tmp_path):
    import pytest
    from baybayin_ocr.training import find_resume_checkpoint

    out_dir = tmp_path / "ckpt"
    out_dir.mkdir(parents=True)

    # When None, returns None
    assert find_resume_checkpoint(out_dir, None) is None

    # Error when missing
    with pytest.raises(FileNotFoundError):
        find_resume_checkpoint(out_dir, "auto")

    # Finds last_model.pt
    (out_dir / "last_model.pt").write_text("dummy", encoding="utf-8")
    assert find_resume_checkpoint(out_dir, "auto") == out_dir / "last_model.pt"

    # Finds explicit file
    explicit = out_dir / "my_custom.pt"
    explicit.write_text("dummy", encoding="utf-8")
    assert find_resume_checkpoint(out_dir, str(explicit)) == explicit



