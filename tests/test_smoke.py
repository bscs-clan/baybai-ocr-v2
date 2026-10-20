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
    assert tuple(y.shape) == (2, 128, 8)


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
