import json
from pathlib import Path
from typing import Iterable, List, Sequence
import unicodedata


class BaybayinVocabulary:
    """Character-level vocabulary with CTC blank fixed at class index 0.

    The project README's example lists are not sufficiently precise to hard-code
    a Unicode mapping. Build this vocabulary from the actual ground-truth labels
    and save it with every checkpoint for deterministic inference.
    """

    BLANK_TOKEN = "<CTC_BLANK>"
    BAYBAYIN_PUNCTUATION = ("᜵", "᜶")  # U+1735 and U+1736

    def __init__(self, characters: Sequence[str]):
        chars = list(characters)
        if any(len(c) != 1 for c in chars):
            raise ValueError("Each vocabulary item must be exactly one Unicode code point.")
        if len(set(chars)) != len(chars):
            raise ValueError("Duplicate characters in vocabulary.")
        if self.BLANK_TOKEN in chars:
            raise ValueError("CTC blank is reserved and must not occur in labels.")
        self.characters = list(chars)
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.idx_to_char[0] = ""

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "BaybayinVocabulary":
        # Always reserve output classes for the Philippine single and double
        # punctuation marks, even when a particular dataset does not use them.
        chars = set(cls.BAYBAYIN_PUNCTUATION)
        for text in texts:
            normalized = unicodedata.normalize("NFC", text)
            chars.update(normalized)
        chars.discard("\r")
        chars.discard("\n")
        return cls(sorted(chars, key=ord))

    @property
    def num_classes(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str) -> List[int]:
        text = unicodedata.normalize("NFC", text)
        unknown = sorted(set(text) - set(self.char_to_idx))
        if unknown:
            raise ValueError(f"Text contains characters outside saved vocabulary: {unknown!r}")
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices: Iterable[int]) -> str:
        return "".join(self.idx_to_char[int(i)] for i in indices if int(i) != 0)

    def to_dict(self) -> dict:
        return {"blank_token": self.BLANK_TOKEN, "characters": self.characters}

    @classmethod
    def from_dict(cls, value: dict) -> "BaybayinVocabulary":
        return cls(value["characters"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BaybayinVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
