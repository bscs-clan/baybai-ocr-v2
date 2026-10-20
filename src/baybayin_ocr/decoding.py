from typing import Iterable, List, Sequence, Tuple
import torch
from .vocabulary import BaybayinVocabulary


def greedy_decode_with_confidence(logits: torch.Tensor, vocabulary: BaybayinVocabulary) -> Tuple[str, float]:
    """Decode one [T, C] tensor and return text plus mean emitted-token probability."""
    probs = torch.softmax(logits.detach().float(), dim=-1)
    max_probs, best = probs.max(dim=-1)
    output: List[str] = []
    emitted_scores: List[float] = []
    previous = -1
    for token, score in zip(best.tolist(), max_probs.tolist()):
        if token != previous and token != 0:
            output.append(vocabulary.idx_to_char[int(token)])
            emitted_scores.append(float(score))
        previous = int(token)
    confidence = sum(emitted_scores) / len(emitted_scores) if emitted_scores else 0.0
    return "".join(output), confidence


def greedy_decode_batch(logits: torch.Tensor, input_lengths: Sequence[int],
                        vocabulary: BaybayinVocabulary) -> List[str]:
    results = []
    for i in range(logits.shape[0]):
        length = int(input_lengths[i])
        text, _ = greedy_decode_with_confidence(logits[i, :length], vocabulary)
        results.append(text)
    return results
