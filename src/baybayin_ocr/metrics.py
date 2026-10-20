from typing import Iterable, Sequence
import unicodedata


INSERTION_LABEL = "<inserted>"
DELETION_LABEL = "<deleted>"

# Baybayin Unicode letters in Filipino Abakada order. E/O have no separate
# traditional Baybayin letters; the corresponding I/U forms are used.
ABAKADA_LETTERS = (
    "A", "BA", "KA", "DA", "GA", "HA", "I", "LA", "MA", "NA", "NGA",
    "PA", "RA", "SA", "TA", "U", "WA", "YA",
)
ABAKADA_RANK = {name: rank for rank, name in enumerate(ABAKADA_LETTERS)}
MARK_RANK = {
    "TAGALOG VOWEL SIGN I": 0,
    "TAGALOG VOWEL SIGN U": 1,
    "TAGALOG SIGN VIRAMA": 2,
    "TAGALOG SIGN PAMUDPOD": 3,
}


def _abakada_sort_key(char: str):
    name = unicodedata.name(char, "")
    if name.startswith("TAGALOG LETTER "):
        syllable = name.removeprefix("TAGALOG LETTER ")
        if syllable in ABAKADA_RANK:
            return (0, ABAKADA_RANK[syllable], name)
    if name in MARK_RANK:
        return (1, MARK_RANK[name], name)
    if char == "᜵":  # Baybayin single punctuation, comma-like
        return (2, 0, name)
    if char == "᜶":  # Baybayin double punctuation, period-like
        return (2, 1, name)
    if char in {",", "."}:
        return (2, 2 if char == "," else 3, name)
    # Keep any additional Tagalog marks with the other signs, before punctuation.
    if name.startswith("TAGALOG "):
        return (1, 4, name)
    if "PUNCTUATION" in name:
        return (2, 4, name)
    return (3, name or char)


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        current = [i]
        for j, h in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    edits = sum(edit_distance(r, h) for r, h in zip(references, hypotheses))
    chars = sum(len(r) for r in references)
    return edits / max(1, chars)


def exact_match_accuracy(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if not references:
        return 0.0
    return sum(r == h for r, h in zip(references, hypotheses)) / len(references)


def character_confusion_matrix(references: Sequence[str], hypotheses: Sequence[str],
                               characters: Sequence[str]):
    """Return an aligned character confusion matrix, including insertion/deletion slots.

    Rows are reference characters and columns are predicted characters.
    Insertions use a dedicated row; deletions use a dedicated column. Character
    labels are ordered by Filipino Abakada order. Inserted and deleted slots
    follow the character labels.
    """
    ordered_characters = sorted(set(characters), key=_abakada_sort_key)
    labels = ordered_characters + [INSERTION_LABEL, DELETION_LABEL]
    indices = {char: i for i, char in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    insertion_index = indices[INSERTION_LABEL]
    deletion_index = indices[DELETION_LABEL]

    for reference, hypothesis in zip(references, hypotheses):
        rows, cols = len(reference) + 1, len(hypothesis) + 1
        costs = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            costs[i][0] = i
        for j in range(cols):
            costs[0][j] = j
        for i in range(1, rows):
            for j in range(1, cols):
                costs[i][j] = min(
                    costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                    costs[i - 1][j] + 1,
                    costs[i][j - 1] + 1,
                )

        i, j = len(reference), len(hypothesis)
        while i or j:
            if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
                matrix[indices[reference[i - 1]]][indices[hypothesis[j - 1]]] += 1
                i -= 1
                j -= 1
            elif i and costs[i][j] == costs[i - 1][j] + 1:
                matrix[indices[reference[i - 1]]][deletion_index] += 1
                i -= 1
            else:
                matrix[insertion_index][indices[hypothesis[j - 1]]] += 1
                j -= 1

    return labels, matrix
