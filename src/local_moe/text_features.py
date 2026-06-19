from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata


def vectorize(text: str, ngram_min: int, ngram_max: int) -> Counter[str]:
    normalized = normalize_text(text)
    if not normalized:
        return Counter()

    features: Counter[str] = Counter()
    words = normalized.split()
    for word in words:
        features[f"w:{word}"] += 2
    padded = f" {normalized} "
    for size in range(ngram_min, ngram_max + 1):
        if len(padded) < size:
            continue
        for index in range(0, len(padded) - size + 1):
            gram = padded[index : index + size]
            if gram.strip():
                features[f"c:{gram}"] += 1
    return features


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^\w\s]+", " ", without_marks, flags=re.UNICODE)
    return re.sub(r"\s+", " ", cleaned, flags=re.UNICODE).strip()


def cosine(left: Counter[str] | dict[str, float], right: Counter[str] | dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in overlap)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
