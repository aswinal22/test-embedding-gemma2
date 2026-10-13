"""Small measurements that make invisible things visible.

Standard library only, so these work before the model is downloaded.
"""

from __future__ import annotations

import math
from typing import Sequence


def vector_norm(vector: Sequence[float]) -> float:
    """Length of a vector: sqrt of the sum of its squares.

    This model's final layer normalises its output, so every embedding you get
    back has a norm of 1.0. That guarantee is what makes cosine similarity
    equal to a plain dot product -- and it is the thing that silently breaks
    when you truncate a vector. Lesson L08 covers the consequence.
    """
    return math.sqrt(sum(x * x for x in vector))


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain dot product. Unlike cosine, this is sensitive to vector length."""
    return sum(x * y for x, y in zip(a, b))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def column_means(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Average of each column (each dimension) across all rows."""
    if not matrix:
        return []
    width = len(matrix[0])
    return [mean([row[j] for row in matrix]) for j in range(width)]


def best_separating_dimension(
    matrix_a: Sequence[Sequence[float]],
    matrix_b: Sequence[Sequence[float]],
) -> tuple[int, float]:
    """Find the single dimension that best separates two groups of vectors.

    Used in lesson L02 to answer the question "is there one dimension that
    means 'technical'?" by measuring how cleanly the groups split along each
    dimension, then showing the answer is no -- the best one still leaves real
    overlap.

    Returns:
        ``(dimension_index, separation)`` where separation is the difference
        between group means divided by the smaller within-group spread. Higher
        is cleaner. 1.0 means the groups barely overlap.
    """
    mean_a = column_means(matrix_a)
    mean_b = column_means(matrix_b)
    spread_a = column_spread(matrix_a)
    spread_b = column_spread(matrix_b)

    best_index, best_score = 0, float("-inf")
    for j in range(len(mean_a)):
        denominator = min(spread_a[j], spread_b[j]) or 1e-9
        score = abs(mean_a[j] - mean_b[j]) / denominator
        if score > best_score:
            best_index, best_score = j, score

    return best_index, best_score


def column_spread(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Standard deviation of each column. Larger means more variation."""
    if not matrix:
        return []
    width = len(matrix[0])
    spreads = []
    for j in range(width):
        values = [row[j] for row in matrix]
        avg = mean(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        spreads.append(math.sqrt(variance))
    return spreads


# --- Ranking statistics ----------------------------------------------------
# Used by lesson L08 to answer the question that actually matters when
# truncating an embedding: does the ORDER of results survive, not just the
# individual scores?


def ranks(values: Sequence[float]) -> list[float]:
    """Convert scores to ranks, where 1 is best (highest score).

    Ties share the average of the positions they span, which is the standard
    treatment and stops tie-heavy score lists from corrupting the correlation.
    """
    n = len(values)
    # Descending: the highest score must land on rank 1, matching the
    # documented contract and how retrieval results are conventionally shown.
    order = sorted(range(n), key=lambda i: values[i], reverse=True)
    result = [0.0] * n

    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        # Ranks are 1-based, and the tied block spans [start, stop] 0-based.
        average_rank = (start + stop) / 2.0 + 1.0
        for position in range(start, stop + 1):
            result[order[position]] = average_rank
        start = stop + 1

    return result


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation. Returns 0.0 when either input is constant."""
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0

    mean_a, mean_b = mean(a), mean(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    deviation_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    deviation_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))

    if deviation_a == 0.0 or deviation_b == 0.0:
        return 0.0
    return numerator / (deviation_a * deviation_b)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation between two score lists.

    1.0 means the two lists order items identically. 0.0 means no relationship.
    Negative means the order reversed.

    This is the right metric for judging truncated embeddings: retrieval cares
    about the ORDER of results, not whether each score moved by 0.001.
    """
    if len(a) != len(b):
        raise ValueError(f"Cannot correlate lists of length {len(a)} and {len(b)}")
    if len(a) < 2:
        return 0.0
    return pearson(ranks(a), ranks(b))


def topk_overlap(
    reference: Sequence[int],
    candidate: Sequence[int],
    k: int = 3,
) -> float:
    """How much two rankings agree in their top-k, as a Jaccard overlap.

    Returns a value in [0, 1]: 1.0 means identical top-k sets, 0.0 means no
    overlap at all. Unlike "did the top-1 match", this tolerates reordering
    within the result set, which is usually harmless.
    """
    reference_set = set(reference[:k])
    candidate_set = set(candidate[:k])
    if not reference_set and not candidate_set:
        return 1.0
    union = reference_set | candidate_set
    if not union:
        return 1.0
    return len(reference_set & candidate_set) / len(union)


def topk_indices(scores: Sequence[float], k: int = 3) -> list[int]:
    """Indices of the k highest scores, best first."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]


def top1_accuracy(predicted: Sequence[int], expected: Sequence[int]) -> float:
    """Fraction of queries whose top-ranked result was the expected one.

    The bluntest retrieval metric, and the one to reach for first: if top-1 is
    wrong, nothing about the rest of the ranking matters. Lesson L14 uses it to
    measure cross-lingual retrieval per language.
    """
    if not predicted or len(predicted) != len(expected):
        return 0.0
    hits = sum(1 for p, e in zip(predicted, expected) if p == e)
    return hits / len(predicted)


def storage_bytes(dim: int, count: int, bytes_per_value: int = 4) -> int:
    """Bytes needed to store `count` vectors of width `dim` at a given precision.

    4 bytes per value is float32, which is what `encode()` returns by default.
    """
    return dim * bytes_per_value * count


def storage_report(dimensions: Sequence[int], count: int = 1_000_000) -> list[dict]:
    """Storage cost of a corpus at each truncation dimension.

    Defaults to one million vectors, because that is roughly where the numbers
    stop being hypothetical.
    """
    native = max(dimensions)
    rows = []
    for dim in dimensions:
        total = storage_bytes(dim, count)
        rows.append(
            {
                "dim": dim,
                "bytes": total,
                "gib": total / (1024**3),
                "savings_pct": (1.0 - dim / native) * 100,
            }
        )
    return rows