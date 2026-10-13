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