"""Comparing embeddings -- cosine similarity, built by hand and verified.

`cosine` and `cosine_matrix` are written in plain Python with no imports. That
is deliberate. Lesson L03 compares this implementation against
`SentenceTransformer.similarity` so you can see that the library is not magic:
it is this formula, applied to every pair.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from . import output


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, in plain Python.

        cos(a, b) = (a . b) / (|a| * |b|)

    Two vectors pointing the same way give 1.0. Perpendicular gives 0.0.
    Opposite gives -1.0. Length is ignored entirely -- only the angle matters.

    Works on lists, tuples, or numpy arrays; there is nothing to import.
    """
    if len(a) != len(b):
        raise ValueError(f"Cannot compare vectors of length {len(a)} and {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        # A zero vector has no direction, so cosine is undefined. Returning 0.0
        # keeps ranking code from crashing on an all-zero embedding.
        return 0.0

    return dot / (norm_a * norm_b)


def cosine_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    """Every row of `a` against every row of `b`. Returns a plain nested list."""
    return [[cosine(row_a, row_b) for row_b in b] for row_a in a]


def similarity_matrix(a: Any, b: Any, model: Any = None) -> list[list[float]]:
    """Similarity matrix, preferring the model's own implementation.

    Pass `model` to use `SentenceTransformer.similarity`. Without it, falls back
    to the hand-rolled `cosine_matrix`. Lesson L03 uses both and shows that
    they agree to floating-point noise.
    """
    if model is not None:
        return model.similarity(a, b).cpu().tolist()
    return cosine_matrix(a, b)


def rank(
    query_embedding: Sequence[float],
    document_embeddings: Sequence[Sequence[float]],
    documents: Sequence[Any] = (),
    k: int = 3,
    show: bool = False,
) -> list[dict]:
    """Rank documents against one query, best first.

    Returns a list of dicts with keys: index, score, title, text.
    """
    scores = [cosine(query_embedding, doc) for doc in document_embeddings]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for rank_position, idx in enumerate(order[:k], start=1):
        entry: dict[str, Any] = {"rank": rank_position, "index": idx, "score": scores[idx]}
        if idx < len(documents):
            doc = documents[idx]
            if isinstance(doc, dict):
                entry["title"] = doc.get("title", f"Document {idx}")
                entry["text"] = doc.get("text", "")
            else:
                entry["title"] = str(doc)
                entry["text"] = str(doc)
        else:
            entry["title"] = f"Vector {idx}"
            entry["text"] = ""
        results.append(entry)

    if show:
        show_ranking(results)
    return results


def show_ranking(results: Sequence[dict]) -> None:
    """Print ranked results in the format every lesson uses."""
    output.table(
        ["#", "Score", "Title"],
        [[r["rank"], f"{r['score']:.4f}", r["title"]] for r in results],
    )
    for r in results:
        if r.get("text"):
            flat = " ".join(r["text"].split())
            preview = flat if len(flat) <= 88 else flat[:85] + "..."
            print(f"    {output.paint('  ', 'dim')}{preview}")


def show_similarity_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    title: str = "Cosine similarity",
) -> None:
    """Print an N x N similarity matrix with row/column headers.

    A square matrix (documents against themselves) has a diagonal of exactly
    1.0 -- every vector is perfectly aligned with itself. That is not a bug and
    is worth pointing at the first time you see one.
    """
    short = [label if len(label) <= 14 else label[:13] + "." for label in labels]
    width = max(6, max(len(s) for s in short) + 1)

    print()
    print(f"  {output.paint(title, 'bold')}")
    print("  " + " " * width + "".join(f"{s:>{max(width, 6)}}" for s in short))
    print("  " + " " * width + "-" * (max(width, 6) * len(short)))
    for i, label in enumerate(short):
        cells = "".join(f"{value:>{max(width, 6)}.3f}" for value in matrix[i])
        print(f"  {label:<{width}}{cells}")