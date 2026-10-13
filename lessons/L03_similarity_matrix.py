"""
L03_similarity_matrix.py
=======================

Lesson 03 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to compute the cosine similarity between two
embeddings using plain Python, get the same answer the library gives you, and
explain why length is deliberately ignored.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 02. You know there is no way to read a single dimension, so comparison is
the only option.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Lesson 02 left us stuck. Meaning lives in the whole vector, so to use it we have
to compare two whole vectors. This is how.

Two vectors from the same origin form an ANGLE. The angle is the thing that
carries meaning:

    angle 0    same direction     -> meaning is similar
    angle 90   perpendicular      -> unrelated
    angle 180  opposite direction -> opposed

COSINE SIMILARITY is just a number for that angle, rescaled so you get:

    1.0   identical direction
    0.0   perpendicular
    -1.0  opposite direction

The formula is:

    cos(a, b) = (a . b) / (|a| * |b|)

Divide by the two lengths and those two lengths vanish from the answer. That is
deliberate and it is the important part: cosine only looks at DIRECTION and
completely ignores how long the vectors are.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
Ignoring length is right for text. "The cat sat on the mat" and "the cat sat on
the mat, quietly, in the afternoon sun" describe almost the same thing. You do
not want the longer sentence to score as a different kind of thing merely for
being longer.

The classic mistake is using a plain DOT PRODUCT, which does not divide by
length and therefore rewards longer vectors. Section 8 shows exactly how badly
that goes wrong.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L03_similarity_matrix.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 03: Comparing two vectors
      Build cosine similarity by hand, then check the library
    ================================================================

      [1/6] Cosine similarity in plain Python
      -> hand-written : 0.7210
      -> library      : 0.7210
      -> difference   : 0.000000
      -> AGREE. The library is just this formula, applied to every pair.

      [2/6] Cosine ignores length; dot product does not
      -> cos(a, a)      = 1.0000
      -> cos(a, 10 * a) = 1.0000   <- same direction, ten times longer
      -> dot(a, a)      = 768.0000
      -> dot(a, 10 * a) = 7680.0000 <- the dot product changed by 10x

      [3/6] The pairwise matrix ...
      ... diagonal is exactly 1.0000 because every vector is perfectly
      aligned with itself. That is correct, not a bug.

      [4/6] Verified against the library: max difference 0.000000

      [5/6] Reading the matrix ...

      [6/6] Ranking a query against a small corpus ...

    (Exact numbers vary by machine. What matters: the hand-written cosine and
    the library agree, and cosine is blind to length while dot product is not.)

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import output

SENTENCES = [
    "How do I reverse a list in Python?",
    "python code snippet to reverse an array or list",
    "What is the best proofing temperature for sourdough bread?",
    "How long should bread dough rest before baking it?",
]


def main() -> int:
    total = 6
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 03: Comparing two vectors",
        "Build cosine similarity by hand, then check the library",
    )

    # --- 1. Load ---
    n += 1
    output.section("Load the model", n)
    output.step("Loading the text-only backbone", 1, total)
    try:
        model = eg2.load_model(modalities="text")
    except ImportError as exc:
        output.error(str(exc))
        return 1

    output.step(f"Encoding {len(SENTENCES)} sentences...", 1, total)
    vectors = model.encode(SENTENCES)

    # --- 2. Cosine by hand ---
    n += 1
    output.section("Cosine similarity in plain Python", n)
    output.step("Computing it with no libraries at all, then comparing...", n, total)

    a, b = vectors[0], vectors[1]

    # eg2.cosine is six lines of arithmetic. No numpy, no torch.
    hand = eg2.cosine(a, b)

    # The library does the same thing with a matrix multiply.
    library = float(model.similarity(a, b).cpu().numpy()[0][0])

    difference = abs(hand - library)

    output.kv(
        [
            ("hand-written", f"{hand:.6f}"),
            ("library", f"{library:.6f}"),
            ("difference", f"{difference:.6f}"),
        ]
    )

    if difference < 1e-5:
        output.ok("They agree.")
        output.explain(
            "SentenceTransformer.similarity is not magic. It is this formula,\n"
            "      applied to every pair of vectors in the set."
        )
    else:
        output.warn(f"They differ by {difference:.6f}. Something is off -- check your versions.")

    # --- 3. Why divide by the lengths ---
    n += 1
    output.section("Cosine ignores length; dot product does not", n)
    output.step("Comparing a vector against a stretched copy of itself...", n, total)

    # Multiply every element by 10: same direction, ten times the length.
    stretched = [float(x) * 10 for x in a]

    output.kv(
        [
            ("cos(a, a)", f"{eg2.cosine(a, a):.4f}"),
            ("cos(a, 10a)", f"{eg2.cosine(a, stretched):.4f}   <- same direction, 10x longer"),
            ("dot(a, a)", f"{eg2.dot_product(a, a):.4f}"),
            ("dot(a, 10a)", f"{eg2.dot_product(a, stretched):.4f}   <- dot product changed"),
        ]
    )

    output.explain(
        "Cosine is blind to length. The dot product scales with it.\n"
        "      That is the whole reason cosine is the default here."
    )

    # --- 4. The pairwise matrix ---
    n += 1
    output.section("The pairwise matrix", n)
    output.step("Comparing every sentence against every other sentence...", n, total)

    matrix = eg2.cosine_matrix(vectors, vectors)
    labels = [f"#{i + 1}" for i in range(len(SENTENCES))]
    eg2.show_similarity_matrix(matrix, labels, title="Cosine similarity (hand-written)")

    output.note(
        "The diagonal is exactly 1.0000. Every vector is perfectly aligned with itself.\n"
        "  That is correct behaviour, not a bug -- and it is the first thing people\n"
        "  think is broken when they see a similarity matrix for the first time."
    )

    # --- 5. Verify the whole matrix against the library ---
    n += 1
    output.section("Verify the whole matrix against the library", n)
    output.step("Comparing all 16 pairs...", n, total)

    from_library = model.similarity(vectors, vectors).cpu().tolist()
    worst = max(abs(matrix[i][j] - from_library[i][j]) for i in range(len(vectors)) for j in range(len(vectors)))

    output.kv([("max difference over all pairs", f"{worst:.8f}")])
    if worst < 1e-4:
        output.ok("Your six lines of Python match the library exactly.")
    else:
        output.error(f"Maximum difference is {worst:.6f} -- investigate before continuing.")

    # --- 6. Read the matrix + rank ---
    n += 1
    output.section("Reading the matrix", n)

    # Off-diagonal entries, most similar first. This is the actual signal.
    pairs = []
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            pairs.append((matrix[i][j], i, j))
    pairs.sort(reverse=True)

    print()
    print(f"  {output.paint('Most similar pairs (diagonal excluded)', 'bold')}")
    for score, i, j in pairs[:3]:
        print(f"    {score:.4f}   #{i + 1} <-> #{j + 1}")

    output.explain(
        f"#1 and #2 are both about reversing a list. #3 and #4 are both about bread.\n"
        "      Those pairs score highest. That is the whole idea working."
    )

    # --- 7. Rank a query ---
    n += 1
    output.section("Rank a query against the corpus", n)
    output.step("Encoding a query and ranking the four sentences against it...", n, total)

    query = "write a function to flip a python array"
    output.kv([("query", repr(query))])

    # Documents get the Document prefix, the query gets SearchQuery. Queries and
    # documents are different kinds of text, and the model was trained to expect
    # them prefixed differently. Lesson L06 covers that properly; L07 shows what
    # happens when you get it wrong.
    doc_embeddings = eg2.encode_documents(model, list(SENTENCES), verbose=False)
    query_embedding = eg2.encode_queries(model, [query], verbose=False)

    scores = [eg2.cosine(query_embedding[0], doc) for doc in doc_embeddings]
    best = max(range(len(scores)), key=lambda i: scores[i])

    print()
    for idx, score in sorted(enumerate(scores), key=lambda p: p[1], reverse=True):
        marker = " <-- best" if idx == best else ""
        flat = SENTENCES[idx]
        print(f"    {score:.4f}   {flat}{marker}")

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "Cosine divides by both lengths. What does that division actually do?",
        "It cancels out the length of each vector, leaving only the ANGLE between\n"
        "  them. So cosine answers 'do these point the same way?' and ignores 'how long\n"
        "  are they?'. A vector and a ten-times-longer copy of itself both score 1.0.",
    )

    output.checkpoint(
        "In the matrix, why is the diagonal 1.0000?",
        "Because every vector is being compared with itself. Same vector means\n"
        "  exactly the same direction, which is an angle of zero, which is 1.0.\n"
        "  It looks like a bug the first time you see it. It is not.",
    )

    output.footer("Next: python lessons/L04_whats_the_prefix.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())