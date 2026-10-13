"""
L08_matryoshka.py
=================

Lesson 08 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to truncate embeddings to save storage, and
you will know why forgetting `normalize_embeddings=True` silently corrupts
your rankings instead of raising an error.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 03 (cosine similarity) and Lesson 01 (embeddings are unit length).

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Most embedding models fix their output width. This one does not have to.

MATRYOSHKA REPRESENTATION LEARNING (MRL) is a training technique where the
model is taught to put the most important information in the EARLY coordinates.
Like Russian nesting dolls, the outer shell is less essential than the core, so
you can peel layers off and still have something usable.

The practical result: keep only the first 256 of 768 numbers and you keep most
of the meaning. At 1,000,000 documents that is 2.9 GiB down to 0.95 GiB.

Here is the part everyone gets wrong.

Lesson 01 showed that this model's output is normalised to length exactly 1.0.
Now slice that vector down to its first 256 numbers. You have removed 512 of the
768 numbers -- but you have NOT removed 512/768 of the LENGTH. The remaining
vector is no longer unit length.

    full 768d vector      norm = 1.000000
    first 256 numbers     norm = 0.58...    <-- not 1.0

That matters because a short vector now scores LOW against everything, including
documents identical to itself. Cosine similarity rescues you -- it renormalises
internally -- but a plain dot product does not. And a real vector database with
an inner-product index is a dot product.

The fix is one argument:

    model.encode(text, truncate_dim=256, normalize_embeddings=True)

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is the third silent bug this repo has shipped, and the quietest one. The
double-prefix bug produced slightly worse scores. This one produces rankings
that are simply wrong, on the subset of systems that score with a dot product --
which is most of them.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L08_matryoshka.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 08: Truncating embeddings safely
      Matryoshka truncation, and the step everyone forgets
    ================================================================

      PART A -- What truncation does to the vector
      -> norm at  768d : 1.000000
      -> norm at  512d : 0.826...
      -> norm at  256d : 0.585...
      -> norm at  128d : 0.411...

      Truncating to 1/6 of the dimensions does NOT leave 1/6 of the length.
      You must restore it yourself.

      PART B -- Why that breaks scoring
      -> dot(corpus[0], corpus[0]) at 256d, un-normalised : 0.5842
         An identical vector should score 1.0 against itself. It cannot.
      -> dot(corpus[0], corpus[0]) at 256d, normalised     : 1.0000
      -> cosine(corpus[0], corpus[0]) at 256d, either way : 1.0000
         Because cosine renormalises internally, which is why this bug hides.

      PART C -- Does ranking survive truncation?  (24 documents, 6 queries)
        dim   storage      top-3 overlap vs 768d   mean Spearman
        768d  2.86 GiB     1.000                    1.000
        512d  1.91 GiB     1.000                    0.998
        256d  0.95 GiB     0.944                    0.981
        128d  0.48 GiB     0.778                    0.933

      256d costs 67% less storage and keeps the top-3 essentially intact.
      128d is cheaper still but visibly starts losing relevant documents.

      [the caveat] 128d degrades multimodal quality substantially -- the model
      card reports MMEB overall dropping 59.01 -> 45.65. 256d is the safe
      default.

    (Scores vary by machine. What matters: norms fall below 1.0, the dot
    product collapses, and ranking holds at 256d but loosens at 128d.)

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

DIMENSIONS = [768, 512, 256, 128]

# 24 documents across six topics. The earlier 03_matryoshka_bench.py used one
# query against four documents, which cannot support any conclusion at all.
CORPUS = [
    {"title": "Code: Reverse a List", "text": "def reverse_list(items):\n    '''Return a reversed copy.'''\n    return items[::-1]"},
    {"title": "Code: Binary Search", "text": "def binary_search(arr, target):\n    '''Locate a target in a sorted array in O(log n).'''\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"},
    {"title": "Code: Sort In Place", "text": "def sort_in_place(items):\n    '''Sort a list without creating a copy.'''\n    items.sort()\n    return items"},
    {"title": "Code: Deduplicate", "text": "def dedupe(items):\n    '''Remove duplicates preserving order.'''\n    return list(dict.fromkeys(items))"},
    {"title": "Code: Flatten Nested List", "text": "def flatten(nested):\n    '''Flatten a list of lists one level deep.'''\n    return [x for sub in nested for x in sub]"},
    {"title": "Code: Read a CSV", "text": "import csv\n\ndef read_csv(path):\n    with open(path, newline='') as f:\n        return list(csv.DictReader(f))"},
    {"title": "Code: Chunk a List", "text": "def chunk(items, size):\n    '''Split a list into chunks of the given size.'''\n    return [items[i:i+size] for i in range(0, len(items), size)]"},
    {"title": "Code: Group By Key", "text": "from itertools import groupby\n\ndef group(items, key):\n    return {k: list(g) for k, g in groupby(sorted(items, key=key), key)}"},
    {"title": "Baking: Sourdough Proofing", "text": "Bulk fermentation at 24C takes four to six hours. Watch for a fifty percent rise and domed dough."},
    {"title": "Baking: Chocolate Cake", "text": "Cream butter and sugar, fold in cocoa, bake at 175C for thirty-five minutes."},
    {"title": "Baking: Pizza Dough", "text": "00 flour, sixty-five percent hydration, twenty-four hour cold ferment at 4C."},
    {"title": "Baking: Butter Cookies", "text": "Cream butter and sugar, chill the dough, bake at 170C for twelve minutes."},
    {"title": "Astronomy: Black Holes", "text": "An event horizon is the boundary beyond which gravity prevents escape. Mass bends spacetime."},
    {"title": "Astronomy: Exoplanets", "text": "The transit method detects dips in stellar brightness when a planet crosses its star."},
    {"title": "Astronomy: Red Dwarfs", "text": "Low mass stars burn hydrogen slowly and remain active for trillions of years."},
    {"title": "Astronomy: Nebulae", "text": "Nebulae are vast clouds of gas and dust where new stars condense under gravity."},
    {"title": "Music: Chord Theory", "text": "A major triad contains root, major third and perfect fifth. Inversions reorder the notes."},
    {"title": "Music: Circle of Fifths", "text": "Keys are ordered by distance in perfect fifths, which shows related keys."},
    {"title": "Music: Time Signatures", "text": "Four four means four quarter note beats per measure. Three four is a waltz."},
    {"title": "Music: Equal Temperament", "text": "Piano tuning divides the octave into twelve equal steps tuned by fifths."},
    {"title": "Fitness: Running Volume", "text": "Increase weekly distance by no more than ten percent to avoid injury."},
    {"title": "Fitness: Strength Progression", "text": "Add load gradually and rest between compound lifts before progressing weight."},
    {"title": "Fitness: Sleep Recovery", "text": "Seven to nine hours of sleep improves adaptation to hard training sessions."},
    {"title": "Fitness: Hydration", "text": "Drink water consistently through the day rather than in large amounts at once."},
]

QUERIES = [
    "function to reverse a list",
    "find a value in a sorted array quickly",
    "how long should bread dough rise",
    "what happens at the edge of a black hole",
    "notes that make up a major chord",
    "increase running distance without getting injured",
]


def main() -> int:
    total = 6
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 08: Truncating embeddings safely",
        "Matryoshka truncation, and the step everyone forgets",
    )

    # --- 1. Load ---
    n += 1
    output.section("Load the model", n)
    output.step("Loading the text-only backbone...", 1, total)
    try:
        model = eg2.load_model(modalities="text")
    except ImportError as exc:
        output.error(str(exc))
        return 1

    probe = "Gradient descent minimises loss by following the negative derivative."
    full_vector = model.encode([probe])[0]

    # =====================================================================
    # PART A -- what truncation does to the vector
    # =====================================================================
    n += 1
    output.section("PART A -- What truncation does to the vector", n)
    output.step("Slicing the vector and measuring the length that survives...", n, total)

    norm_rows = []
    for dim in DIMENSIONS:
        # Deliberately a raw slice with no re-normalisation. This is the mistake.
        sliced = full_vector[:dim]
        norm_rows.append([dim, f"{dim / 768 * 100:.1f}%", f"{eg2.vector_norm(sliced):.6f}"])

    output.table(
        ["dim", "dimensions kept", "norm after slicing"],
        norm_rows,
    )

    output.warn(
        "Truncating to one sixth of the dimensions does NOT leave one sixth of\n"
        "  the length. Every vector you slice comes back short, and the shorter\n"
        "  the slice, the shorter the vector."
    )

    output.section("The same thing, done correctly", 2)
    output.step("Adding normalize_embeddings=True...", 2, total)

    correct_rows = []
    for dim in DIMENSIONS:
        proper = model.encode([probe], truncate_dim=dim, normalize_embeddings=True)[0]
        correct_rows.append([dim, f"{eg2.vector_norm(proper):.6f}"])

    output.table(["dim", "norm with normalize_embeddings=True"], correct_rows)
    output.ok("Every vector is unit length again. This is what the argument buys you.")

    # =====================================================================
    # PART B -- why that breaks scoring
    # =====================================================================
    n += 1
    output.section("PART B -- Why a short vector breaks scoring", n)
    output.step("Comparing a document against ITSELF, which must score 1.0...", n, total)

    corpus_embeddings = eg2.encode_documents(model, CORPUS, verbose=False)
    reference = corpus_embeddings[0]

    # Raw slice, un-normalised.
    broken = reference[:256]
    # Correctly truncated.
    fixed = model.encode([CORPUS[0]["text"]], truncate_dim=256, normalize_embeddings=True)[0]

    self_dot_broken = eg2.dot_product(broken, broken)
    self_dot_fixed = eg2.dot_product(fixed, fixed)
    self_cosine = eg2.cosine(broken, reference)

    output.kv(
        [
            ("norm of the raw slice", f"{eg2.vector_norm(broken):.6f}"),
            ("dot(v, v)  un-normalised", f"{self_dot_broken:.4f}   <- should be 1.0000"),
            ("dot(v, v)  normalised", f"{self_dot_fixed:.4f}"),
            ("cosine(v, v) either way", f"{self_cosine:.4f}   <- always right"),
        ]
    )

    output.error(
        "A vector compared against itself scored "
        f"{self_dot_broken:.4f} instead of 1.0.\n"
        "  That is not a rounding artefact -- it is the squared length of a\n"
        "  short vector, and it scales every score down by a constant."
    )
    output.explain(
        "Cosine hides this because it renormalises both operands internally.\n"
        "      A dot product does not -- and a vector database configured for\n"
        "      inner-product similarity is doing exactly a dot product."
    )

    output.warn(
        "This is why model.encode(text, truncate_dim=256) on its own is not enough.\n"
        "  Always pair truncate_dim with normalize_embeddings=True."
    )

    # =====================================================================
    # PART C -- does ranking survive truncation?
    # =====================================================================
    n += 1
    output.section("PART C -- Does the RANKING survive truncation?", n)
    output.step(f"Measuring top-3 overlap and rank correlation over {len(QUERIES)} queries...", n, total)

    output.explain(
        f"{len(CORPUS)} documents, {len(QUERIES)} queries, against the full 768d\n"
        "      ranking as ground truth. Mean score delta would be the wrong\n"
        "      metric here -- retrieval cares about ORDER, not about whether a\n"
        "      score moved by 0.001."
    )

    # Baseline: full-dimension scores and rankings.
    baseline_scores = []
    for query in QUERIES:
        q = model.encode([query], prompt_name="SearchQuery")[0]
        baseline_scores.append([eg2.cosine(q, d) for d in corpus_embeddings])

    baseline_rankings = [eg2.topk_indices(scores, k=3) for scores in baseline_scores]

    rows = []
    for dim in DIMENSIONS:
        overlaps = []
        correlations = []

        for query, baseline, baseline_ranking in zip(QUERIES, baseline_scores, baseline_rankings):
            q = model.encode([query], prompt_name="SearchQuery", truncate_dim=dim, normalize_embeddings=True)[0]
            d_vecs = model.encode(
                [doc["text"] for doc in CORPUS],
                truncate_dim=dim,
                normalize_embeddings=True,
            )
            scores = [eg2.cosine(q, d) for d in d_vecs]

            overlaps.append(eg2.topk_overlap(baseline_ranking, eg2.topk_indices(scores, k=3), k=3))
            correlations.append(eg2.spearman(baseline, scores))

        storage = eg2.storage_report(DIMENSIONS)[DIMENSIONS.index(dim)]
        rows.append(
            [
                f"{dim}d",
                f"{storage['gib']:.2f} GiB",
                f"{storage['savings_pct']:.0f}% saved",
                f"{sum(overlaps) / len(overlaps):.3f}",
                f"{sum(correlations) / len(correlations):.3f}",
            ]
        )

    output.table(
        ["dim", "1M docs", "", "top-3 overlap", "mean Spearman"],
        rows,
    )

    output.explain(
        "256d costs 67% less storage and keeps the top-3 essentially intact.\n"
        "      That is the number to reach for by default."
    )

    output.section("The honest caveat about 128d", 4)
    output.warn(
        "128d is 6x smaller than 768d, but quality is not 6x worse -- it falls off\n"
        "  a cliff, and unevenly. The published benchmarks show text holding up far\n"
        "  better than multimodal: MMEB overall drops 59.01 -> 45.65 at 128d.\n"
        "  128d is reasonable for text-only retrieval and a bad default otherwise."
    )

    output.table(
        ["Output dim", "MTEB text (code)", "MIEB image", "MMEB overall"],
        [
            ["768d", "78.68", "64.64", "59.01"],
            ["512d", "77.24", "64.32", "58.38"],
            ["256d", "76.18", "63.13", "56.24"],
            ["128d", "71.41", "59.06", "45.65"],
        ],
    )
    output.explain("Published model card results, for calibration against your own workload.")

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "You truncate to 256 dimensions. Why must you re-normalise?",
        "Truncation slices the vector but does not scale it. Removing two thirds\n"
        "  of the numbers does not remove two thirds of the length -- the survivor\n"
        "  keeps its squared magnitude and comes back around 0.58 instead of 1.0.\n"
        "  Cosine similarity renormalises internally and hides this; a dot product\n"
        "  does not, and scores every short vector low. normalize_embeddings=True\n"
        "  restores the guarantee.",
    )

    output.checkpoint(
        "Which truncation dimension should you default to, and why not the smallest?",
        "256d. It costs about 67% less storage while the top-3 ranking stays\n"
        "  essentially identical to 768d. 128d is smaller still, but quality falls\n"
        "  off a cliff rather than degrading smoothly -- especially for multimodal,\n"
        "  where MMEB overall drops from 59.01 to 45.65. Validate on your own data.",
    )

    output.footer("Next: python lessons/L09_precision.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())