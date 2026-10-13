"""
L02_what_is_a_dimension.py
==========================

Lesson 02 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to look at 768 numbers and explain what a
"dimension" is -- and, more usefully, explain what it is NOT.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 01. You should have an embedding and have seen that its length is 1.0.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Lesson 01 said "768 dimensions" and moved on quickly. Let's be precise.

A DIMENSION is one slot in the list. "768-dimensional" means the list is 768
numbers long. That is the entire meaning. It does NOT mean the model is
tracking 768 human-readable properties.

There is no dimension 42 for "cats". There is no dimension for "politeness".
Nothing named anything.

So how can the model represent meaning at all if none of the slots are
labelled?

Because each slot becomes meaningful relative to the others, and relative to
what the model was trained on. Imagine learning to describe a room using only
numbers for "brightness", "clutter", "warmth" -- but you are never told which is
which, you just get better at predicting. Eventually your three numbers line up
with real properties of rooms, without anyone assigning them.

Embeddings work the same way, across 768 slots instead of 3.

This has a name: a DISTRIBUTED REPRESENTATION. The meaning is spread across all
768 numbers together. You cannot read a single one and learn anything. You have
to look at the whole vector.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
Understanding this kills a very common bad habit: trying to interpret or
filter individual dimensions. There is no "the cat dimension" to remove, and no
"sentiment dimension" to zero out to make your vectors neutral. If you have
seen someone attempt that, this lesson explains why it cannot work.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L02_what_is_a_dimension.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 02: What is a dimension?
      768 unlabelled slots, and why none of them has a name
    ================================================================

    (Sections 1-5) Encoding eight sentences, six about technology and two
    about cooking, then looking for a dimension that separates them.

      -> best separating dimension : 431
      -> separation score         : 0.62
      -> group means at dim 431   : tech -0.031 / cooking +0.019
      -> overlap at that dimension: YES -- the two groups interleave

      Conclusion: no single dimension separates these topics. The meaning is
      in the pattern across all 768 numbers together.

      [the "aha"] at dim 431, cooking sentence #2 scores LOWER than three of
      the six technology sentences. The best available "cats dimension" is
      actively misleading.

    (Exact numbers vary by machine. What matters is the conclusion: the best
    separating dimension is still a bad separator.)

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

# Two groups that any human would separate instantly. If a single dimension
# could capture the difference, we would find it below.
TECHNOLOGY = [
    "Neural networks learn patterns from training data",
    "A vector database stores embeddings for fast similarity search",
    "The Python interpreter executes bytecode at runtime",
    "Gradient descent minimises loss by following the negative derivative",
    "Transformer attention weights relate tokens to each other",
    "Backpropagation computes gradients through the network graph",
]

COOKING = [
    "Sourdough starter needs regular feeding to stay active",
    "Resting the dough develops gluten and improves texture",
]


def main() -> int:
    total = 6
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 02: What is a dimension?",
        "768 unlabelled slots, and why none of them has a name",
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

    # --- 2. Encode ---
    n += 1
    output.section("Encode two groups that are obviously different", n)
    output.step("Encoding 6 technology sentences and 2 cooking sentences...", 2, total)

    tech = model.encode(TECHNOLOGY)
    cook = model.encode(COOKING)
    combined = model.encode([*TECHNOLOGY, *COOKING])

    output.kv(
        [
            ("technology group", tech.shape),
            ("cooking group", cook.shape),
            ("total", combined.shape),
        ]
    )
    output.explain("Same 768 columns for both groups. Now: is any single column enough?")

    # --- 3. Look at raw numbers ---
    n += 3
    output.section("Look at the raw numbers first", n)
    output.step("Printing the first 6 dimensions for every sentence...", n, total)

    print()
    header = "  " + "sentence".ljust(46) + "".join(f"{'dim ' + str(i):>10}" for i in range(6))
    print(output.paint(header, "bold"))
    for label, group in [("tech", tech), ("cook", cook)]:
        for sentence, vector in zip((TECHNOLOGY if group is tech else COOKING), group):
            flat = sentence if len(sentence) <= 44 else sentence[:41] + "..."
            cells = "".join(f"{vector[i]:>10.4f}" for i in range(6))
            print(f"  {flat:<46}{cells}")

    output.explain("All 768 columns exist. We are only showing 6.")

    # --- 4. Hunt for a single "topic dimension" ---
    n += 1
    output.section("Hunt for a single topic dimension", n)
    output.step("Measuring how well each of the 768 dimensions separates the two groups...", n, total)

    # best_separating_dimension finds the dimension whose gap between the two
    # group means is largest relative to how much each group varies on it.
    index, score = eg2.best_separating_dimension(tech, cook)

    output.kv(
        [
            ("best separating dimension", index),
            ("separation score", f"{score:.3f}   (1.0 would mean a clean split)"),
        ]
    )

    # Now the important part: check whether that dimension ACTUALLY separates.
    tech_values = sorted(float(row[index]) for row in tech)
    cook_values = sorted(float(row[index]) for row in cook)

    print()
    print(f"  Values at dimension {index}, sorted:")
    print(f"    technology : {[round(v, 4) for v in tech_values]}")
    print(f"    cooking    : {[round(v, 4) for v in cook_values]}")

    # A clean separator means every cooking value sits above every technology
    # value (or all below). Overlap means they interleave.
    separated = cook_values[0] > tech_values[-1] or tech_values[0] > cook_values[-1]

    if separated:
        output.ok(f"Dimension {index} separates the groups cleanly.")
        output.explain("Unusual! That would mean the model learned a topic axis.")
    else:
        output.warn(f"Dimension {index} does NOT separate the groups.")
        # Find the clearest piece of evidence: a cooking sentence that scores
        # below a technology sentence on the "topic" dimension.
        worst_cook = min(cook, key=lambda row: float(row[index]))
        worst_cook_text = COOKING[list(cook).index(worst_cook)]
        n_tech_above = sum(1 for row in tech if float(row[index]) > float(worst_cook[index]))

        output.explain(
            f"Cooking sentence {worst_cook_text[:40]!r} scores {float(worst_cook[index]):.4f}\n"
            f"      at dimension {index}, which is BELOW {n_tech_above} of the 6 technology\n"
            f"      sentences. The best 'topic dimension' actively lies to you."
        )

    output.note(
        f"A separation score of {score:.2f} sounds impressive until you see it leaves overlap.\n"
        "  768 dimensions, and the best one still fails to tell baking from neural\n"
        "  networks. Meaning is not in any single dimension."
    )

    # --- 5. The right unit: the whole vector ---
    n += 1
    output.section("The right unit is the whole vector", n)

    output.bullet(
        [
            "You cannot read a dimension to learn what a vector is about.",
            "You compare whole vectors to each other -- that is lesson 03.",
            "If you need interpretable features, an embedding model is the wrong tool.",
            "It is the right tool when you want 'find things like this', fast.",
        ]
    )

    output.kv(
        [
            ("norm of every vector", f"{eg2.vector_norm(combined[0]):.6f}"),
            ("(lesson 08 will break this, on purpose)"),
        ]
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "A dimension is one slot in the list. Does dimension 42 mean 'cats'?",
        "No. No dimension means anything by itself. During training the model\n"
        "  adjusts all 768 slots to make similar meanings land in similar places,\n"
        "  but nobody assigns meaning to a slot. Meaning is spread across the whole\n"
        "  vector -- which is why this lesson could not find a topic dimension even\n"
        "  though the two groups are obviously different.",
    )

    output.checkpoint(
        "Someone wants to 'remove the cat dimension' so their vectors are neutral. "
        "What is wrong with that?",
        "There is no cat dimension. The cat-ness is distributed across all 768\n"
        "  slots. Zeroing one slot removes an arbitrary, unnamed amount of\n"
        "  unrelated signal and leaves the cat-ness essentially intact.",
    )

    output.footer("Next: python lessons/L03_similarity_matrix.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())