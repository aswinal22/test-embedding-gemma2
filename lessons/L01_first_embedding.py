"""
L01_first_embedding.py
======================

Lesson 01 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to turn a sentence into an array of 768
numbers, and explain what those numbers are -- and, just as importantly, what
they are NOT.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 00 must pass. Run:

    python lessons/L00_check_environment.py

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
An EMBEDDING is a list of numbers that stands in for some piece of content --
a sentence, a paragraph, an image, an audio clip.

That is genuinely all it is. There is no hidden structure. When you see:

    [ 0.012, -0.884,  0.031, ... ]     (768 of them)

you are looking at 768 ordinary floating point numbers. There is no dimension
called "politeness" and no coordinate you can point at and say "this bit means
cats".

So what makes the list useful?

Two sentences that mean similar things end up with lists that point in similar
DIRECTIONS, even when they share almost no words:

    "How do I reverse a list in Python?"
    "python code snippet to reverse an array or list"

Those share the word "reverse". Try something with no shared words at all:

    "How do I make sourdough bread rise properly?"
    "What is the correct proofing temperature for a starter?"

Both get pushed toward "baking / fermentation / time", because the model
learned from millions of examples that those concepts belong together. The
direction of the vector encodes meaning; the individual numbers do not.

A DIMENSION is just one slot in that list. "768-dimensional" means "768 numbers
long". It does not mean there are 768 things being tracked.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This one idea is the whole basis of:

    semantic search      find documents by meaning, not keywords
    RAG                  feed a language model relevant passages
    recommendations      "customers who liked this also liked..."
    clustering           group 10,000 documents into 50 themes automatically
    duplicate detection  spot near-copies of the same content
    multimodal search    find an image using a text description

None of those work by reading the numbers. They all work by comparing the
directions.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L01_first_embedding.py

First run downloads the model (~1.5 GB, a few minutes). Every run after that
starts in seconds.

    ================================================================
      EmbeddingGemma 2 -- Lesson 01: Your first embedding
      Turn a sentence into 768 numbers
    ================================================================

      [1/5] Loading google/embeddinggemma-2 from Hugging Face
          device     : cpu
          precision  : float32
          encoders   : text mode {'vision_config': None, 'audio_config': None}
          -> Text-only mode builds the 270M-parameter backbone and skips the
             vision and audio towers. The download is the same either way;
             this saves RAM, not bandwidth.

      [2/5] Encoding one sentence...

      [3/5] Here is what came back:

      -> shape        : (1, 768)
      -> data type    : float32
      -> first 12 of 768 numbers:
           0.0123  -0.8841   0.0310   0.5567  -0.2210   0.7743
          -0.0098   0.4412   0.1189  -0.6621   0.0034   0.2956
      -> length (norm): 1.0000
      -> smallest / largest value: -0.9842 / 0.9913

      [4/5] What the shape means...

      [5/5] Compare two sentences...

(The actual numbers are different on your machine -- they depend on the
library versions and are not meant to be memorised. The SHAPES, the norm of
1.0000, and the interpretation are what matter.)

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


# Two sentences that mean similar things but share almost no vocabulary.
# Nothing in common lexically; everything in common semantically.
SIMILAR_SENTENCES = [
    "How do I reverse a list in Python?",
    "python code snippet to reverse an array or list",
]

# One about something completely different.
UNRELATED_SENTENCE = "What is the correct proofing temperature for sourdough starter?"


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 01: Your first embedding",
        "Turn a sentence into 768 numbers",
    )

    # --- 1. Load the model ---
    n += 1
    output.section("Load the model", n)

    # text-only mode: builds the 270M text backbone and skips the vision and
    # audio towers. See docs/08-modular-encoders.md (Phase 4) for the details.
    #
    # We pass no precision, so eg2 picks: bfloat16 if the GPU supports it,
    # float32 otherwise. float16 is never chosen -- it breaks this model.
    output.step(f"Loading {eg2.MODEL_ID} from Hugging Face", 1, total)
    try:
        model = eg2.load_model(modalities="text")
    except ImportError as exc:
        output.error(str(exc))
        output.note("Then run lesson 00 to confirm the environment.")
        return 1

    output.explain(
        "Text-only mode builds the 270M-parameter backbone and skips the vision\n"
        "and audio towers. The download is the same either way; this saves RAM."
    )

    # --- 2. Encode one sentence ---
    n += 1
    output.section("Encode one sentence", n)
    sentence = "The cat sat on the mat."
    output.step(f"Encoding: {sentence!r}", 2, total)

    # model.encode() returns a plain numpy array by default: shape (1, 768).
    # convert_to_tensor=True would give you a torch.Tensor instead.
    embedding = model.encode(sentence)

    output.explain(f"Encode finished. The model returned an array of shape {embedding.shape}.")

    # --- 3. Look at the actual numbers ---
    n += 1
    output.section("What came back", n)
    output.step("Printing the shape, the type, and the first 12 numbers...", 3, total)

    first_twelve = embedding[0][:12]
    numbers_text = "  ".join(f"{v: .4f}" for v in first_twelve)

    output.kv(
        [
            ("shape", f"{embedding.shape}   <- 1 sentence, 768 numbers"),
            ("data type", f"{embedding.dtype}"),
            ("length (norm)", f"{_norm(embedding[0]):.4f}   <- always ~1.0"),
            ("smallest value", f"{embedding.min():.4f}"),
            ("largest value", f"{embedding.max():.4f}"),
        ]
    )
    print()
    print(f"    first 12 of {embedding.shape[1]} numbers:")
    for row_start in (0, 6):
        row = "  ".join(f"{v: .4f}" for v in first_twelve[row_start : row_start + 6])
        print(f"      {row}")
    print()
    output.explain(f"Those 12 numbers are a slice of {embedding.shape[1]}. The rest exist too.")
    output.explain("None of them is labelled. That is normal.")

    # --- 4. Explain the shape (section 7 of the lesson template) ---
    n += 1
    output.section("What the shape means", n)

    output.table(
        ["You see", "It means"],
        [
            ["(1, 768)", "1 sentence in, 768 numbers out."],
            ["768", "The model's fixed output width. Always 768 unless you ask to truncate."],
            ["float32", "Each number uses 4 bytes of memory. 768 x 4 = 3,072 bytes per sentence."],
            ["norm ~= 1.0", "The model normalises its output to unit length. It already did this for you."],
            ["values near 0", "Most are small. That is normal for an embedding, not a bug."],
            ["negative values", "Also normal. Embeddings are not percentages and are not 0-to-1."],
        ],
    )

    output.note(
        "The norm being exactly 1.0 is not a coincidence -- this model ends with a\n"
        "  normalisation layer. Remember this: it is the reason lesson 08 has to tell\n"
        "  you to re-normalise after truncating."
    )

    # --- 5. Encode more and see them in parallel ---
    n += 1
    output.section("Encode several at once", n)
    output.step("Encoding three sentences in a single call...", 5, total)

    # One call for three sentences is much faster than three calls -- the model
    # batches them together on the device.
    sentences = [*SIMILAR_SENTENCES, UNRELATED_SENTENCE]
    vectors = model.encode(sentences)

    output.kv([("shape", f"{vectors.shape}   <- 3 sentences, 768 numbers each")])
    output.explain(f"Same call, three rows. This is how you index a whole corpus.")

    print()
    for text, vec in zip(sentences, vectors):
        print(f"    {text}")
        print(f"      {output.paint('...', 'dim')}  norm {_norm(vec):.4f}")

    output.explain("Lesson 03 teaches you how to compare these to each other.")
    output.explain("For now just notice: three rows, each one a point in the same space.")

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "An embedding is a list of numbers. What makes the list useful?",
        "The DIRECTION of the list encodes meaning. Two sentences that mean similar\n"
        "things get similar directions, even with no words in common. Comparing the\n"
        "numbers one at a time tells you almost nothing -- it is the angle between\n"
        "two lists that carries the meaning.",
    )

    output.checkpoint(
        "Why is the norm always 1.0 here?",
        "The model has a normalisation layer as its final step, so it hands you\n"
        "unit-length vectors. That means cosine similarity can be computed as a plain\n"
        "dot product. Hold onto this -- when you truncate a vector in lesson 08, you\n"
        "break this guarantee, and you have to restore it yourself.",
    )

    output.footer("Next: python lessons/L02_what_is_a_dimension.py")
    return 0


def _norm(vector) -> float:
    """Length of a vector -- sqrt of the sum of squared numbers."""
    return float((vector**2).sum() ** 0.5)


if __name__ == "__main__":
    raise SystemExit(main())