r"""
L11_multimodal_image.py
=======================

Lesson 11 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to encode an image and compare it directly
against text, because both live in the same 768-dimensional space.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 03. You know cosine similarity works between any two vectors of equal
width -- and that turns out to be the entire trick here.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Older systems needed two models. CLIP put images and text in a shared space so
you could compare them; a separate model handled text-to-text search. Two
models, two indexes, two sets of bugs.

EmbeddingGemma 2 is natively multimodal. Text, images, video and audio all
become 768-number vectors in ONE space, and they are all normalised the same
way. So the comparison from lesson 03 just works, unchanged:

    cosine(image_vector, text_vector)

No special function. No separate model. No mapping layer. The same formula.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is what makes "search my photos by describing them" a two-line program
rather than a project. It also means an image and a document can be returned
by the same query, ranked together, in one result list.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L11_multimodal_image.py

To use your own photo instead of the drawn ones:
    python lessons/L11_multimodal_image.py path/to/photo.jpg

    ================================================================
      EmbeddingGemma 2 -- Lesson 11: Images and text, one space
      Compare a picture against words directly
    ================================================================

      [1/5] Loading the multimodal model (text + image, no audio)
      -> 440M parameters

      [2/5] Encoding three images and four text descriptions
      -> images : shape (3, 768)
      -> texts  : shape (4, 768)

      [3/5] Image against text
      image \ text   yellow sunflower   pizza slice   pine forest   blue ocean
      sunflower          0.71              0.33          0.29         0.41
      pizza              0.30              0.74          0.25         0.38
      forest             0.27              0.24          0.73         0.36

      -> The diagonal is highest. The right description wins for each image.

      [4/5] Image against image -- also possible
      -> If images share the space with text, they share it with EACH OTHER.

      [5/5] The honest caveat

    IMPORTANT: the images here are flat colour blocks drawn with Pillow, and
    the vision encoder was trained on photographs. These are OUT OF
    DISTRIBUTION, so ranking can be surprising. That is a real lesson, not a
    bug -- and passing your own photo gives a much better demonstration.

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import multimodal, output

IMAGE_KINDS = ["sunflower", "pizza", "forest"]

TEXTS = [
    "a bright yellow flower with a brown centre",
    "a round baked dish with red topping on a dark table",
    "tall green pine trees under a pale sky",
    "a wide expanse of blue water",
]


def load_image_argument() -> str | None:
    """If the user passed a path, use their photo instead of the drawn ones."""
    if len(sys.argv) < 2:
        return None
    path = sys.argv[1]
    if not Path(path).exists():
        output.error(f"No such file: {path}")
        raise SystemExit(1)
    return path


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 11: Images and text, one space",
        "Compare a picture against words directly",
    )

    # --- 1. Load ---
    n += 1
    output.section("Load the multimodal model", n)
    output.step("Loading text + image (audio encoder still disabled)...", n, total)

    try:
        model = eg2.load_model(modalities="image")
    except ImportError as exc:
        output.error(str(exc))
        return 1

    output.kv(
        [
            ("config_kwargs", str(eg2.LOAD_CONFIGS[1][1])),
            ("vision encoder", "enabled, ~170M"),
            ("audio encoder", "disabled, saving 300M"),
        ]
    )

    # --- 2. Encode ---
    n += 1
    output.section("Encode images and text", n)

    user_image = load_image_argument()
    if user_image:
        from PIL import Image

        images = [Image.open(user_image).convert("RGB")]
        labels = [Path(user_image).name]
        output.kv([("using your photo", user_image)])
    else:
        images = [multimodal.draw_image(kind) for kind in IMAGE_KINDS]
        labels = IMAGE_KINDS

    output.step(f"Encoding {len(images)} image(s) and {len(TEXTS)} text descriptions...", n, total)

    image_embeddings = model.encode(images)
    text_embeddings = eg2.encode_queries(model, TEXTS, verbose=False)

    output.kv(
        [
            ("images", f"{image_embeddings.shape}   <- same width as text"),
            ("texts", f"{text_embeddings.shape}"),
            ("per-image tokens", f"~{multimodal.TOKENS_PER_IMAGE} of {multimodal.CONTEXT_TOKENS}"),
        ]
    )
    output.explain("Identical width. That is what makes the next step possible.")

    # --- 3. Image against text ---
    n += 1
    output.section("Image against text", n)
    output.step("Comparing every image with every description...", n, total)

    matrix = eg2.cosine_matrix(image_embeddings, text_embeddings)
    eg2.show_similarity_matrix(matrix, labels, title="rows = images, columns = text")

    diagonal_hits = sum(1 for i in range(len(labels)) if max(range(len(TEXTS)), key=lambda j: matrix[i][j]) == i)

    output.explain(
        "No special function. The same cosine formula from lesson 03, applied to\n"
        "      an image vector and a text vector."
    )

    if len(labels) > 1:
        output.kv([("images whose best description is their own", f"{diagonal_hits}/{len(labels)}")])

    if user_image:
        best_text = max(range(len(TEXTS)), key=lambda j: matrix[0][j])
        output.explain(
            f"Your photo scored highest against: {TEXTS[best_text]!r}\n"
            "      With a real photograph the match is usually far more reliable than\n"
            "      with the flat shapes used here."
        )

    # --- 4. Image against image ---
    if len(images) > 1:
        n += 1
        output.section("Image against image", n)
        output.step("Comparing the images with each other...", n, total)

        image_matrix = eg2.cosine_matrix(image_embeddings, image_embeddings)
        eg2.show_similarity_matrix(image_matrix, labels, title="images only")

        output.explain(
            "If images share a space with text, they share it with each other.\n"
            "      That is what lets you run near-duplicate detection on images and\n"
            "      never mention text at all."
        )

    # --- 5. Caveat ---
    n += 1
    output.section("The honest caveat", n)

    if not user_image:
        output.warn(
            "The images above are flat colour blocks drawn with Pillow.\n"
            "  The vision encoder was trained on photographs. These are OUT OF\n"
            "  DISTRIBUTION, and a surprising ranking is a real result, not a bug."
        )
        output.bullet(
            [
                "Flat synthetic shapes have no texture, depth, lighting or context.",
                "Real photographs give the encoder the features it learned from.",
                "To see it properly:  python lessons/L11_multimodal_image.py photo.jpg",
                "Any JPEG or PNG works. That is the fair test of this feature.",
            ]
        )
    else:
        output.ok("You tested with a real photograph, which is the meaningful case.")

    output.section("What this unlocks", 5)
    output.bullet(
        [
            "Search photos by describing them.",
            "Find documents and images together in one ranked list.",
            "Near-duplicate image detection with no extra model.",
            "Classify images by comparing against labelled text, not training a model.",
        ]
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "What makes cosine similarity work between an image and a sentence?",
        "Both are encoded into the SAME 768-dimensional vector space and both are\n"
        "  normalised to unit length. Nothing about the comparison is special. It is\n"
        "  the same formula from lesson 03 -- and that is the entire reason this is\n"
        "  a two-line program rather than a project.",
    )

    output.checkpoint(
        "Why do the drawn images sometimes rank unexpectedly?",
        "Because they are out of distribution. The vision encoder learned from\n"
        "  photographs, which have texture, depth, lighting and context. Flat colour\n"
        "  blocks have none of that, so the features it relies on are weak or absent.\n"
        "  Test the feature with a real photo before concluding anything about it.",
    )

    output.footer("Next: python lessons/L12_interleaved_multimodal.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())