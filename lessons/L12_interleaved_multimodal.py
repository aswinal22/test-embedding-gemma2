"""
L12_interleaved_multimodal.py
=============================

Lesson 12 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to produce ONE embedding that represents
text and images together, and retrieve it with a plain text query.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 11. You know images and text share a vector space.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Lesson 11 compared an image against text as two separate inputs. That covers
"find the photo matching this description".

But a product page is not a description of a product. It is a paragraph of text
with photographs sitting in the middle, in a particular order, each one related
to the text around it.

Interleaving handles that. You write text with PLACEHOLDER TOKENS marking where
each piece of media belongs, then hand over the media separately:

    "Waterproof running shoes. <|image|> Breathable mesh upper. <|image|> Grip test: <|video|>"

    model.encode({
        "text":  "Waterproof running shoes. <|image|> Breathable mesh upper. <|image|> Grip test: <|video|>",
        "image": ["shoe.jpg", "mesh.jpg"],
        "video": "demo.mp4",
    })

Each placeholder is filled from the matching key, left to right. Two `<|image|>`
markers consume the first two entries of the image list. The result is ONE
768-number vector representing the whole thing.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is how you index a document that CONTAINS images, rather than one that
describes them. The vector captures the combined meaning, so a text-only query
can retrieve it.

It also means your index does not care what a "document" is. A passage, a
product listing with photos, a slide deck, a video with captions -- all become
the same shape of vector, all searchable with the same query.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L12_interleaved_multimodal.py [image1.png image2.png ...]

    ================================================================
      EmbeddingGemma 2 -- Lesson 12: Text and images, together
      One embedding for a whole mixed document
    ================================================================

      [1/5] The interleaved syntax
      -> text  : "Yellow petals around a brown centre. <|image|> ... <|image|>"
      -> image : [sunflower.png, sunflower.png]
      -> video : (none)

      [2/5] Building a mixed document
      -> segments: text, image, text, image

      [3/5] Encoding it as ONE vector
      -> shape (1, 768)     one sentence AND two images in a single number

      [4/5] Retrieving it with plain text
      -> "a flower"                    -> score 0.58  <- retrieved
      -> "a mountain landscape"         -> score 0.21

      [5/6] The shared context budget
      -> 2 images = 560 tokens of 8192 (6.8%)

    NOTE: sentence-transformers >= 6.1.0 is REQUIRED for this. Older versions
    ignore the order of multimodal dict inputs and cannot put several images
    into a single embedding.

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


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 12: Text and images, together",
        "One embedding for a whole mixed document",
    )

    try:
        import sentence_transformers
        from packaging.version import parse as parse_version

        installed = parse_version(sentence_transformers.__version__)
        if installed < parse_version("6.1.0"):
            output.error(
                f"sentence-transformers {sentence_transformers.__version__} is too old "
                "for interleaved input.\n"
                "  Run: python -m pip install -U 'sentence-transformers>=6.1.0'"
            )
            return 1
    except ImportError:
        output.error("sentence-transformers is not installed.")
        return 1

    # --- 1. Syntax ---
    n += 1
    output.section("The interleaved syntax", n)

    segments = [
        ("Bright yellow petals radiating from a brown centre.", "text"),
        ("<|image|>", "image"),
        ("The flower stands against open blue sky.", "text"),
        ("<|image|>", "image"),
    ]
    text_stream = multimodal.interleave(segments)

    output.kv(
        [
            ("text", repr(text_stream)),
            ("image", "[sunflower.png, sunflower.png]"),
            ("video", "(none)"),
        ]
    )
    output.explain(
        "Each <|image|> is filled from the image list, left to right.\n"
        "      Two markers, two entries. The order matters."
    )

    # --- 2. Build the document ---
    n += 1
    output.section("Build a mixed document", n)

    user_images = sys.argv[1:]
    if user_images:
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in user_images]
        output.kv([("using your images", ", ".join(user_images))])
    else:
        images = [multimodal.draw_image("sunflower"), multimodal.draw_image("sunflower")]
        output.note(
            "No image paths given, so using the drawn sunflower twice.\n"
            "  Pass real photos for a meaningful test: thisLesson.py a.jpg b.jpg"
        )

    # Shrink the stream if the user supplied fewer images than placeholders.
    while segments.count(("<|image|>", "image")) > len(images):
        for position in range(len(segments) - 1, -1, -1):
            if segments[position] == ("<|image|>", "image"):
                del segments[position]
                break
    text_stream = multimodal.interleave(segments)

    output.kv(
        [
            ("segments", " -> ".join(modality for _, modality in segments)),
            ("placeholders", text_stream.count("<|image|>")),
            ("images supplied", len(images)),
        ]
    )

    # --- 3. Encode as one vector ---
    n += 1
    output.section("Encode it as ONE vector", n)
    output.step("Loading the multimodal model...", n, total)

    try:
        model = eg2.load_model(modalities="image")
    except ImportError as exc:
        output.error(str(exc))
        return 1

    output.step("Encoding text and images in a single call...", n, total)
    interleaved = model.encode(
        {
            "text": text_stream,
            "image": images,
        }
    )

    output.kv([("shape", f"{interleaved.shape}   <- ONE vector for text + {len(images)} image(s)")])
    output.explain(
        "Compare with lesson 11, where an image was its own vector. Here the\n"
        "      paragraphs and the pictures are fused into a single representation."
    )

    # --- 4. Retrieve with plain text ---
    n += 1
    output.section("Retrieve it with plain text", n)

    probes = [
        "a flower with bright yellow petals",
        "an outdoor nature scene",
        "a mountain covered in snow",
        "a database schema for user accounts",
    ]

    probe_embeddings = eg2.encode_queries(model, probes, verbose=False)
    scores = [eg2.cosine(probe_embeddings[i], interleaved[0]) for i in range(len(probes))]

    order = sorted(range(len(probes)), key=lambda i: scores[i], reverse=True)

    output.table(
        ["Plain-text query", "score"],
        [[probes[i], f"{scores[i]:.4f}"] for i in order],
    )

    output.explain(
        "No image in the query, no image in the index entry's filename. A text-only\n"
        "      query retrieved a document that CONTAINS images."
    )

    # --- 5. Budget ---
    n += 1
    output.section("The shared context budget", n)

    total_tokens, percent = multimodal.budget_for({"image": len(images)})
    output.table(
        ["Item", "Tokens", "% of 8192"],
        [
            [f"{len(images)} image(s)", f"~{total_tokens}", f"{percent:.1f}%"],
            ["text", "~30", "~0.4%"],
            ["combined", f"~{total_tokens + 30}", f"{(total_tokens + 30) / multimodal.CONTEXT_TOKENS * 100:.1f}%"],
        ],
    )

    output.kv(
        [
            ("images that fit alone", f"{multimodal.CONTEXT_TOKENS // multimodal.TOKENS_PER_IMAGE}"),
            ("audio seconds that fit", f"{multimodal.CONTEXT_TOKENS // multimodal.TOKENS_PER_AUDIO_SECOND}"),
        ]
    )
    output.explain(
        "One budget for everything. Interleaving is not free -- media consumes\n"
        "      context that text would otherwise have had."
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "What does <|image|> in the text stream actually do?",
        "It marks the POSITION of an image inside the sequence. The model fills\n"
        "  each placeholder from the matching key in the dict, left to right. Two\n"
        "  markers consume the first two entries of the image list, so the order of\n"
        "  the list and the order of the placeholders must agree.",
    )

    output.checkpoint(
        "Why does the whole thing become one vector, and why does that matter?",
        "Because the text and the media are fused during encoding, the result\n"
        "  represents the document as a whole rather than as separate parts. That is\n"
        "  what lets a plain text query retrieve a document that contains images --\n"
        "  your index no longer needs a separate path for text and for media.",
    )

    output.footer("Next: python lessons/L13_audio_and_video.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())