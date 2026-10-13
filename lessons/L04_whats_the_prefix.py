"""
L04_whats_the_prefix.py
======================

Lesson 04 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to state, without looking it up, why
formatting a title yourself AND passing prompt_name="Document" is a bug --
and know which of the two correct approaches to use.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 03. You can compare embeddings and know why cosine is the right tool.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
EmbeddingGemma 2 was trained with short instruction PREFIXES glued to the front
of text. These prefixes teach the model what JOB the text is doing: is this a
document being indexed, or a search query looking for one? Is this code, or
prose?

A prefix is not magic. It is literally a string concatenated onto your input.
That is the entire mechanism, and this lesson shows it happening.

Here is the trap. The `Document` prefix is not empty:

    Document   ->  "title: none | text: "

Read that again. It already inserts the words `title: none | text: ` for you.

So the obvious-looking code:

    model.encode(f"title: {title} | text: {text}", prompt_name="Document")

produces this input:

    "title: none | text: title: My Title | text: the actual content"

Nothing crashes. No warning appears. You just get quietly worse retrieval.

There are exactly TWO correct ways to encode a document, and you must pick one:

    UNTITLED body text
        model.encode("the actual content", prompt_name="Document")
        -> "title: none | text: the actual content"          CORRECT

    TITLED document
        model.encode("title: My Title | text: the actual content")
        -> "title: My Title | text: the actual content"       CORRECT
        (no prompt_name at all)

`eg2.encode_documents()` picks the right one for you based on whether you gave
it a string or a dict with a title.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This bug is invisible. There is no exception, no warning, and the output still
looks like a plausible 768-number vector. You would only notice if you A/B
tested it, which almost nobody does. It shipped in this repo's own scripts and
sat there producing quietly suboptimal rankings.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L04_whats_the_prefix.py

    (Sections 1-3) Need no model. This lesson works offline.

      -> SearchQuery   : "task: search result | query: "
      -> CodeRetrieval : "task: code retrieval | query: "
      -> Document      : "title: none | text: "

      [4/6] Option A: untitled text with prompt_name
      -> final input : 'title: none | text: the actual content'     CORRECT

      [5/6] Option B: title formatted manually, no prompt_name
      -> final input : 'title: My Title | text: the actual content' CORRECT

      [6/6] The bug: both at once
      -> final input : 'title: none | text: title: My Title | text: the actual content'
      -> 1 extra 'title:' , 1 extra 'none', 1 extra 'text:'   <-- BUG

    (The strings above are exact and do not vary by machine -- prefixing is
    plain string concatenation.)

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


def main() -> int:
    total = 6
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 04: What the model actually receives",
        "Prompts are string concatenation -- and one of them bites",
    )

    output.note(
        "This lesson needs no model. Everything here is plain string work,\n"
        "  so it runs instantly and offline. Keep it in mind: there is no neural\n"
        "  network anywhere in the next few minutes."
    )

    # --- 1. The prompt table ---
    n += 1
    output.section("The prompt prefixes this model knows", n)
    output.step("Listing every prompt_name and the string it expands to...", 1, total)

    output.table(
        ["prompt_name", "expands to"],
        [
            ["SearchQuery", repr(eg2.PROMPTS["SearchQuery"])],
            ["CodeRetrieval", repr(eg2.PROMPTS["CodeRetrieval"])],
            ["Document", repr(eg2.PROMPTS["Document"])],
            ["Classification", repr(eg2.PROMPTS["Classification"])],
            ["Clustering", repr(eg2.PROMPTS["Clustering"])],
            ["SentenceSimilarity", repr(eg2.PROMPTS["SentenceSimilarity"])],
        ],
    )

    output.warn(
        'Stop and read the Document row. It is "title: none | text: ".\n'
        "  It is not empty, and it already contains the words 'title' and 'text'."
    )

    # --- 2. What a prefix does ---
    n += 1
    output.section("What a prefix actually does", n)
    output.step("Applying the SearchQuery prefix to a query...", 2, total)

    query = "how do I reverse a list in python"
    result = eg2.preview_input(query, prompt_name="SearchQuery")

    output.explain(
        "That is the entire mechanism. f-string concatenation.\n"
        "      No hidden tensors, no special token routing, no model internals."
    )

    output.note("Images, audio and video take NO prefix -- only text does.")

    # --- 3. Option A ---
    n += 1
    output.section("Option A: untitled text, with prompt_name", n)
    output.step("Encoding body text that has no title...", 3, total)

    body = "the actual content"
    option_a = eg2.preview_input(body, prompt_name="Document")
    output.ok("CORRECT -- the prefix supplies 'title: none' for you.")

    # --- 4. Option B ---
    n += 1
    output.section("Option B: title formatted manually, no prompt_name", n)
    output.step("Encoding a document that has a real title...", 4, total)

    titled = eg2.format_document("My Title", "the actual content")
    output.kv([("your input", repr(titled))])

    # Note: no prompt_name here. That omission is the entire fix.
    option_b = eg2.preview_input(titled, prompt_name=None)
    output.ok("CORRECT -- you supplied the title, so the model must not add one.")

    # --- 5. The bug ---
    n += 1
    output.section("The bug: doing both at once", n)
    output.step("Formatting the title AND asking for the Document prefix...", 5, total)

    broken = eg2.preview_input(titled, prompt_name="Document")

    # Count the duplicated fragments so the damage is concrete rather than vague.
    doubled_title = broken.count("title:")
    doubled_text = broken.count("text:")
    expected_titles = titled.count("title:")
    expected_texts = titled.count("text:")

    output.warn("WRONG -- and nothing will tell you so.")
    output.kv(
        [
            ("'title:' occurrences", f"{doubled_title}   (expected {expected_titles})"),
            ("'text:' occurrences", f"{doubled_text}   (expected {expected_texts})"),
        ]
    )
    output.explain(
        "The prefix added a second title and a second text marker.\n"
        "      The model now sees a document whose title is itself the string\n"
        "      'title: My Title | text: the actual content'."
    )

    # --- 6. The fix ---
    n += 1
    output.section("How to never hit this", n)
    output.step("Letting the helper choose for you...", 6, total)

    output.bullet(
        [
            "Pass plain strings      -> encode_documents adds the Document prefix.",
            "Pass {'title':..., 'text':...} -> formatted manually, NO prefix added.",
            "Never do both yourself.",
        ]
    )

    output.section("Summary")

    output.table(
        ["You want", "Do this", "Result"],
        [
            [
                "untitled body text",
                'encode(text, prompt_name="Document")',
                "title: none | text: text",
            ],
            [
                "a titled document",
                'encode("title: T | text: B")',
                "title: T | text: B",
            ],
            [
                "a titled document (WRONG)",
                'encode("title: T | text: B", prompt_name="Document")',
                "title: none | text: title: T | text: B",
            ],
        ],
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        'Why is passing prompt_name="Document" alongside a manually formatted title a bug?',
        'Because the Document prefix is literally the string "title: none | text: ".\n'
        "  It already inserts a title marker and a text marker. Formatting the title\n"
        "  yourself adds a second set, so the model reads a document whose title is\n"
        "  the string 'title: My Title | text: the actual content'. No error is\n"
        "  raised, which is what makes it dangerous.",
    )

    output.checkpoint(
        "Which prefix does an IMAGE get?",
        "None. Prefixes apply to text only. Images, audio and video are passed\n"
        "  directly with no prompt_name -- they already occupy a shared space with\n"
        "  text, so they need no instruction telling the model what they are.",
    )

    output.footer("Next: python lessons/L05_tokenizer_peek.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())