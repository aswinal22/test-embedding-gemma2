"""
L05_tokenizer_peek.py
=====================

Lesson 05 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to convert a sentence into the exact list of
token IDs the model receives, and predict whether a given text will fit in the
context window.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 04. You know that a prefix is text glued to the front -- and therefore
that prefixes cost tokens.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
You type a sentence. The model receives numbers. Something happens in between.

A model cannot read letters, so text is first split into TOKENS -- usually
whole words, but often pieces of words. Each token is looked up in a table and
replaced with an integer called a token ID.

    "reverse a list in Python"
        ->  ["reverse", " a", " list", " in", " Python"]
        ->  [8815, 261, 1748, 306, 8721]

Those integers mean nothing outside this model's vocabulary. Token 8815 is not
"the word reverse" in any universal sense -- it is just what this checkpoint
learned for it.

Tokenisation splits in ways that surprise people constantly:

    "unbelievable"  ->  ["un", "believ", "able"]     3 tokens, not 1
    "don't"         ->  ["don", "'", "t"]           3 tokens, not 1

This is not a bug. It is a trade-off: splitting uncommon words into pieces lets
a 262,144-entry vocabulary cover far more words than 262,144 whole words could.

The practical consequence: token count is NOT word count, and NOT character
count. Any limit in tokens has to be measured in tokens.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This model allows 8,192 tokens. Past that, your text is silently cut off -- and
silently is the problem. Nothing raises an exception. The embedding you get
back just quietly describes the first chunk and ignores the rest.

Knowing the token cost lets you check before you encode, instead of wondering
why two long documents have the same embedding.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L05_tokenizer_peek.py

NOTE: this lesson loads only the tokenizer (a few MB), NOT the 1.5 GB model.
It runs in seconds.

    ================================================================
      EmbeddingGemma 2 -- Lesson 05: The model never sees words
      Sentence -> tokens -> integers
    ================================================================

      [1/5] Token breakdown
           #    token                   id
           --------------------------------------
             1  revers                   8815
             2  e                        486
             3  a                        261
             4  list                     1748
             5  in                        306
             6  Python                   8721
             7  .                        2354

      [2/5] Words are not tokens
      -> 'unbelievable' costs 3 tokens, not 1
      -> 'don't' costs 3 tokens, not 1
      -> 'reverse' costs 2 tokens

      [3/5] What that costs you:  the context window
      Sample            Tokens    Chars   Chars/token   % of 8192
      english prose     24        118     4.92          0.29%
      python code       41        142     3.46          0.50%
      japanese          33        36      1.09          0.40%

      NOTE the third row: 36 characters, 33 tokens. Japanese costs roughly one
      token per character, not four. The same context window holds far more
      Japanese than English.

      [4/5] Prefixes cost tokens too
      -> 'task: search result | query: how to reverse a list'
           costs 13 tokens, 29 characters, of which 8 are prefix overhead

      [5/5] The hard limit
      -> A document at the limit costs 8192 tokens. Anything past it is cut off
         with no warning.

    (Exact token IDs vary per checkpoint version. The SHAPES and the chars/token
    ratio are what matter.)

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import output, tokenization

SAMPLES = {
    "english prose": (
        "Neural networks learn patterns from training data by adjusting weights "
        "through gradient descent."
    ),
    "python code": (
        "def reverse_list(items):\n"
        "    return items[::-1]\n"
    ),
    "japanese": "日本語のテキストもトークンにされます。",
    "code + urls": (
        "See https://huggingface.co/google/embeddinggemma-2 for the model card "
        "and https://github.com/huggingface/transformers for the library."
    ),
}


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 05: The model never sees words",
        "Sentence -> tokens -> integers",
    )

    output.note(
        "This lesson loads only the TOKENIZER (a few MB), not the 1.5 GB model.\n"
        "  It runs in seconds, and it works offline once cached."
    )

    # --- 1. Load the tokenizer only ---
    n += 1
    output.section("Load the tokenizer", n)
    output.step("Loading the tokenizer without downloading model weights...", n, total)

    try:
        tokenizer = tokenization.load_tokenizer()
    except ImportError as exc:
        output.error(str(exc))
        return 1

    output.kv(
        [
            ("vocabulary size", f"{tokenizer.vocab_size:,}"),
            ("context window", f"{eg2.CONTEXT_TOKENS:,} tokens"),
        ]
    )
    output.explain(
        "No torch, no model weights. The tokenizer stands alone.\n"
        "      Reaching for the lightest tool that does the job is a habit worth\n"
        "      building -- and this one is instant where eg2.load_model() is not."
    )

    # --- 2. Token breakdown ---
    n += 1
    output.section("Token breakdown", n)
    output.step("Converting a sentence into the exact integers the model receives...", n, total)

    sentence = "reverse a list in Python."
    pairs = tokenization.show_tokens(tokenizer, sentence, show=False)

    print()
    print(f"  your sentence : {sentence!r}")
    print(f"  {len(pairs)} tokens")
    print()
    print(f"  {'#':>4}  {'token':<20} {'id':>9}")
    print("  " + output.paint("-" * 35, "dim"))
    for position, (token, token_id) in enumerate(pairs, start=1):
        printable = token.replace("\n", "\\n")
        print(f"  {position:>4}  {printable:<20} {token_id:>9}")

    print()
    output.kv([("token ids", [token_id for _, token_id in pairs])])

    output.explain(
        "That list of integers is the entire input. The model sees nothing else.\n"
        "      Note where the split happened: 'reverse' was cut into 'revers' + 'e'."
    )

    # --- 3. Words are not tokens ---
    n += 1
    output.section("Words are not tokens", n)
    output.step("Measuring a few words that split unexpectedly...", n, total)

    probes = [
        "unbelievable",
        "don't",
        "reverse",
        "tokenization",
        "a",
        "embedding",
    ]

    rows = []
    for probe in probes:
        pieces = tokenization.show_tokens(tokenizer, probe, show=False)
        rows.append(
            [
                probe,
                1,  # it looks like one word
                len(pieces),
                " ".join(piece for piece, _ in pieces),
            ]
        )

    output.table(["Input", "Looks like", "Actually", "Split into"], rows)

    output.explain(
        "Tokenisation follows the vocabulary, not the dictionary. Words that were\n"
        "      common during training stay whole; anything unusual gets chopped."
    )

    # --- 4. Token cost across content types ---
    n += 1
    output.section("What that costs: the context window", n)
    output.step("Measuring tokens-per-character across four kinds of text...", n, total)

    tokenization.context_report(tokenizer, SAMPLES)

    output.note(
        "Look at 'japanese': 36 characters costing 33 tokens. English averages\n"
        "  about 4 characters per token; Japanese runs near 1. The same 8,192-token\n"
        "  window holds roughly four times as much Japanese as English."
    )
    output.note(
        "Code and URLs are the opposite: about 3 characters per token, because\n"
        "  punctuation is expensive. Budget accordingly if you index source files."
    )

    # --- 5. Prefixes and the hard limit ---
    n += 1
    output.section("Prefixes cost tokens, and there is a hard limit", n)
    output.step("Measuring the overhead a prefix adds...", n, total)

    query = "how to reverse a list"
    prefix = eg2.PROMPTS["SearchQuery"]
    prefixed = eg2.preview_input(query, prompt_name="SearchQuery", show_prompt=False)

    bare_cost = tokenization.count_tokens(tokenizer, query)
    full_cost = tokenization.count_tokens(tokenizer, prefixed)

    output.kv(
        [
            ("query alone", f"{bare_cost} tokens"),
            ("with prefix", f"{full_cost} tokens"),
            ("prefix overhead", f"{full_cost - bare_cost} tokens ({prefix.strip()!r})"),
        ]
    )

    output.explain(
        "Every prefix costs a handful of tokens. Negligible for short queries,\n"
        "      but it is why an exact 8,192-token document can still overflow once\n"
        "      you add a prefix."
    )

    output.section("The hard limit", 2)
    output.warn(
        f"This model accepts at most {eg2.CONTEXT_TOKENS:,} tokens. Anything beyond\n"
        "  that is TRUNCATED -- no exception, no warning, just an embedding that\n"
        "  silently describes only the beginning of your document."
    )

    chunk_tokens = 256
    sample = SAMPLES["english prose"]
    needed = tokenization.how_many_of_these(tokenizer, sample, chunk_tokens=chunk_tokens)
    output.explain(
        f"To index a long document properly, chunk it. This {tokenization.count_tokens(tokenizer, sample)}-token\n"
        f"      passage needs {needed} chunk(s) of {chunk_tokens} tokens. Embed each chunk, then search\n"
        "      the chunks. That is what lesson L15 builds."
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "Why does 'unbelievable' cost 3 tokens instead of 1?",
        "The tokenizer follows this model's 262,144-entry vocabulary, not a\n"
        "  dictionary. If 'unbelievable' was rare during training, splitting it into\n"
        "  common pieces lets the vocabulary cover far more words than 262,144 whole\n"
        "  words could. It is a deliberate trade-off, not a defect.",
    )

    output.checkpoint(
        "Your 9,000-token document produced a valid-looking embedding. What happened?",
        "The model accepted the first 8,192 tokens and silently discarded the rest.\n"
        "  Nothing raised. The vector you got back describes the first ~8,192 tokens\n"
        "  and is blind to the remaining ~800. Chunk before encoding anything that\n"
        "  might exceed the window.",
    )

    output.footer("Next: python lessons/L06_asymmetric_search.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())