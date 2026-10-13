"""Looking inside the tokenizer.

Lesson 05 needs to know what the model actually receives, and that does not
require 1.5 GB of weights. The tokenizer is a few megabytes and runs on its own.

Kept separate from `loader` on purpose: `eg2.load_model()` loads the whole
network, `eg2.load_tokenizer()` loads only the text pipeline. Knowing which one
you need is a habit worth building.
"""

from __future__ import annotations

from typing import Any

from . import output
from .config import CONTEXT_TOKENS, MODEL_ID

_MISSING_HINT = (
    "transformers is not installed, so the tokenizer cannot be loaded. "
    "Run: python -m pip install -r requirements.txt"
)


def load_tokenizer(model_id: str = MODEL_ID):
    """Load just the tokenizer -- no model weights, no torch required at runtime.

    Returns a Hugging Face tokenizer.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_MISSING_HINT) from exc

    return AutoTokenizer.from_pretrained(model_id)


def count_tokens(tokenizer: Any, text: str, add_special_tokens: bool = True) -> int:
    """How many tokens this text costs.

    Args:
        add_special_tokens: Include the model's own framing tokens (BOS, EOS,
            and for this model a task-instruction marker). Leave True for the
            honest "what does this cost me against the context window" number.
    """
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def show_tokens(
    tokenizer: Any,
    text: str,
    max_rows: int = 20,
    add_special_tokens: bool = False,
    show: bool = True,
) -> list[tuple[str, int]]:
    """Break a string into (token, id) pairs and print the alignment.

    Showing the pieces side by side is the whole point. The vertical bars line
    up with the spaces in your sentence, so you can see exactly where the model
    divided the text -- which is rarely where you would have.
    """
    ids = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    pairs = list(zip(tokens, ids))

    if show:
        output.section("Token breakdown")
        print()
        header = f"  {'#':>4}  {'token':<22} {'id':>9}"
        print(output.paint(header, "bold"))
        print("  " + output.paint("-" * (len(header) - 2), "dim"))
        for position, (token, token_id) in enumerate(pairs[:max_rows], start=1):
            printable = token.replace("\n", "\\n").replace("\t", "\\t")
            print(f"  {position:>4}  {printable:<22} {token_id:>9}")
        if len(pairs) > max_rows:
            print(f"  ... and {len(pairs) - max_rows} more")

    return pairs


def fit_to_context(
    tokenizer: Any,
    text: str,
    limit: int = CONTEXT_TOKENS,
) -> tuple[int, float]:
    """How much of the context window this text consumes.

    Returns ``(token_count, fraction_of_window)``.
    """
    used = count_tokens(tokenizer, text)
    return used, used / limit


def context_report(
    tokenizer: Any,
    samples: dict[str, str],
    limit: int = CONTEXT_TOKENS,
) -> list[dict]:
    """Measure several texts against the context window.

    Useful because the cost of text is not proportional to its length in
    characters. Code and non-English text are consistently more expensive per
    character than plain English prose, and this makes that visible.
    """
    output.section("How many tokens does each sample cost?")
    print()

    rows = []
    for label, text in samples.items():
        used, fraction = fit_to_context(tokenizer, text, limit)
        rows.append(
            {
                "label": label,
                "tokens": used,
                "characters": len(text),
                "chars_per_token": len(text) / used if used else 0.0,
                "pct_of_window": fraction * 100,
            }
        )

    output.table(
        ["Sample", "Tokens", "Chars", "Chars/token", "% of 8192"],
        [
            [
                row["label"],
                row["tokens"],
                row["characters"],
                f"{row['chars_per_token']:.2f}",
                f"{row['pct_of_window']:.2f}%",
            ]
            for row in rows
        ],
    )
    return rows


def how_many_of_these(
    tokenizer: Any,
    text: str,
    limit: int = CONTEXT_TOKENS,
    chunk_tokens: int = 256,
) -> int:
    """How many chunks of `chunk_tokens` it takes to hold this text."""
    used = count_tokens(tokenizer, text)
    return -(-used // chunk_tokens)  # ceiling division