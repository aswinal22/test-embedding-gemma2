"""Formatting documents and queries correctly.

The important idea in this module is that there are **two different ways** to
give the model a document, and mixing them up silently produces garbage:

    Untitled text:
        model.encode("some body text", prompt_name="Document")
        -> "title: none | text: some body text"

    Titled document:
        model.encode("title: Some Title | text: some body text")
        -> "title: Some Title | text: some body text"      (no prompt_name!)

The trap: `prompt_name="Document"` expands to the literal string
`"title: none | text: "`. So using it *and* formatting the title yourself gives:

    "title: none | text: title: Some Title | text: some body text"

Nothing crashes. It just scores worse than it should. `preview_input` exists so
you can see this before it costs you a debugging session.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import output
from .config import PROMPTS

#: Documents with a real title: format this way, and do NOT pass prompt_name.
def format_document(title: str | None, text: str) -> str:
    """Render a titled document the way the model expects.

    Use this and then encode WITHOUT `prompt_name`. See module docstring.

    Args:
        title: The document title. Empty/None becomes "none", which is what the
            model itself uses for untitled documents.
        text: The body text.
    """
    resolved = title if title else "none"
    return f"title: {resolved} | text: {text}"


def resolve_prompt(prompt_name: str | None, model: Any = None) -> str | None:
    """Look up the prefix text for a prompt name.

    Prefers the loaded model's own prompt table so we always reflect what the
    checkpoint actually does, falling back to the copy in `eg2.config` when no
    model is available.
    """
    if prompt_name is None:
        return None
    prompts = getattr(model, "prompts", None) or PROMPTS
    try:
        return prompts[prompt_name]
    except (KeyError, TypeError):
        known = ", ".join(sorted(prompts))
        raise KeyError(
            f"Unknown prompt_name {prompt_name!r}. Known names: {known}"
        ) from None


def preview_input(
    text: str,
    prompt_name: str | None = None,
    model: Any = None,
    show_prompt: bool = True,
) -> str:
    """Return -- and by default print -- the exact string the model will see.

    This is the function that makes prompts concrete. Everything a prompt does
    is string concatenation, and seeing it removes all the mystery.

    Args:
        text: Your sentence, document, or query.
        prompt_name: e.g. "SearchQuery", "Document", "CodeRetrieval". None for
            no prefix.
        model: Optional loaded model, used to read its real prompt table.
        show_prompt: Print a breakdown of what was added.

    Returns:
        The final string, prefix included.
    """
    prefix = resolve_prompt(prompt_name, model)
    final = f"{prefix}{text}" if prefix else text

    if show_prompt:
        output.section("What the model will actually receive")
        output.kv(
            [
                ("your text", repr(text)),
                ("prompt_name", repr(prompt_name)),
                ("prefix added", repr(prefix) if prefix else "(none)"),
                ("final input", repr(final)),
            ]
        )
        extra = len(final) - len(text)
        if extra:
            output.explain(
                f"{extra} characters were glued to the front of your text."
            )

    return final


def encode_documents(
    model: Any,
    documents: Sequence[Any],
    normalize: bool = True,
    verbose: bool = True,
    **encode_kwargs: Any,
):
    """Encode a corpus of documents, with or without titles, correctly.

    Accepts either plain strings (treated as untitled, given the `Document`
    prefix) or dicts with "title" and "text" keys (formatted manually, given no
    prefix). You cannot get this wrong from the call site.

    Args:
        model: A loaded SentenceTransformer.
        documents: Strings, or dicts shaped ``{"title": str, "text": str}``.
        normalize: L2-normalise the output. This model's default module stack
            already normalises, so this is belt-and-braces for safety.
        verbose: Print what was sent to the model.
        **encode_kwargs: Passed through to ``model.encode``.

    Returns:
        A numpy array of shape ``(len(documents), 768)``.
    """
    texts = [d if isinstance(d, str) else format_document(d.get("title"), d.get("text", "")) for d in documents]

    titled = [not isinstance(d, str) for d in documents]
    if any(titled) and not all(titled):
        # Mixed input is legal but surprising; say so rather than let it slide.
        output.warn(
            "Mixed titled and untitled documents in one batch. Titled documents "
            "get no prefix; untitled ones get the Document prefix."
        )

    if verbose:
        output.section(f"Encoding {len(texts)} document(s)")
        for i, text in enumerate(texts[:3], start=1):
            preview = text if len(text) < 110 else text[:107] + "..."
            prefix = "none (titled, formatted manually)" if titled[i - 1] else repr(PROMPTS["Document"])
            output.kv([(f"doc {i}", preview), ("  prefix", prefix)])
        if len(texts) > 3:
            output.explain(f"...and {len(texts) - 3} more.")

    return model.encode(
        texts,
        prompt_name=None if all(titled) else "Document",
        normalize_embeddings=normalize,
        **encode_kwargs,
    )


def encode_queries(
    model: Any,
    queries: Iterable[str],
    prompt_name: str = "SearchQuery",
    normalize: bool = True,
    verbose: bool = True,
    **encode_kwargs: Any,
):
    """Encode queries for asymmetric retrieval.

    Queries and documents are different KINDS of text -- short and
    question-shaped versus long and declarative -- and the model was trained
    with different prefixes for each. Using the same prefix for both measurably
    hurts retrieval quality; lesson L07 demonstrates it.
    """
    queries = list(queries)
    if verbose:
        output.section(f"Encoding {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}")
        output.kv(
            [
                ("prompt_name", repr(prompt_name)),
                ("prefix", repr(resolve_prompt(prompt_name, model))),
            ]
        )
    return model.encode(
        queries,
        prompt_name=prompt_name,
        normalize_embeddings=normalize,
        **encode_kwargs,
    )