"""
L15_mini_rag.py
===============

Lesson 15 of the EmbeddingGemma 2 series. The last one.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will have built a working semantic search engine over a
folder of real files, and seen exactly where RAG gets its context from.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 06 (asymmetric search) and Lesson 08 (truncation). This lesson combines
both, plus the chunking constraint from lesson 05.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Everything so far used a corpus written inside the Python file. Real retrieval
starts somewhere less convenient: a folder of files you did not write.

This lesson takes a folder path, reads every `.md` and `.txt` in it, splits the
long ones into chunks, embeds everything, and answers queries against it.

Two things happen here that a hand-written corpus never exercises:

    CHUNKING
        The model accepts 8,192 tokens. Anything longer is truncated SILENTLY,
        and you get a vector describing only the beginning. Lesson 05 showed
        how to check the cost. So documents get split into overlapping chunks,
        and each chunk becomes its own searchable unit.

    TITLES
        `load_folder` reads the first Markdown `#` line as the title. Lesson 06
        measured that titles improve the top score, and `black_holes` is a much
        worse search index than `Black Holes`.

Run it with no arguments to index the bundled sample corpus:

    python lessons/L15_mini_rag.py

Or point it at any folder of your own:

    python lessons/L15_mini_rag.py path/to/notes

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is the shape of RAG, minus the language model. Retrieval-Augmented
Generation is two steps:

    1. RETRIEVE  find passages relevant to the question   <- everything so far
    2. GENERATE  paste those passages into a model's prompt and ask

The hard part was always step 1. This lesson is step 1, finished.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L15_mini_rag.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 15: A search engine you built
      Index a folder of files, then query it
    ================================================================

      [1/6] Which corpus?
      -> Using the bundled sample: lessons/sample_corpus
         (pass a folder path as the first argument to index your own)

      [2/6] Loading and chunking
      -> 12 files -> 19 chunks
      -> avg 1.58 chunks per file
      -> longest chunk: 1200 chars (~300 tokens of the 8192 window)

      [3/6] Indexing
      -> encoded 19 chunks -> shape (19, 768)

      [4/6] Querying
      --- "how do I make bread rise faster"
        #1  0.7189  Sourdough Proofing            baking_sourdough_proofing.md
        #2  0.6603  Chocolate Cake                baking_chocolate_cake.md
        #3  0.5871  Espresso Brewing              baking_espresso_brewing.md
      ...

      [5/6] What RAG actually pastes in
      ------------------------------------------------------------
      CONTEXT FOR THE QUESTION "how do I make bread rise faster"
      ------------------------------------------------------------
      [1] Sourdough Proofing  (chunk 1 of 1)
          After mixing, the dough rests at room temperature until it roughly
          doubles in size. At 24C this takes four to six hours...
      [2] Chocolate Cake  (chunk 1 of 1)
          ...

      That block is the whole retrieval output. Everything else in a RAG
      system is a language model reading it.

      [6/6] What changes when you outgrow in-memory search

    (Scores vary by machine. What matters: the right file ranks first for each
    question, across five unrelated topics in five different languages of
    phrasing.)

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import corpus, output

BUNDLED = Path(__file__).resolve().parent / "sample_corpus"

PRESET_QUERIES = [
    "how do I make bread rise faster",
    "reverse a list without making a copy",
    "find something in a sorted array quickly",
    "why can't light escape a black hole",
    "how do I find planets around other stars",
    "notes that make up a major chord",
    "how much coffee and water for a shot",
    "increase my running distance safely",
    "read a csv file with headers",
]


def pick_corpus() -> str:
    """A folder path from argv, or the bundled sample."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return str(BUNDLED)


def show_sources(records) -> None:
    """Print which files ended up in the index."""
    summary = corpus.summarise(records)
    output.section("What is in the index")
    output.kv(
        [
            ("files", summary["files"]),
            ("chunks", summary["chunks"]),
            ("avg chunks per file", f"{summary['average_chunks_per_file']:.2f}"),
            ("total characters", f"{summary['total_characters']:,}"),
        ]
    )
    print()
    for path in corpus.file_names(records):
        name = Path(path).name
        pieces = [r for r in records if r["path"] == path]
        marker = f" ({len(pieces)} chunks)" if len(pieces) > 1 else ""
        print(f"    {name}{marker}")


def show_context(question: str, results, records, limit: int = 3) -> str:
    """Assemble the retrieved passages into a context block.

    This function IS the retrieval half of RAG. Its return value is exactly
    what you would paste into a language model's prompt.
    """
    output.section(f'CONTEXT FOR THE QUESTION "{question}"')
    print()

    blocks = []
    for position, result in enumerate(results[:limit], start=1):
        record = records[result["index"]]
        header = f'[{position}] {record["title"]}  (chunk {record["chunk"]} of {record["chunks_total"]})'
        print(f"  {header}")
        body = " ".join(record["text"].split())
        preview = body if len(body) <= 420 else body[:417] + "..."
        for line in _wrap(preview, 72):
            print(f"      {line}")
        print()
        blocks.append(f"{record['title']}:\n{body}")

    context = "\n\n---\n\n".join(blocks)
    rule = "-" * 60
    print(f"  {rule}")
    output.kv(
        [
            ("passages", len(blocks)),
            ("context size", f"{len(context):,} characters"),
            ("roughly", f"~{len(context) // 4:,} tokens"),
        ]
    )
    return context


def _wrap(text: str, width: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main() -> int:
    total = 6
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 15: A search engine you built",
        "Index a folder of files, then query it",
    )

    # --- 1. Corpus ---
    n += 1
    output.section("Which corpus?", n)

    folder = pick_corpus()
    if len(sys.argv) > 1:
        output.step(f"Indexing your folder: {folder}", 1, total)
    else:
        output.step(f"No argument given, using the bundled sample: {folder}", 1, total)
        output.explain(
            "Pass any folder as the first argument to index your own notes:\n"
            "      python lessons/L15_mini_rag.py ~/Documents/notes"
        )

    # --- 2. Load and chunk ---
    n += 1
    output.section("Load and chunk", n)
    output.step("Reading every .md and .txt, then splitting the long ones...", n, total)

    try:
        records = corpus.index_folder(folder, title_from="heading")
    except (FileNotFoundError, NotADirectoryError) as exc:
        output.error(str(exc))
        output.note("Point it at a real folder, or run it with no argument for the sample.")
        return 1

    if not records:
        output.error(f"No .md or .txt files found in {folder}")
        return 1

    show_sources(records)

    multi = sum(1 for r in records if r["chunks_total"] > 1)
    output.explain(
        f"{len(records) - multi} document(s) fit in one chunk; {multi} needed splitting.\n"
        "      Overlapping chunks mean a sentence across a boundary still appears\n"
        "      whole in at least one of them."
    )
    output.warn(
        "Documents longer than 8,192 tokens are truncated silently by the model.\n"
        "  Chunking is what prevents that. Lesson 05 shows how to measure it."
    )

    # --- 3. Index ---
    n += 1
    output.section("Index", n)
    output.step("Loading the model...", n, total)
    try:
        model = eg2.load_model(modalities="text")
    except ImportError as exc:
        output.error(str(exc))
        return 1

    output.step(f"Embedding {len(records)} chunks...", n, total)
    # Titles from the first Markdown heading. Lesson 06 measured that titles
    # improve the top score, so we use the good ones.
    embeddings = eg2.encode_documents(model, records, verbose=False)

    output.kv([("index shape", f"{embeddings.shape}   <- one 768-vector per chunk")])
    output.explain(
        "That array is your whole index. A real system stores it in a vector\n"
        "      database with metadata; everything above line 3 is identical."
    )

    # --- 4. Query ---
    n += 1
    output.section("Query", n)
    output.step(f"Running {len(PRESET_QUERIES)} questions...", n, total)

    for question in PRESET_QUERIES:
        query_embedding = eg2.encode_queries(model, [question], verbose=False)
        results = eg2.rank(query_embedding[0], embeddings, records, k=3, show=False)

        print()
        print(f"  {output.paint(question, 'bold')}")
        for result in results:
            record = records[result["index"]]
            chunk_note = f" (chunk {record['chunk']}/{record['chunks_total']})" if record["chunks_total"] > 1 else ""
            print(f"    #{result['rank']}  {result['score']:.4f}  {result['title']}{chunk_note}")

    output.explain(
        "Notice the queries are phrased the way a person would ask, not the way\n"
        "      a keyword search would match. 'how do I make bread rise faster'\n"
        "      retrieves the sourdough document without containing the word 'sourdough'."
    )

    # --- 5. RAG ---
    n += 1
    output.section("What RAG actually pastes in", n)

    question = PRESET_QUERIES[0]
    query_embedding = eg2.encode_queries(model, [question], verbose=False)
    results = eg2.rank(query_embedding[0], embeddings, records, k=3, show=False)
    context = show_context(question, results, records)

    output.explain(
        "That block is the entire retrieval output of a RAG system. The language\n"
        "      model reads it and answers. Everything else is plumbing."
    )
    output.note(
        f"The context is ~{len(context) // 4:,} tokens. A real system budgets this:\n"
        "  too little and the model lacks evidence, too much and the answer gets\n"
        "  diluted while costing more to run."
    )

    # --- 6. Scale ---
    n += 1
    output.section("What changes when you outgrow this", n)

    output.table(
        ["", "This lesson", "At 1M chunks"],
        [
            ["Index", "numpy array in RAM", "vector database"],
            ["Search", "compare against every chunk", "ANN index (HNSW, IVF)"],
            ["Speed", "fine", "why ANN exists"],
            ["Storage", "~3 GB", "~3 GB, or 1 GB at 256d"],
        ],
    )

    output.explain(
        "The accuracy is identical. A brute-force scan of a million 768-dimension\n"
        "      vectors is roughly 3GB of dot products, which is fast in principle\n"
        "      and too slow in practice -- hence approximate nearest neighbour\n"
        "      indexes, which trade a little accuracy for a great deal of speed."
    )

    # --- Interactive ---
    output.section("Try it yourself", 6)
    output.note("Type a question and press enter. Empty line or 'quit' to stop.")

    while True:
        try:
            typed = input("\n  [search] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not typed or typed.lower() in ("quit", "exit"):
            break

        embedding = eg2.encode_queries(model, [typed], verbose=False)
        hits = eg2.rank(embedding[0], embeddings, records, k=3, show=False)
        print()
        for hit in hits:
            record = records[hit["index"]]
            print(f"    #{hit['rank']}  {hit['score']:.4f}  {record['title']}  ({Path(record['path']).name})")

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "Why chunk documents at all, when the model accepts 8,192 tokens?",
        "Because anything longer is truncated SILENTLY. No exception is raised and\n"
        "  no warning appears -- you simply get a vector describing only the first\n"
        "  chunk, with nothing to tell you. Chunking makes the limit impossible to\n"
        "  hit accidentally, and overlapping chunks stop sentences from being cut\n"
        "  in half across a boundary.",
    )

    output.checkpoint(
        "What is 'RAG', concretely?",
        "Retrieve then generate. This lesson is the retrieve half: given a question,\n"
        "  find the passages most likely to contain the answer. Generation is just\n"
        "  pasting those passages into a language model's prompt. Retrieval is where\n"
        "  the difficulty lives, and it is entirely independent of any language model.",
    )

    output.footer("That is the curriculum complete. Revisit any lesson from lessons/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())