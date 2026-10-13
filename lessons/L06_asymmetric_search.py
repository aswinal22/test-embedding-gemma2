"""
L06_asymmetric_search.py
========================

Lesson 06 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to build a working semantic search index:
pre-encode a corpus once, then answer arbitrary queries against it.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 05. You know what the model receives, and that prefixes cost tokens.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Most search systems are ASYMMETRIC. The two sides are different kinds of text:

    a QUERY      short, question-shaped, written by a person in a hurry
                 "how do I flip a list"

    a DOCUMENT   long, declarative, written by someone else, on a different day
                 "title: List Reverse | text: def reverse_list(items): ..."

These are not the same kind of text. Pretending they are costs you accuracy.

EmbeddingGemma 2 was trained with different instruction prefixes for each side:
`SearchQuery` on the query, `Document` on the corpus. Using them is what
"asymmetric retrieval" means in practice.

The retrieval loop itself is simple:

    1. Index once.   Encode every document. Do this when the corpus changes.
    2. Query often.  Encode the query. Compare against everything you stored.
    3. Rank.        Return the top-k.

Step 1 is the expensive one. Steps 2 and 3 are a single forward pass plus some
arithmetic. That asymmetry is why re-indexing is expensive and querying is not.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is the shape of essentially every production retrieval system. RAG,
document search, code search, product matching -- all of it is "index once,
query many times, compare directions".

Once you have seen this loop, the rest of the series is detail.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L06_asymmetric_search.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 06: Asymmetric search
      Index a corpus once, then query it
    ================================================================

      [1/6] Building the index (done once)
      -> 10 documents encoded -> shape (10, 768)
      -> each title formatted manually, no prompt_name. See L04.

      [2/6] Indexing cost vs querying cost
      -> indexing 10 documents : 412.3 ms
      -> querying 1 document  :   6.8 ms

      [3/6] Running 3 queries against the index
      --- Query 1: "how do I flip a list in python" [SearchQuery]
        #1  0.7421  Python: Reverse a List
        #2  0.6803  Python: Sort a List
        #3  0.5012  Python: Binary Search
      --- Query 2: "no idea how long to proof dough" [SearchQuery]
        #1  0.7104  Baking: Sourdough Proofing
        #2  0.6688  Baking: Chocolate Cake
        #3  0.4990  Python: Binary Search

      [4/6] Does the title actually help?
      -> best match score WITH title    : 0.7421
      -> best match score WITHOUT title : 0.6803
      -> the title moved the top score by 0.0618

      [5/6] What the raw vector store looks like

      [6/6] The shape of a real system

    (Scores vary by machine. What matters: the correct document ranks first, and
    indexing is far more expensive than querying.)

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import output

# A deliberately mixed corpus: three topics, ten documents. Small enough to
# read, varied enough that a wrong answer is visible.
CORPUS = [
    {"title": "Python: Reverse a List", "text": "def reverse_list(items):\n    '''Return a reversed copy using slice indexing.'''\n    return items[::-1]"},
    {"title": "Python: Sort a List", "text": "def sort_list(items):\n    '''Return a new sorted copy of the list.'''\n    return sorted(items)"},
    {"title": "Python: Binary Search", "text": "def binary_search(arr, target):\n    '''Find target in a sorted array in O(log n).'''\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"},
    {"title": "Python: Dict Comprehension", "text": "def invert_dict(d):\n    '''Invert key-value pairs.'''\n    return {v: k for k, v in d.items()}"},
    {"title": "Baking: Sourdough Proofing", "text": "Bulk fermentation at 24C takes 4 to 6 hours. Watch for 50% rise and doming dough."},
    {"title": "Baking: Chocolate Cake", "text": "Cream butter and sugar, fold in cocoa, bake at 175C for 35 minutes."},
    {"title": "Baking: Pizza Dough", "text": "00 flour, 65% hydration, 24 hour cold ferment at 4C for flavour development."},
    {"title": "Astronomy: Black Holes", "text": "An event horizon is the boundary beyond which gravity prevents escape. Singularity mass bends spacetime."},
    {"title": "Astronomy: Exoplanets", "text": "The transit method detects dips in stellar brightness when a planet crosses its star."},
    {"title": "Music: Chord Theory", "text": "A major triad contains root, major third and perfect fifth. Inversions reorder the notes."},
]

QUERIES = [
    {"text": "how do I flip a list in python", "prompt_name": "SearchQuery", "expect": "Python: Reverse a List"},
    {"text": "no idea how long to proof dough", "prompt_name": "SearchQuery", "expect": "Baking: Sourdough Proofing"},
    {"text": "function to find a value in a sorted array", "prompt_name": "CodeRetrieval", "expect": "Python: Binary Search"},
    {"text": "why does light bend near a massive object", "prompt_name": "SearchQuery", "expect": "Astronomy: Black Holes"},
]


def main() -> int:
    total = 6
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 06: Asymmetric search",
        "Index a corpus once, then query it",
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

    # --- 2. Index ---
    n += 1
    output.section("Build the index (once)", n)
    output.step(f"Encoding {len(CORPUS)} documents...", 2, total)

    t0 = time.perf_counter()
    # encode_documents sees dicts with titles, formats them, and adds no
    # prompt_name. You cannot get this wrong from the call site.
    doc_embeddings = eg2.encode_documents(model, CORPUS, verbose=False)
    index_ms = (time.perf_counter() - t0) * 1000

    output.kv(
        [
            ("documents", len(CORPUS)),
            ("shape", f"{doc_embeddings.shape}   <- one 768-vector per document"),
            ("indexing time", f"{index_ms:.1f} ms"),
        ]
    )
    output.explain("This array is your index. Keep it in memory or a database.")

    # --- 3. Cost asymmetry ---
    n += 1
    output.section("Indexing cost vs querying cost", n)
    output.step("Timing one query against the finished index...", n, total)

    t0 = time.perf_counter()
    eg2.encode_queries(model, ["a warm-up query"], verbose=False)
    query_ms = (time.perf_counter() - t0) * 1000

    output.kv(
        [
            (f"indexing {len(CORPUS)} documents", f"{index_ms:.1f} ms"),
            ("encoding 1 query", f"{query_ms:.1f} ms"),
        ]
    )
    output.explain(
        "Index once, query forever. That is why re-indexing is the expensive\n"
        "      operation and search is the cheap one -- and why a corpus you edit\n"
        "      often costs you more than you might expect."
    )

    # --- 4. Query ---
    n += 1
    output.section("Run some queries", n)
    output.step(f"Running {len(QUERIES)} queries against the index...", n, total)

    hits = 0
    for index, query in enumerate(QUERIES, start=1):
        # Asymmetric: SearchQuery for prose, CodeRetrieval for code.
        embedding = eg2.encode_queries(model, [query["text"]], prompt_name=query["prompt_name"], verbose=False)
        results = eg2.rank(embedding[0], doc_embeddings, CORPUS, k=3, show=True)

        top = results[0]
        correct = top["title"] == query["expect"]
        hits += correct

        print()
        status = (
            output.paint("correct", "green") if correct else output.paint(f"expected '{query['expect']}'", "yellow")
        )
        print(f"  query  : {query['text']!r}")
        print(f"  prefix : {query['prompt_name']}")
        print(f"  verdict: {status}")
        print()

    output.kv([("queries where the top hit was correct", f"{hits}/{len(QUERIES)}")])

    # --- 5. Does the title help? ---
    n += 1
    output.section("Does the title actually help?", n)
    output.step("Re-scoring with the titles stripped out...", n, total)

    stripped = [doc["text"] for doc in CORPUS]
    stripped_embeddings = eg2.encode_documents(model, stripped, verbose=False)

    query = QUERIES[0]["text"]
    with_title = eg2.encode_queries(model, [query], verbose=False)
    without_title = eg2.encode_queries(model, [query], verbose=False)

    best_with = max(eg2.cosine(with_title[0], d) for d in doc_embeddings)
    best_without = max(eg2.cosine(without_title[0], d) for d in stripped_embeddings)

    output.kv(
        [
            ("best match WITH title", f"{best_with:.4f}"),
            ("best match WITHOUT title", f"{best_without:.4f}"),
            ("difference", f"{best_with - best_without:+.4f}"),
        ]
    )

    output.explain(
        "Titles add a short, high-signal chunk of text. That is why the model\n"
        "      supports them -- and why `title: none` exists for documents that\n"
        "      genuinely do not have one."
    )

    # --- 6. What the index looks like ---
    n += 1
    output.section("What you are actually storing", n)

    output.table(
        ["Property", "Value"],
        [
            ["vectors", f"{doc_embeddings.shape[0]:,}"],
            ["dimensions each", f"{doc_embeddings.shape[1]:,}"],
            ["bytes each (float32)", f"{doc_embeddings.shape[1] * 4:,}"],
            ["total index size", f"{doc_embeddings.nbytes:,} bytes ({doc_embeddings.nbytes / 1024:.1f} KB)"],
            ["context window", f"{eg2.CONTEXT_TOKENS:,} tokens per document"],
        ],
    )

    output.note(
        f"At float32, a {doc_embeddings.shape[1]}-dimension vector is {doc_embeddings.shape[1] * 4:,} bytes.\n"
        "  One million documents is about 3 GB. Matryoshka truncation (lesson L08)\n"
        "  is how you cut that."
    )

    # --- 7. The production shape ---
    output.section("How this becomes a real system", 7)

    output.bullet(
        [
            "1. Index: encode the corpus, store vectors + metadata in Qdrant/Milvus/pgvector.",
            "2. Query: encode the incoming question, ask the store for top-k.",
            "3. Scale: the store uses an ANN index, not a brute-force matrix.",
            "4. RAG: paste the top-k passages into a language model's prompt.",
        ]
    )

    output.explain(
        "Everything above scales the same way. The lesson L15 version indexes a\n"
        "      folder of real files instead of a hardcoded list."
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "What does 'asymmetric' retrieval mean?",
        "Queries and documents are different kinds of text -- short questions\n"
        "  versus long explanations -- so they get DIFFERENT prefixes.\n"
        "  SearchQuery on the query, Document on the corpus.",
    )

    output.checkpoint(
        "Why is indexing expensive and querying cheap?",
        "Indexing is one model forward pass over the ENTIRE corpus, and it has to\n"
        "  be redone whenever the corpus changes. A query is one forward pass over a\n"
        "  single short string plus some arithmetic against the stored vectors. That\n"
        "  asymmetry is why you re-index rarely and query constantly.",
    )

    output.footer("Next: python lessons/L07_prefix_matters.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())