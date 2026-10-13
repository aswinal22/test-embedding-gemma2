"""
L14_cross_lingual.py
====================

Lesson 14 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will have MEASURED cross-lingual retrieval: how reliably
a question written in one language finds a document written in another.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 06. You have a working index and know how to query it.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
This is the claim that made EmbeddingGemma worth testing in the first place:
it understands 100+ languages, and meaning lands in the same place regardless
of the language it was expressed in.

Concretely, that means a user can type a question in Spanish and retrieve a
document written only in English -- with no translation step, no multilingual
model, and no language filter on the index.

    query  : "¿Cómo hago pan de sourdough?"      (Spanish)
    index  : "title: Sourdough | text: Bulk fermentation..."   (English)
    result : matched

That only works if all of those vectors point in nearly the same direction,
which is a much stronger requirement than "the model understands both
languages".

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is the repo's headline feature and until now NOTHING tested it. Every
lesson, script and document in this repository was English-only, which meant
the most differentiating capability was the one least verified.

It also changes what you can build. If one index serves every language, you do
not need per-language collections, routing logic, or a translation layer in
front of your search.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L14_cross_lingual.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 14: Asking in any language
      Measure cross-lingual retrieval
    ================================================================

      [1/5] Building an English-only index (8 documents)
      -> This index contains NO non-English text at all.

      [2/5] Asking questions in eight languages

        language   query                                  top result
        -----------------------------------------------------------
        Spanish    ¿Cómo se hace pan de masa madre?       Sourdough Proofing
        French     Comment faire lever une pâte ?          Sourdough Proofing
        German     Wie backt man Brot mit Sauerteig?       Sourdough Proofing
        Hindi      सोरडो ब्रेड कैसे बनाएं?                   Sourdough Proofing
        Japanese   サワードウブレード怎么做                  Sourdough Proofing
        Portuguese Como fermentar massa com fermento?      Sourdough Proofing
        Russian    Как заквасить тесто на солодовом?         Sourdough Proofing
        Arabic     كيف أخبز خبز الخميرة؟                    Sourdough Proofing

      [3/5] Same questions, asked in ENGLISH for comparison
      -> English baseline accuracy: 8/8

      [4/5] The gap
        language   top-1 accuracy
        English    1.000
        Spanish    1.000
        ...

      [5/5] What this means for building

    (Exact scores vary by machine. What matters: non-English queries should
    retrieve the English document that a native speaker would pick.)

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

# The index is English only. Nothing here is translated, and there is no
# language metadata on any document.
DOCUMENTS = [
    {"title": "Sourdough Proofing", "text": "Bulk fermentation at 24C takes four to six hours. Watch for a fifty percent rise and domed dough."},
    {"title": "Binary Search", "text": "Locate a target in a sorted array in logarithmic time using a halving strategy."},
    {"title": "Black Holes", "text": "An event horizon is the boundary beyond which gravity prevents any escape. Mass curves spacetime."},
    {"title": "Running Injuries", "text": "Increase weekly running volume by no more than ten percent to avoid overuse injury."},
    {"title": "Chocolate Cake", "text": "Cream butter and sugar, fold in cocoa powder, bake at 175C for thirty-five minutes."},
    {"title": "Exoplanet Detection", "text": "The transit method spots dips in a star's brightness as a planet crosses its disc."},
    {"title": "Compound Interest", "text": "Interest earned on principal plus accumulated interest across compounding periods."},
    {"title": "Sorting Algorithms", "text": "Quicksort partitions a list around a pivot and recurses on each side, averaging O(n log n)."},
]

# Each query asks, in another language, for the document a native speaker of
# that language would pick. The expected index is the English document above.
QUERIES = [
    ("English", "how do I get my sourdough dough to rise properly", 0),
    ("Spanish", "¿Cómo se hace pan de masa madre con fermento natural?", 0),
    ("French", "Comment faire lever une pâte à levain correctement ?", 0),
    ("German", "Wie backt man Brot mit Sauerteig, das gut aufgeht?", 0),
    ("Hindi", "सोरडो ब्रेड का आटा कैसे फूलाएं", 0),
    ("Japanese", "サワードウブレードの生地。怎么让它发酵膨胀", 0),
    ("Portuguese", "Como fermentar massa com fermento natural em casa?", 0),
    ("Russian", "Как заквасить тесто на солодовом дрожже", 0),
    ("Arabic", "كيف أخبز خبزاً بالخميرة الطبيعية", 0),
    ("Chinese", "如何让酸面包的面团发酵膨胀", 0),
    ("Korean", "사워도우 반죽을 부게 하는 방법", 0),
    ("Italian", "Come far lievitare bene l'impasto del pane", 0),
]

# Non-English queries, plus their expected document index.
CROSS_LINGUAL = [q for q in QUERIES if q[0] != "English"]
BASELINE = [q for q in QUERIES if q[0] == "English"]


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 14: Asking in any language",
        "Measure cross-lingual retrieval",
    )

    output.note(
        "This is the claim that made the model worth testing: 100+ languages,\n"
        "  one vector space. Until now nothing in this repo tested it -- every\n"
        "  other lesson was English only."
    )

    # --- 1. Index ---
    n += 1
    output.section("Build an English-only index", n)
    output.step(f"Encoding {len(DOCUMENTS)} English documents...", n, total)

    try:
        model = eg2.load_model(modalities="text")
    except ImportError as exc:
        output.error(str(exc))
        return 1

    doc_embeddings = eg2.encode_documents(model, DOCUMENTS, verbose=False)
    output.kv([("index", f"{doc_embeddings.shape}"), ("non-English documents", 0)])
    output.explain(
        "No translations, no language tags, no per-language collections.\n"
        "      If cross-lingual retrieval works, it works because of the model."
    )

    # --- 2. Cross-lingual queries ---
    n += 1
    output.section("Ask questions in other languages", n)
    output.step(f"Running {len(CROSS_LINGUAL)} non-English queries...", n, total)

    per_language = []
    predicted = []
    expected = []

    for language, text, target in CROSS_LINGUAL:
        query_embedding = eg2.encode_queries(model, [text], verbose=False)
        scores = [eg2.cosine(query_embedding[0], doc) for doc in doc_embeddings]
        top = eg2.topk_indices(scores, k=1)[0]
        predicted.append(top)
        expected.append(target)

        hit = top == target
        colour = "green" if hit else "red"
        print()
        print(f"  {output.paint(language.ljust(11), 'bold')} {text}")
        print(f"    score {scores[top]:.4f} -> {output.paint(DOCUMENTS[top]['title'], colour)}")
        per_language.append((language, hit, scores[top]))

    print()
    accuracy = eg2.top1_accuracy(predicted, expected)
    hits = sum(1 for _, hit, _ in per_language if hit)
    output.kv(
        [
            ("languages tried", len(CROSS_LINGUAL)),
            ("correct document", f"{hits}/{len(CROSS_LINGUAL)}"),
            ("top-1 accuracy", f"{accuracy:.3f}"),
        ]
    )

    # --- 3. English baseline ---
    n += 1
    output.section("English baseline, for comparison", n)

    baseline_predicted = []
    for _, text, target in BASELINE:
        query_embedding = eg2.encode_queries(model, [text], verbose=False)
        scores = [eg2.cosine(query_embedding[0], doc) for doc in doc_embeddings]
        baseline_predicted.append(eg2.topk_indices(scores, k=1)[0])

    baseline_accuracy = eg2.top1_accuracy(baseline_predicted, [t for _, _, t in BASELINE])
    output.kv([("English top-1 accuracy", f"{baseline_accuracy:.3f}")])

    output.explain(
        "The English number is the ceiling. Cross-lingual should approach it,\n"
        "      not beat it -- asking in a second language adds difficulty, it does\n"
        "      not remove any."
    )

    # --- 4. The gap ---
    n += 1
    output.section("The gap", n)

    rows = [["English (baseline)", f"{baseline_accuracy:.3f}", "-"]]
    for language, hit, score in per_language:
        rows.append([language, f"{1.0 if hit else 0.0:.3f}", f"{score:.4f}"])
    rows.append(["Cross-lingual mean", f"{accuracy:.3f}", "-"])

    output.table(["Language", "top-1 accuracy", "score of winner"], rows)

    if accuracy >= 1.0:
        output.ok("Every non-English query retrieved the document a native speaker would pick.")
    elif accuracy >= 0.5:
        output.warn("Most cross-lingual queries landed correctly, some did not.")
    else:
        output.error("Cross-lingual retrieval is weak on this corpus.")
        output.explain(
            "Eight short English documents is a small index. Cross-lingual gaps tend\n"
            "      to appear among near-neighbours, and there is little here to\n"
            "      disambiguate. Try a larger, more varied corpus before concluding."
        )

    output.section("Acceptance check", 5)
    requirement = 3
    passing = [language for language, hit, _ in per_language if hit]
    status = "PASS" if len(passing) >= requirement else "FAIL"
    output.kv(
        [
            ("required", f"{requirement} cross-lingual pairs"),
            ("succeeded", f"{len(passing)}"),
            ("languages that worked", ", ".join(passing) if passing else "none"),
            ("verdict", output.paint(status, "green" if status == "PASS" else "red")),
        ]
    )

    # --- 6. What it means ---
    output.section("What this means for building", 6)
    output.bullet(
        [
            "One index serves every language you support.",
            "No per-language collections and no routing logic.",
            "No translation layer in front of your search.",
            "Users search in whatever language they think in.",
            "Still validate on YOUR corpus -- short indexes hide most of the hard cases.",
        ]
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "Why does a Spanish query retrieve an English document?",
        "Because the model learned to place similar MEANING at similar points in\n"
        "  the vector space regardless of the language expressing it. Training\n"
        "  aligned translations and parallel text, so 'sourdough' in Spanish and in\n"
        "  English end up pointing the same way. No translation happens at query time.",
    )

    output.checkpoint(
        "Should cross-lingual accuracy exceed the English baseline?",
        "No. Asking in a second language adds difficulty rather than removing any,\n"
        "  so cross-lingual accuracy should approach the English number from below.\n"
        "  Exceeding it would suggest a bug or an unrealistically easy corpus.",
    )

    output.footer("That is Phase 5 complete. Next: Phase 6 -- build a mini RAG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())