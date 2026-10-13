# PLAN.md — Making `test-embedding-gemma2` Beginner-Friendly

> **Status:** Proposed. Nothing in this plan has been implemented yet.
> **Audience:** Python-comfortable, new to ML/embeddings. Can read a `for`-loop and an f-string; has used `pip`. Does **not** know tensors, cosine similarity, tokenizers, dimensions, float precision, RAG, or multimodal encoders.
> **Scope:** Restructure for teaching. The existing `01_…`–`04_…` scripts and `interactive_explorer.py` are kept (not deleted) so git history and any existing links stay intact.

---

## 1. Goal

Turn a working-but-shallow demo repo into a **curriculum**. The end state is not "more scripts" — it is a repo where a beginner can go from *"what is an embedding?"* to *"I built a working semantic search index"* without ever being confused about why the code does what it does, and without ever silently getting a wrong answer.

Success is measured by the checklist in §2, not by script count.

---

## 2. Definition of Beginner-Friendly

Concrete, testable criteria. Every item is objectively checkable.

| # | Criterion | How we verify it |
|---|---|---|
| B1 | Every script runs with **one command, no arguments, no manual setup** | Run each from a clean shell |
| B2 | The **first** command in the README succeeds on a CPU-only laptop | Fresh venv, no GPU |
| B3 | Every script prints **what it's about to do before it does it** | Template §3 |
| B4 | Every script's header shows its **expected output verbatim** | Header comment block |
| B5 | **No unexplained jargon.** Every term is defined in `docs/00-glossary.md` at first use | Glossary link in each lesson |
| B6 | **No unexplained numbers.** Every printed value has a "what am I looking at?" line | Template §3 step 7 |
| B7 | Every non-obvious parameter has an **inline comment stating its effect** | Code review |
| B8 | Every lesson has **one deliberate failure demo** ("now break it") | Template §3 step 8 |
| B9 | Every lesson ends with a **self-check question + answer** | Template §3 step 9 |
| B10 | Nothing requires Colab, a GPU, or a paid service | All run on CPU |
| B11 | **No new required dependencies** beyond `torch`, `sentence-transformers`, `transformers`, `numpy`, `pillow` | `requirements.txt` diff |
| B12 | First successful run completes in **under 5 minutes** on a CPU laptop (excluding model download) | Timed |

**Anti-goal:** the plan must not add jargon, must not add dependencies, and must not make the repo harder to skim for someone who already knows embeddings.

---

## 3. Pedagogy: The Lesson Template

Every lesson in `lessons/` follows the same 9-section shape. Consistency is the whole point — a beginner who learns the shape once can navigate every file after.

```
lessons/
└── L08_matryoshka.py
```

### The 9 sections

| § | Name | Content |
|---|---|---|
| 1 | **Objective** | One sentence: "After this you'll be able to …" |
| 2 | **Prerequisite** | Link to prior lesson. Print a warning and exit if it's unmet. |
| 3 | **What is this?** | 2–3 paragraphs. Plain language. No jargon. Link each term to the glossary. |
| 4 | **Why it matters** | The real-world consequence of getting it wrong. |
| 5 | **The code** | Heavily commented. Every parameter explained inline. |
| 6 | **Expected output** | Verbatim, at the top of the file — so they can diff against reality. |
| 7 | **Read your output** | A table mapping each printed number to its meaning. **This is the highest-value section.** |
| 8 | **Break it on purpose** | A "now do the wrong thing" block showing the failure mode *and its symptom*. |
| 9 | **Checkpoint** | 2–3 questions, answers in a `<details>` block. |

### Why section 7 is the centerpiece

A beginner running `sim[0][1] = 0.847` has learned nothing unless they know *what that number means*. Section 7 forces that. This is the single highest-leverage thing in the plan.

### Why section 8 matters

Silent failure is the enemy of learning. Every hard-won gotcha in this repo (`Document` double-prefix, truncation normalization, fp16 NaN) becomes a **teachable moment** instead of a trap. Section 8 is where the bugs from the code review get turned into lessons.

---

## 4. Target Repo Layout

```
test-embedding-gemma2/
├── PLAN.md                          ← this file
├── START_HERE.md                    ← the new front door
├── README.md                        ← rewritten: 60-second start + roadmap
│
├── src/eg2/                         ← shared "engine" package (the code, not the teaching)
│   ├── __init__.py                  ← public API re-exports
│   ├── config.py                    ← MODEL_ID, MIN_VERSIONS, constants
│   ├── loader.py                    ← load_model(), describe_hardware(), compare_configs()
│   ├── documents.py                 ← format_document(), encode_documents(), encode_queries()
│   ├── similarity.py                ← cosine() by hand, similarity_matrix(), rank()
│   ├── stats.py                     ← vector_norm(), spearman(), topk_agreement()
│   ├── output.py                    ← section(), explain(), table(), checkpoint()
│   └── doctor.py                    ← python -m eg2.doctor
│
├── lessons/                         ← the curriculum (16 lessons, ordered by dependency)
│   ├── L00_check_environment.py
│   ├── L01_first_embedding.py
│   └── …
│
├── docs/                            ← the concept library (read on demand)
│   ├── 00-glossary.md
│   ├── 01-what-is-an-embedding.md
│   ├── … through …
│   └── 09-vector-search-in-practice.md
│
├── exercises/                       ← 5 practice problems with solutions in README
│
├── 01_quickstart_text.py            ← KEPT, bug-fixed, header points at L03
├── 02_task_prompts_search.py        ← KEPT, bug-fixed, header points at L06
├── 03_matryoshka_bench.py           ← KEPT, bug-fixed, header points at L08
├── 04_multimodal_image.py           ← KEPT, header points at L12
└── interactive_explorer.py          ← KEPT, bug-fixed, header points at L06/L08
```

### Design rule for the `eg2` package

> **Silence is the enemy.** Helpers **print what they're doing by default**; pass `quiet=True` to suppress.

A beginner reading `eg2.encode_documents(model, docs)` should *see* the exact string that will hit the tokenizer. That's how a concept library and a shared package coexist without the abstraction hiding the learning.

---

## 5. The `eg2` Package Specification

| Module | Public API | Purpose |
|---|---|---|
| `config.py` | `MODEL_ID`, `MIN_SENTENCE_TRANSFORMERS`, `MIN_TRANSFORMERS` | Single source of truth. **Fixes the `transformers>=4.49` bug.** |
| `loader.py` | `load_model(text_only=True, device=None, precision=None)` | Encodes the "never fp16 / bf16 on GPU" rule in one place. |
| | `describe_hardware()` | Prints Python/torch/ST/transformers versions, GPU, RAM, cache status. |
| | `compare_configs()` | Measures real RSS for the 4 `config_kwargs` combos. **Replaces the README's wrong "191MB".** |
| `documents.py` | `format_document(title, text)` | Returns `title: {title} \| text: {text}` — **no `prompt_name`**. Encodes the §6.1 fix. |
| | `encode_documents(model, docs, quiet=False)` | Correct by construction. Impossible to double-prefix. |
| | `encode_queries(model, queries, prompt="SearchQuery")` | Asymmetric counterpart. |
| | `preview_input(text, prompt_name=None)` | **Prints the exact string the tokenizer will see.** The teaching workhorse. |
| `similarity.py` | `cosine(a, b)` | Cosine similarity in ~6 lines of pure Python. No torch. |
| | `similarity_matrix(a, b)` | Wraps `model.similarity`. |
| | `rank(query_emb, doc_embs, k=3)` | Returns ranked results with titles. |
| `stats.py` | `vector_norm(v)` | Makes truncation's effect on length visible. |
| | `spearman(a, b)` | Ranking correlation — the metric that actually matters for MRL. |
| | `topk_agreement(a, b, k)` | Does the top-3 survive truncation? |
| `output.py` | `section(title)` | Consistent visual structure. |
| | `explain(text)` | One-line "what just happened" narration. |
| | `checkpoint(q, a)` | Question + `<details>` answer. |
| `doctor.py` | `python -m eg2.doctor` | Full preflight check in one command. |

---

## 6. Correctness Fixes Carried Forward

These came out of the code review and are **prerequisites**, not optional cleanup — a beginner who hits a silent wrong answer learns the wrong lesson.

### 6.1 `Document` double-prefix — Phase 1

`config_sentence_transformers.json` on the Hub defines `"Document": "title: none | text: "`. So `prompt_name="Document"` **already injects that prefix**. Passing a pre-formatted `title: … | text: …` string produces:

```
title: none | text: title: <title> | text: <content>
```

The model card states it plainly: use `prompt_name="Document"` for untitled docs (`title: none`), or format manually and **omit** `prompt_name`.

**Fix:** make it structurally impossible via `eg2.documents.format_document()` + `encode_documents()`. Fixes `02_task_prompts_search.py`, `03_matryoshka_bench.py`, `interactive_explorer.py`, notebook cell 4. **Becomes lesson L04.**

### 6.2 Truncation without re-normalization — Phase 4

Verified in `sentence_transformers/sentence_transformer/model.py:962-991`: truncation happens **after** the `2_Normalize` module, and re-normalization runs only if you request it.

`normalize_embeddings` defaults to `False`. **12 `truncate_dim` call sites, 0 `normalize_embeddings`.**

Impact, stated precisely: printed scores are *not* wrong, because `cos_sim` normalizes both operands internally. But the README teaches `encode(truncate_dim=256)` → "store in a vector DB", and any inner-product index or hand-rolled `q @ corpus.T` silently returns bad rankings.

**Fix:** `normalize_embeddings=True` on every truncated call. **Becomes lesson L08 — "Break it on purpose" section.**

### 6.3 `transformers>=4.49.0` cannot load this model — Phase 0

`config.json` declares `model_type: embedding_gemma2`, saved with `transformers_version: 5.18.0.dev0`. A beginner's very first `pip install -r requirements.txt` produces an environment where nothing runs. **This breaks B2.**

**Fix:** `transformers>=5`, `sentence-transformers>=6.1.0` (per the model's own `requirements` block). Drop the dead `tabulate`, add `matplotlib` (notebook needs it, not listed), resolve or drop `numpy<2`. `L00_check_environment.py` asserts these versions.

### 6.4 `trust_remote_code=True` — Phase 1

Present in all 5 scripts and 3 notebook cells. The Hub repo has **no Python files** — only configs and `model.safetensors`. `EmbeddingGemma2Model` is native to `transformers`. It's a no-op that teaches unsafe habits (ST's own docstring warns it executes Hub code locally). **Delete.**

### 6.5 Remaining review items — Phases 1, 4, 5, 6

- README "*~191MB RAM footprint*" is wrong (270M params ≈ 1.06 GB fp32 / 540 MB bf16). Replace with measured numbers from `compare_configs()`.
- Mermaid labels 128d "Ultra Lightweight" with no quality caveat. Model card: 128d degrades multimodal quality substantially (MMEB overall 59.01 → 45.65).
- Notebook cell 6 loads a second full model without freeing the first — OOM risk on Colab. Add `del text_model; torch.cuda.empty_cache()`.
- `interactive_explorer.py:120` keyword-sniffs `"sort"`/`"dict"` to guess code intent; misfires. Replace with explicit `code <query>` / `search <query>`.
- `.gitignore` missing `sample_*.png` (written into CWD by `04_multimodal_image.py:67`), `*.log`, `.ruff_cache/`.
- No `LICENSE` despite the README's Apache-2.0 headline.

---

## 7. Curriculum Ladder

Ordered by **dependency**, not by excitement. Each lesson adds exactly one new concept.

### Phase 0 — Can I run anything? *(prereq: none)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L00_check_environment.py` | Versions, device, RAM, model cache | "You don't need a GPU for any of this." |
| `L01_first_embedding.py` | One sentence → one array of numbers | "An embedding is just a list of floats. That's it." |

**Acceptance:** `python -m eg2.doctor` passes on a CPU-only laptop; `L00` exits non-zero with a clear message if a version is too old.

### Phase 1 — What the numbers mean *(prereq: Phase 0)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L02_what_is_a_dimension.py` | Unpacking "768 numbers"; norm; no human-readable axes | "A dimension is a learned axis. Nobody named it." |
| `L03_similarity_matrix.py` | Cosine similarity built by hand, then verified against `model.similarity()` | "Cosine = 'same direction', regardless of length." |
| `L04_whats_the_prefix.py` | The exact string the model sees | "A prompt is just text glued to the front — including the bug." |

**Acceptance:** a reader can explain, unprompted, why `format_document()` must not be combined with `prompt_name="Document"` (§6.1).

### Phase 2 — Inside the model *(prereq: Phase 1)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L05_tokenizer_peek.py` | Sentence → token IDs; the 8,192-token window | "The model never sees words. It sees integers." |
| `L06_asymmetric_search.py` | Full document/query loop | "Documents and queries are different *kinds* of text." |
| `L07_prefix_matters.py` | A/B: `SearchQuery` vs `CodeRetrieval` vs none | "The prefix measurably changes the answer." |

**Acceptance:** L07 prints a measurable score delta for all three prompt choices.

### Phase 3 — Compression *(prereq: Phase 2)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L08_matryoshka.py` | Truncation; norm collapse; re-normalization; Spearman/top-k agreement | "Truncating breaks length. That's why you must re-normalize." |

**Acceptance:** L08 uses a corpus of ≥20 items and ≥5 queries (the current `03` uses 1 query / 4 docs — statistically meaningless), and shows top-k agreement across 768/512/256/128.

### Phase 4 — Hardware *(prereq: Phase 2)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L09_precision.py` | fp32 vs bf16 vs fp16, run side by side | "fp16 returns NaN instead of raising. That's the danger." |
| `L10_modular_loading.py` | Measured RSS for the 4 `config_kwargs` combos | "You can load 270M params instead of 740M — here's the proof." |

**Acceptance:** L09 reproduces the fp16 degradation; L10 prints measured MB per config, replacing the README's wrong number.

### Phase 5 — Beyond text *(prereq: Phase 1)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L11_multimodal_image.py` | Image + text in one space | "One vector space, two modalities." |
| `L12_interleaved_multimodal.py` | `<|image|>` placeholders | "One embedding can represent text *and* pictures together." |
| `L13_audio_and_video.py` | The modalities the README advertises but never demos | "It's not just images." *(skippable if torchaudio absent)* |
| `L14_cross_lingual.py` | Query in Spanish, doc in English | "The 100+ language claim — now actually tested." |

**Acceptance:** L14 succeeds on at least 3 cross-lingual pairs. *(Cross-lingual is the model's headline feature and is currently entirely untested.)*

### Phase 6 — Put it together *(prereq: Phases 2–3)*

| Lesson | Teaches | Key "aha" |
|---|---|---|
| `L15_mini_rag.py` | Index a folder of `.txt`/`.md`, query it | "Now it's a search engine you built yourself." |

**Acceptance:** `L15` indexes any local text folder passed on the command line, or a bundled sample corpus if none given.

---

## 8. `docs/` Concept Library

Short (≤2 min read), plain-language, and **linked from first use in each lesson**. This is where the "all the architecture" requirement lives.

| Doc | Contents |
|---|---|
| `00-glossary.md` | Every term: embedding, vector, dimension, cosine similarity, token, tokenizer, context window, prefix/prompt, MRL, modality, encoder, bfloat16, RAG, ANN, embedding space. **Backs criterion B5.** |
| `01-what-is-an-embedding.md` | Concept-first, no model needed. What a vector is and why it works. |
| `02-how-a-model-reads-text.md` | Tokens, IDs, attention, pooling, projection. The `512 → 768` step. |
| `03-why-cosine-similarity.md` | Geometry intuition. Why not dot product, why not Euclidean. |
| `04-architecture-of-embeddinggemma2.md` | The full stack, with corrected Mermaid: 740M = 270M text (130M transformer + 140M embedder) + 170M vision + 300M audio; 8,192 ctx; 768d out; 512→768 projection. |
| `05-precision-and-dtypes.md` | Why fp16 overflows, why bf16 is safe (8-bit exponent), why fp32 on CPU. |
| `06-task-prefixes-reference.md` | The full prompt table: `SearchQuery`, `Document`, `CodeRetrieval`, `QuestionAnswering`, `FactChecking`, `Classification`, `Clustering`, `SentenceSimilarity` — and the `title: none` rule. |
| `07-matryoshka-explained.md` | Nesting-doll intuition, storage math, quality table, the re-normalization rule. |
| `08-modular-encoders.md` | The 4 `config_kwargs` combos with measured footprints. |
| `09-vector-search-in-practice.md` | The full retrieval loop end to end. |
| `10-multimodal-explained.md` | Shared space, interleaving, vision token budget, audio/video sampling rates. |

---

## 9. `docs/00-glossary.md` — the Architecture Requirement

You asked specifically for *all the architecture*. These are the concepts currently used in the repo without definition anywhere:

**Foundations:** embedding · vector · dimension · vector space · cosine similarity · dot product · Euclidean distance · similarity vs. distance · normalization

**Model internals:** token · tokenizer · token ID · context window · hidden state · embedding layer · attention · pooling · projection layer · encoder · backbone · modality

**This model specifically:** prefix / task prompt · asymmetric retrieval · MRL (Matryoshka Representation Learning) · `truncate_dim` · modular loading · bfloat16 · `prompt_name` · `config_kwargs`

**Applied:** RAG · ANN index · vector database · top-k retrieval · cross-modal / cross-lingual

---

## 10. Testing Strategy — DROPPED

**Status: not implemented, by decision.** This project is for personal study.
Automated tests were written for Phases 0–5 and then deliberately removed along
with the local dev caches, so the repository stays small and focused on the
lessons. Errors will be found and fixed by running the lessons directly on
Google Colab.

What that gives up, stated plainly so it is a known trade-off rather than a
surprise:

| Regression guard | Consequence if reintroduced |
|---|---|
| §6.1 double-prefix | The bug returns silently in any new script |
| §6.2 truncation normalisation | Wrong rankings under inner-product scoring |
| §6.3 dependency floors | `pip install` produces an environment that cannot load the model |

All three are documented in `docs/` and taught in lessons L04 and L08, so the
knowledge survives even though the automated check does not. If a future
contributor wants the guards back, they are straightforward to reinstate.

**Verification approach now:** run each lesson on Colab and read its output.

---

## 11. Effort and Ordering

| Phase | Deliverable | Size | Depends on |
|---|---|---|---|
| 0 | `requirements.txt` fix, `eg2` package skeleton, `doctor.py`, L00–L01 | **M** | — |
| 1 | Fixes §6.1, §6.4; L02–L04; `docs/00`–`03` | **L** | 0 |
| 2 | L05–L07; `docs/04`, `06` | **M** | 1 |
| 3 | Fixes §6.2; L08; `docs/07` | **M** | 2 |
| 4 | L09–L10; `docs/05`, `08` | **M** | 2 |
| 5 | L11–L14; `docs/10` | **L** | 1 |
| 6 | L15; `docs/09` | **S** | 3 |
| 7 | `README.md`, `START_HERE.md`, `LICENSE`, notebooks, exercises | **M** | all |

Phases 1–6 are **independently shippable**. Phase 0 + 1 alone already satisfy most of §2.

**Risk — scope.** 16 lessons is a lot. Mitigation: phase boundaries above, and each phase is a usable stopping point. If effort is tight, ship 0 → 1 → 7-partial and stop. That alone fixes B1–B5, B7, B9, and both critical bugs.

---

## 12. Definition of Done

- [ ] All 12 criteria in §2 pass, verified on a clean CPU-only laptop
- [ ] `python -m eg2.doctor` passes and exits non-zero with an actionable message on a broken env
- [ ] Both critical bugs (§6.1, §6.2) fixed **and** taught as "Break it on purpose" demos
- [ ] All 5 original scripts still run, bug-fixed, each linking to its lesson
- [ ] `docs/` complete; every glossary term defined at first use
- [x] No new required dependencies (B11)
- [ ] A beginner who has never seen an embedding can go `START_HERE.md` → L15 without asking a question

---

## 13. Open Questions

1. **Should `04_multimodal_image.py` stay top-level?** It duplicates L11. Keeping both means maintaining the same demo twice. *(Default: keep, but make it a 5-line shim that calls `eg2` — reduces the cost.)*
2. **Is the Colab/Notebook track still wanted?** "Save a copy in GitHub" is a dated workflow and duplicates `lessons/`. *(Default: keep one notebook as a fast-track mirror of `lessons/`, move the Git bridge instructions to `CONTRIBUTING.md`.)*
3. **Should audio/video (L13) require `torchaudio`?** That breaks B11 if added to `requirements.txt`. *(Default: optional extra; L13 self-skips with a clear message.)*