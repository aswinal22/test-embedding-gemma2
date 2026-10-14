# Vector search in practice

How the pieces fit into a real system, and what breaks when you scale up.

---

## The pipeline

```
  1. LOAD      read a folder of files
  2. CHUNK     split anything that exceeds the context window
  3. EMBED     one 768-d vector per chunk
  4. STORE     vectors + metadata in a vector database
  5. QUERY     embed the question, fetch top-k
  6. GENERATE  paste those passages into a language model   <- the "RAG" in RAG
```

Steps 1–5 are retrieval. Step 6 is RAG. The retrieval half is the hard part,
and it has nothing to do with language models.

---

## Step 2 — why chunking is not optional

The model accepts 8,192 tokens. A longer document is **truncated silently**:
no exception, no warning, just a vector describing the beginning.

Chunk before you encode. Two details:

- **Chunk size.** Around 1,200 characters is roughly 300 tokens, which leaves
  generous room for a prefix and for a retrieved-neighbourhood budget.
- **Overlap.** Repeat the last ~150 characters of each chunk at the start of
  the next, so a sentence spanning a boundary still appears whole in one of them.

Chunk on paragraph boundaries where possible. A chunk that starts mid-sentence
embeds worse, and you are comparing against whole sentences.

## Step 3 — titles are worth having

Lesson 06 measured that document titles improve the top-1 score. `black_holes`
is a worse index key than `Black Holes`.

Read the first Markdown `#` line as the title, falling back to the filename.
Then use the two-correct-approaches rule from
[06-task-prefixes-reference](06-task-prefixes-reference.md): format the title
yourself and **omit** `prompt_name`.

---

## Step 5 — the query

Asymmetric, exactly as in lesson 06:

```python
embeddings = eg2.encode_documents(model, records)              # no prompt_name, titles
query       = eg2.encode_queries(model, [question])            # SearchQuery prefix
```

Queries get the `SearchQuery` prefix; documents get their title formatted
manually. They are different kinds of text.

---

## Step 4 — when brute force stops working

This repo compares the query against every chunk. That is fine at 19 chunks and
useless at a million.

| | In-memory (here) | Production |
|---|---|---|
| Index | numpy array | vector database |
| Search | compare against all | ANN index (HNSW, IVF) |
| 1M × 768d brute force | ~3 GB of dot products | ~50 ms per query |
| Accuracy | exact | approximate |

**ANN** (approximate nearest neighbour) indexes trade a small amount of
accuracy for a large amount of speed. Common choices:

| Database | Index | Notes |
|---|---|---|
| Qdrant | HNSW | good defaults, filtering |
| Milvus | HNSW, IVF | built for very large scale |
| pgvector | HNSW, IVFFlat | Postgres, no new service |
| Weaviate | HNSW | hybrid keyword + vector |
| LanceDB | IVF | embedded, file-based |

### Compression

Truncation and storage are the same lever — see
[07-matryoshka-explained](07-matryoshka-explained.md). One million 256-d
vectors is about 1 GB instead of 2.9 GB, with ranking essentially intact.

---

## Step 6 — what retrieval hands to the model

The retrieved passages, formatted and joined:

```
Sourdough Proofing:
After mixing, the dough rests at room temperature until it roughly doubles...

---

Chocolate Cake:
Cream the butter and sugar for a full five minutes...
```

That block goes into the prompt. Budget it: too little and the model lacks
evidence, too much and the answer is diluted while the call gets more expensive.

A typical prompt:

```
Answer the question using only the context below.
If the context does not contain the answer, say so.

CONTEXT:
{retrieved passages}

QUESTION: {question}
```

---

## Failure modes worth knowing

| Symptom | Likely cause |
|---|---|
| Good index, bad answers | Retriever is fine; the prompt is not constraining the model |
| Answers cite the wrong chunk | Titles are missing, so chunks are hard to tell apart |
| Quality drops after adding documents | A document exceeded the window and was truncated |
| Works locally, fails in production | Inner-product scoring without re-normalisation |
| Good answers, wrong citations | Overlapping chunks returning near-duplicates |

The first row is the common one. Most "RAG is inaccurate" reports are actually
prompt problems wearing a retrieval costume.

---

## The full loop, as in L15

```python
records  = eg2.index_folder(folder)          # load + chunk
vectors  = eg2.encode_documents(model, records)
query    = eg2.encode_queries(model, [question])
hits     = eg2.rank(query[0], vectors, records, k=3)
```

Six lines. Everything else in a production system is durability, filtering,
evaluation and cost.

---

## Next

Run it: [L15 — a search engine you built](../lessons/L15_mini_rag.py) ·
[matryoshka](07-matryoshka-explained.md) ·
[task prefixes](06-task-prefixes-reference.md)