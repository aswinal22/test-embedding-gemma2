# How a model reads text

You type a sentence. A list of numbers comes out. This is the part in between.

---

## Stage 1 — Text becomes tokens

A model cannot read letters. Text is first split into **tokens** — usually
words, sometimes parts of words:

```
"How do I reverse a list in Python?"
  ↓
["How", " do", " I", " reverse", " a", " list", " in", " Python", "?"]
  ↓
[4711, 420, 1029, 8815, 261, 1748, 306, 8721, 318]
```

Those integers are **token IDs**. They are meaningless outside this model's
vocabulary — token 8815 means nothing in general, only to the model that learned
it.

The vocabulary here has **262,144** entries. That number drives a lot of the
model's size: see stage 2.

## Stage 2 — Token IDs become vectors (the embedder)

Each ID is looked up in a big lookup table — the **embedding layer** — and comes
out as a vector.

This model has a **140M-parameter embedder**. It is *larger than the 130M
transformer that does all the actual thinking*, and that is not a mistake: with
262,144 rows in that table, the table itself dominates the parameter count.

## Stage 3 — Attention mixes context

The transformer (24 layers, hidden width 512) repeatedly updates each token's
vector so it incorporates the tokens around it. After this, the vector for
"mat" in "the cat sat on the mat" knows about the cat.

This is where meaning enters the numbers.

## Stage 4 — Pooling collapses to one vector

You have one vector per token. You need one vector per document.

This model uses **mean pooling**: average all the token vectors into a single
vector. Simple, and surprisingly hard to beat.

## Stage 5 — Projection to 768

The internal width is **512**. The output width you receive is **768**. A small
projection layer maps one to the other.

```
512 (internal)  ──projection──►  768 (your embedding)
```

This is the only step that changes the width. Everything before it is 512.

## Stage 6 — Normalisation

A final layer scales the vector to length exactly 1.0.

This matters more than it looks. Because every vector has length 1:

```
cosine similarity  ==  plain dot product
```

Which makes similarity cheap. And it is also exactly what breaks when you
truncate a vector — slicing off the tail of a unit vector leaves something that
is no longer unit length. See [L08](../lessons/L08_matryoshka.py).

---

## The whole pipeline

```
"reverse a list in Python"
        │
        ├─ tokenizer ──────────► [4711, 420, 1029, ...]      262k-entry vocab
        │
        ├─ embedder ───────────► 9 vectors, width 512         140M params
        │
        ├─ transformer ────────► 9 context-aware vectors      130M params
        │                        24 layers, width 512
        │
        ├─ mean pooling ───────► 1 vector, width 512
        │
        ├─ projection ─────────► 1 vector, width 768
        │
        └─ normalise ──────────► length exactly 1.0
                                 │
                                 ▼
                    your 768-number embedding
```

Total for text only: **270M parameters**.

---

## The context window

All of the above assumes the text fits. This model allows **8,192 tokens** at
once — around 6,000 English words, or a couple of pages.

Text longer than that is truncated. If you are feeding whole books into a single
embedding, chunk first, embed each chunk, then search the chunks.

---

## Next

Run it: [L05 — tokenizer peek](../lessons/L05_tokenizer_peek.py)

Glossary: [00-glossary.md](00-glossary.md)