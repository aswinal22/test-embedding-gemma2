# Matryoshka Representation Learning

How to make an embedding 6x smaller without breaking your search — and the one
argument you must not forget.

---

## The idea

Most embedding models fix their output width. Truncating them is a blunt
instrument: the tail of the vector holds as much signal as the head.

EmbeddingGemma 2 is trained with **Matryoshka Representation Learning** (MRL).
Russian nesting dolls contain smaller dolls inside larger ones. The model is
trained so that the **first N coordinates carry the most important information**,
which means you can stop peeling at any of the supported sizes and still have a
usable vector.

| Output | Compression | Keep |
|---|---|---|
| 768d | 1:1 | everything |
| 512d | 1:1.5 | most |
| 256d | 1:3 | nearly all, for text |
| 128d | 1:6 | text-only work |

Only these four are supported. `truncate_dim=400` is not a trained checkpoint of
this model.

---

## The catch, and it is not optional

This model ends with a **normalisation layer**, so every embedding you receive
has length exactly **1.0**. Lesson 01 shows this.

Now slice that vector down:

```python
full = model.encode("some text")[0]      # norm 1.000000
short = full[:256]                       # norm ~0.58   <-- not 1.0
```

You removed two thirds of the numbers. You did **not** remove two thirds of the
length. The remaining vector still holds its full squared magnitude spread
across fewer coordinates.

Approximate norms for a real embedding:

| dim | kept | norm after slicing |
|---|---|---|
| 768 | 100% | 1.000 |
| 512 | 67% | ~0.83 |
| 256 | 33% | ~0.59 |
| 128 | 17% | ~0.41 |

### Why that ruins scoring

A vector compared against **itself** must score 1.0. With a dot product and an
un-normalised short vector, it does not:

```
dot(v[:256], v[:256])   = 0.34...    instead of 1.0
```

That is the squared norm. Every score gets scaled down by a constant, which
squashes the score distribution and can reorder results.

**Cosine similarity hides this**, because `cos_sim` renormalises both operands
internally. This is why the bug is so easy to miss — the demo prints plausible
numbers. But a vector database configured for inner-product similarity is doing
a dot product, and so is anything you write yourself as `q @ corpus.T`.

### The fix

```python
model.encode(text, truncate_dim=256, normalize_embeddings=True)
```

Always pair the two.

### Why the library does not do it for you

`sentence-transformers` slices *after* the forward pass has already run the
model's own `Normalize` module, and re-normalises only if you ask:

```python
out_features = self(features, **kwargs)
if truncate_dim is not None:
    out_features["sentence_embedding"] = truncate_embeddings(...)
...
if normalize_embeddings:                       # defaults to False
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
```

That ordering is deliberate — it lets you decide. The cost of forgetting is
silent.

---

## Does the ranking actually survive?

That is the question, and it needs the right metric. Mean score delta is
useless: it is trivially `0.0` at 768d (the baseline compared against itself)
and says nothing about whether the *order* changed.

Use ranking metrics:

- **top-k overlap** — Jaccard between the top-k sets at dimension *d* and at 768d
- **Spearman correlation** — 1.0 means identical ordering

Measured over 24 documents and 6 queries (`lessons/L08_matryoshka.py`):

| dim | storage (1M docs) | top-3 overlap | mean Spearman |
|---|---|---|---|
| 768d | 2.86 GiB | 1.000 | 1.000 |
| 512d | 1.91 GiB | 1.000 | ~0.998 |
| 256d | 0.95 GiB | ~0.944 | ~0.981 |
| 128d | 0.48 GiB | ~0.778 | ~0.933 |

256d is the sweet spot: two thirds less storage, ranking essentially intact.

---

## The 128d trap

128d looks like the obvious win — six times smaller. It is not free.

The published benchmarks show the degradation is uneven. Text holds up far
better than images:

| Output | MTEB (code) | MIEB (image) | MMEB overall |
|---|---|---|---|
| 768d | 78.68 | 64.64 | 59.01 |
| 512d | 77.24 | 64.32 | 58.38 |
| 256d | 76.18 | 63.13 | 56.24 |
| 128d | 71.41 | 59.06 | **45.65** |

Look at the MMEB column: 59 → 46. That is a cliff, not a slope.

**256d is the safe default.** 128d is defensible for text-only retrieval if you
have measured it against your own workload.

---

## Storage arithmetic

One million documents, float32 (4 bytes per value):

```
768 * 4 * 1_000_000 = 2.86 GiB
256 * 4 * 1_000_000 = 0.95 GiB   -> 67% less
128 * 4 * 1_000_000 = 0.48 GiB   -> 83% less
```

Truncation also makes approximate-nearest-neighbour search faster, because there
are fewer dimensions to compare.

It does **not** change model memory or inference speed — the transformer still
runs and still emits 768 numbers. You are compressing the *index*, not the model.

---

## Rules to remember

1. Always pass `normalize_embeddings=True` with `truncate_dim`.
2. Query and corpus must share a dimension. A 768d query cannot score a 128d corpus.
3. Re-encode the whole index when you change dimension. Do not mix.
4. Default to 256d. Validate 128d on your own data before adopting it.
5. Measure with ranking metrics, not score deltas.

---

## Next

Run it: [L08 — truncating embeddings safely](../lessons/L08_matryoshka.py)

Source: [model card](https://huggingface.co/google/embeddinggemma-2) ·
[architecture overview](04-architecture-of-embeddinggemma2.md)