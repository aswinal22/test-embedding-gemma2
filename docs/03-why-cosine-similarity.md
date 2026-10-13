# Why cosine similarity

Embeddings only become useful when you can compare them. This is the comparison.

---

## The formula

```
cos(a, b) = (a · b) / (|a| × |b|)
```

Multiply matching numbers, add them up. Do the same for each vector's length.
Divide.

That is all. [L03](../lessons/L03_similarity_matrix.py) implements it in six
lines of plain Python and checks the result against the library.

## Reading the result

| Value | Means |
|---|---|
| `1.0` | Same direction — as similar as it gets |
| `~0.8` | Strongly related |
| `~0.3` | Loosely related |
| `~0.0` | Unrelated — perpendicular |
| `-0.5` | Opposing meanings |

**High is better.** Cosine is a similarity, not a distance.

---

## Why divide by the lengths?

Because length should not matter.

Take a vector and multiply every element by 10. Same direction, ten times
longer — the same meaning, said louder.

```
cos(a, a)   = 1.0
cos(a, 10a) = 1.0      cosine is blind to length

dot(a, a)   = 768.0
dot(a, 10a) = 7680.0   the dot product changed by 10x
```

That division is the entire job of cosine. Strip it out and you get a dot
product, which rewards longer vectors.

### Does text length leak into embeddings?

Barely — modern embedding models are trained so that a short sentence and a
long restatement of it land close together. That is the point of mean pooling
and of normalisation, and it is why cosine is the right default here.

---

## Why not something else?

**Euclidean distance** (`||a - b||`) measures how far apart two points are. It
works, and it is the right choice for spatial data. But it grows with vector
length, and cosine is what the embedding literature settles on.

**Dot product** is cheaper, and on normalised vectors it is *identical* to
cosine. The moment you truncate a vector it is no longer normalised, and the two
diverge. Always re-normalise after truncating.

---

## The diagonal is 1.0, and that is correct

Compare a vector with itself and the angle is zero, so cosine is exactly 1.0.

```
        A     B     C
   A  1.000 0.812 0.233
   B  0.812 1.000 0.298
   C  0.233 0.298 1.000
```

The `1.000` running down the diagonal is the most common source of "is this
broken?" the first time anyone sees a similarity matrix. It is not broken. Every
vector is perfectly aligned with itself.

The interesting numbers are the off-diagonal ones.

---

## In a real system

Real vector databases rarely compute 768×768 matrices. They build an ANN index
(HNSW, IVF) and return the top-k nearest neighbours by cosine — the math here,
done incrementally over millions of vectors.

For a small corpus, the plain matrix is fine and far easier to reason about. Get
comfortable with it first.

---

## Next

Run it: [L03 — similarity matrix](../lessons/L03_similarity_matrix.py)

Then see what prefixes do before you compare anything:
[L04 — what the model receives](../lessons/L04_whats_the_prefix.py)