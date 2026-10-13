# What is an embedding?

Two minutes. No model needed. If you read only one document in this repo, read
this one.

---

## The short version

An embedding is a list of numbers that stands in for some piece of content.

```
"The cat sat on the mat."
        ↓
[ 0.0123, -0.8841,  0.0310, ... 768 numbers ... ]
```

That is all it is. 768 ordinary floating point numbers.

## The surprising part

Nothing in that list is labelled. There is no position 42 for "cats", no
position 99 for "politeness", no coordinate you can point at and say what it
means.

And yet the model clearly knows that these two sentences are about the same
thing:

```
"How do I reverse a list in Python?"
"python code snippet to reverse an array or list"
```

They share almost no vocabulary. They get pushed toward each other anyway,
because during training the model saw millions of examples and learned that
"reverse", "array", "list" and "python" belong near one another.

So what actually moved?

**The direction of the vector.** Meaning lives in which way the whole list
points, not in any individual number.

## Why this is worth caring about

This one idea is the entire basis of:

| Application | What it does |
|---|---|
| Semantic search | Find documents by meaning, not keywords |
| RAG | Feed a language model the passages that matter |
| Recommendations | "Users who liked this also liked…" |
| Clustering | Sort 10,000 documents into themes automatically |
| Duplicate detection | Spot near-copies |
| Multimodal search | Find an image using a text description |

Every one of those works by comparing directions. None of them reads the
numbers.

## A concrete comparison

Take three sentences. No two share a word.

```
a: "How do I make sourdough bread rise properly?"
b: "What is the right proofing temperature for a starter?"
c: "Compile the kernel module before deploying to production"
```

Similarity scores:

```
a <-> b   high    same topic, zero shared vocabulary
a <-> c   low     unrelated
b <-> c   low     unrelated
```

The diagonal of that matrix — every sentence against itself — is always exactly
`1.0`. That looks like a bug the first time you see it. It is not.

## What an embedding is not

- Not a summary. You cannot read one and reconstruct the text.
- Not interpretable. There is no dimension to audit or filter.
- Not a probability. Values are not 0-to-1 and negatives are normal.
- Not deterministic-looking in the way code is. Same input, same output — but
  the output is 768 numbers with no human-readable structure.

## Next

Run the code: [L01 — your first embedding](../lessons/L01_first_embedding.py)

Then find out why no dimension means anything on its own:
[L02 — what is a dimension?](../lessons/L02_what_is_a_dimension.py)