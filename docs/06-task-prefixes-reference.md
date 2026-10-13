# Task prefix reference

Every `prompt_name` this model accepts, what it expands to, and when to use it.

This is the single most bug-prone page in the repo. Read the "two correct
approaches" section below before using `Document`.

---

## What a prefix is

A prefix is a short instruction glued to the front of your text, telling the
model what job the text is doing.

That is the entire mechanism. There is no special token routing and no model
internals involved — it is string concatenation:

```python
model.encode("how do I reverse a list", prompt_name="SearchQuery")
# the model receives:
#   "task: search result | query: how do I reverse a list"
```

See [L04](../lessons/L04_whats_the_prefix.py), which demonstrates this with no
model download required.

**Prefixes apply to text only.** Images, audio and video take no prefix.

---

## The full table

| `prompt_name` | Expands to |
|---|---|
| `SearchQuery` | `task: search result \| query: ` |
| `Document` | `title: none \| text: ` |
| `CodeRetrieval` | `task: code retrieval \| query: ` |
| `QuestionAnswering` | `task: question answering \| query: ` |
| `FactChecking` | `task: fact checking \| query: ` |
| `Classification` | `task: classification \| query: ` |
| `Clustering` | `task: clustering \| query: ` |
| `SentenceSimilarity` | `task: sentence similarity \| query: ` |
| `Retrieval` | `task: search result \| query: ` |
| `Retrieval-query` | `task: search result \| query: ` |
| `Retrieval-document` | `title: none \| text: ` |
| `Reranking` | `task: search result \| query: ` |
| `BitextMining` | `task: search result \| query: ` |
| `PairClassification` | `task: sentence similarity \| query: ` |
| `STS` | `task: sentence similarity \| query: ` |
| `Summarization` | `task: sentence similarity \| query: ` |
| `InstructionRetrieval` | `task: code retrieval \| query: ` |
| `MultilabelClassification` | `task: classification \| query: ` |
| `document` | `title: none \| text: ` |
| `query` | `task: search result \| query: ` |

Verified against the model's own `config_sentence_transformers.json`. A copy
lives in `eg2.PROMPTS`, and a test asserts the copy has not drifted.

### Choosing between aliases

Several names are aliases of each other. Prefer the descriptive one:

- Retrieval → `SearchQuery` + `Document` is clearer than `Retrieval-query` + `Retrieval-document`
- STS → `SentenceSimilarity`
- Several names duplicate `SentenceSimilarity`; use that one

---

## Two correct ways to encode a document

**This is the section that matters.** `Document` expands to
`"title: none | text: "`, which already contains a title marker and a text
marker. So there are exactly two valid approaches, and combining them is a bug.

### Untitled body text — use the prefix

```python
model.encode("the actual content", prompt_name="Document")
# model receives:
#   "title: none | text: the actual content"          CORRECT
```

### Titled document — format it yourself, no prefix

```python
model.encode("title: My Title | text: the actual content")
# model receives:
#   "title: My Title | text: the actual content"      CORRECT
```

### The bug — doing both

```python
model.encode("title: My Title | text: the actual content", prompt_name="Document")
# model receives:
#   "title: none | text: title: My Title | text: the actual content"   WRONG
```

No exception. No warning. You get a plausible-looking vector and quietly worse
retrieval.

**This bug shipped in this repo** and was only found by comparing against the
published model card. A regression test now guards it.

### Letting the helper decide

```python
import eg2

eg2.encode_documents(model, ["untitled body text"])                       # prefixed
eg2.encode_documents(model, [{"title": "T", "text": "body text"}])        # not prefixed
```

Strings get the `Document` prefix; dicts with titles are formatted manually.
There is no call site where both can happen.

---

## Asymmetric tasks

Most retrieval is **asymmetric**: queries and documents are different kinds of
text. Users write short questions; documents are long explanations.

| Use case | Query prefix | Document text |
|---|---|---|
| Web / document search | `SearchQuery` | `title: {title} \| text: {content}` |
| Question answering | `QuestionAnswering` | `title: {title} \| text: {passage}` |
| Fact checking | `FactChecking` | `title: {title} \| text: {evidence}` |
| Code search | `CodeRetrieval` | `title: {filename} \| text: {code}` |

Use a **query** prefix on the thing the user typed and a **document** prefix on
everything you pre-indexed.

## Symmetric tasks

When everything being compared is the same kind of text, use one prefix
everywhere:

| Use case | Prefix |
|---|---|
| Classification | `Classification` |
| Clustering | `Clustering` |
| Measuring similarity | `SentenceSimilarity` |

Comparing a query against a query with `SearchQuery` is a symmetric task that
happens to use an asymmetric prefix. It works; it is just not what the prefix
was tuned for.

---

## Does it actually matter?

Yes, measurably. [L07](../lessons/L07_prefix_matters.py) encodes the same query
three ways — `SearchQuery`, `CodeRetrieval`, and no prefix — and prints the
score against a mixed corpus for each. The ranking shifts.

The effect is not always dramatic on a toy corpus with five documents. It is
consistent, and it is the difference between retrieving the right passage and
retrieving a plausible one from the wrong section.

---

## Inspecting prefixes yourself

```python
import eg2

# No model needed -- works offline.
print(eg2.PROMPTS["Document"])
# title: none | text:

eg2.preview_input("body text", prompt_name="Document")
# prints: 'title: none | text: body text'

# See the damage:
eg2.preview_input("title: T | text: B", prompt_name="Document")
# prints: 'title: none | text: title: T | text: B'
```

---

## Next

- [L04 — what the model actually receives](../lessons/L04_whats_the_prefix.py)
- [L06 — asymmetric search end to end](../lessons/L06_asymmetric_search.py)
- [L07 — does the prefix change the answer?](../lessons/L07_prefix_matters.py)