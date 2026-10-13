# Glossary

Every term used in `lessons/`, defined once, in plain language.

Terms are listed roughly in the order you meet them. Each links to the lesson or
doc where it is put to work.

---

## Foundations

**embedding**
A list of numbers that stands in for some content — a sentence, an image, an
audio clip. This model produces 768 of them per input.
→ [L01](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L01_first_embedding.py)

**vector**
The same thing. "Vector" is just the mathematical word for a list of numbers.
"Embedding" is what we call it when it represents meaning.

**dimension**
One slot in the list. "768-dimensional" means 768 numbers long. It does **not**
mean 768 tracked properties — there is no dimension called "cats".
→ [L02](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L02_what_is_a_dimension.py)

**vector space**
The mathematical room the vectors live in. Two vectors are "close" when they
point in similar directions. When we say embeddings are in "the same space", we
mean they are directly comparable to each other.

**cosine similarity**
A number between -1 and 1 describing the angle between two vectors. 1.0 means
identical direction, 0.0 means perpendicular, -1.0 means opposite. Ignores
length entirely.
→ [L03](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L03_similarity_matrix.py)

**dot product**
Multiply corresponding numbers and add. Unlike cosine, it is sensitive to
length — a longer vector scores higher. Usually the wrong tool for text.

**similarity vs. distance**
Similarity is "how alike" (bigger is better). Distance is "how far apart"
(smaller is better). Cosine gives similarity; the name says so.

**normalisation / L2 normalisation**
Scaling a vector so its length is exactly 1.0, without changing its direction.
This model normalises its output, which is why cosine similarity here can be
computed as a plain dot product. Truncation breaks this — see
[L08](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L08_matryoshka.py).

---

## Model internals

**token**
A chunk of text, usually a word or part of one. Models never see words.

**tokenizer**
The component that splits text into tokens and assigns each an integer ID.

**token ID**
The integer standing for a token. "Cat" might be ID 5183 in one model's
vocabulary and a completely different number in another's.

**context window**
How many tokens the model can hold at once. This model allows **8,192**,
shared across all modalities.

**hidden state**
The numbers a token turns into while passing through the model. The 768
dimensions of an embedding are a pooled, projected summary of these.

**embedding layer**
The table mapping token IDs to their initial vectors. This model has a
**140M-parameter** embedder — it is larger than the transformer itself, because
the vocabulary is 262,144 words.

**attention**
The mechanism deciding which other tokens in the sentence matter for this one.

**pooling**
Turning many per-token vectors into one vector. This model uses **mean
pooling** — average the token vectors.

**projection layer**
The final step mapping the model's internal width (**512**) out to the output
width (**768**).

**encoder / backbone**
The network that converts input into vectors. In this model there are three,
and they are loadable separately — see [modular loading](#modular-loading).

---

## This model specifically

**prefix / task prompt**
A short instruction glued to the front of your text, telling the model what job
the text is doing. It is literal string concatenation — nothing more.
→ [L04](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L04_whats_the_prefix.py)

**`prompt_name`**
The argument naming a prefix in sentence-transformers, e.g. `"SearchQuery"`.
Available names: `SearchQuery`, `Document`, `CodeRetrieval`,
`QuestionAnswering`, `FactChecking`, `Classification`, `Clustering`,
`SentenceSimilarity`.

**asymmetric retrieval**
Using *different* prefixes for queries and documents. They are different kinds
of text, and the model was trained to expect them treated differently.
→ [L06](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L06_asymmetric_search.py)

**MRL (Matryoshka Representation Learning)**
Training so the most important information sits in the earliest coordinates,
letting you chop a 768d vector down to 512, 256 or 128 and re-normalise.
Supported dimensions: **768, 512, 256, 128**.

**`truncate_dim`**
The argument that does that chopping. Always pair it with
`normalize_embeddings=True`, because slicing destroys the unit length.

**modular loading**
Passing `config_kwargs` to build only the encoders you need:

| Active modalities | `config_kwargs` | Params |
|---|---|---|
| Text only | `{"vision_config": None, "audio_config": None}` | 270M |
| Text and image | `{"audio_config": None}` | 440M |
| Text and audio | `{"vision_config": None}` | 570M |
| Full multimodal | `{}` | 740M |

**bfloat16 / float32 / float16**
Ways of storing a number. **Never float16** — this model's activations overflow
its range and it returns NaN or silently degraded vectors instead of raising.
Use bfloat16 on capable GPUs, float32 everywhere else.

**modality**
A kind of input: text, image, video, audio. This model handles all four in one
shared vector space.

---

## Applied

**RAG**
Retrieval-Augmented Generation. Search your documents for relevant passages,
then paste those passages into a language model's prompt so it answers using
your data.

**top-k retrieval**
Take the k highest-scoring documents. A vector search returns a *ranking*, and
you choose how deep to cut.

**ANN (approximate nearest neighbour)**
An index that finds similar vectors fast without checking every pair — what
real vector databases use.

**vector database**
Storage built around embeddings and similarity search (Qdrant, Milvus,
pgvector, Weaviate).

**cross-modal**
Across different modalities — comparing an image against text, for example.

**cross-lingual**
Across languages. This model supports 100+; a query in Spanish can retrieve an
English document. Covered in
[L14](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L14_cross_lingual.py).

**distributed representation**
Meaning spread across all dimensions at once, with no dimension individually
meaningful. This is *why* you cannot read an embedding to understand it.
→ [L02](https://github.com/aswinal22/test-embedding-gemma2/blob/main/lessons/L02_what_is_a_dimension.py)

---

## Still to come

Terms for the later phases — precision (L09), modular loading (L10),
multimodal and interleaving (L11–L13) — will be added to this file as those
lessons land. [L01](../lessons/L01_first_embedding.py) ·
[L02](../lessons/L02_what_is_a_dimension.py) ·
[L03](../lessons/L03_similarity_matrix.py) ·
[L04](../lessons/L04_whats_the_prefix.py)