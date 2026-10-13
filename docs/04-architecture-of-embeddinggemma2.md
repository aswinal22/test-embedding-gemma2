# Architecture of EmbeddingGemma 2

The full picture in one page. Read this before the rest of `docs/`.

Every number below is taken from the checkpoint's own `config.json` on the
Hugging Face Hub, not from marketing copy. Where the model card and the config
disagree, the config wins and the disagreement is called out.

---

## At a glance

| | |
|---|---|
| Repository | `google/embeddinggemma-2` |
| License | Apache 2.0 (ungated) |
| Total parameters | **740M** |
| Output width | **768** floats |
| Context window | **8,192 tokens**, shared by all modalities |
| Modality support | text, code, images, video, audio |
| Languages | 100+ |
| Requires | `sentence-transformers>=6.1.0`, `transformers>=5.18.0` |

---

## Where the 740M goes

```
TOTAL 740M
├── TEXT BACKBONE .................................. 270M
│   ├── transformer ............................... 130M
│   └── embedding table (embedder) ................ 140M
├── VISION ENCODER ................................ 170M
└── AUDIO ENCODER ................................ 300M
```

### The embedder is bigger than the transformer

That looks like a typo. It is not. The vocabulary has **262,144** entries, and
each one needs a learned vector of width 512:

```
262,144 × 512 = 134M parameters just for the lookup table
```

So of the 270M text-only footprint, more than half is the embedding table — not
computation, just vocabulary. This is why models with large vocabularies stay
large even when the transformer itself is small.

---

## Text backbone

| | |
|---|---|
| Layers | 24 |
| Model width (hidden) | 512 |
| Feed-forward width | 2,048 |
| Attention heads | 4 |
| Key-value heads | 2 local, 1 global |
| Attention pattern | 5 sliding : 1 full, repeating |
| Activation | gated FFN with GELU |
| Vocabulary | 262,144 |
| Pooling | mean |
| Projection | 512 → 768 |
| Normalisation | final L2 layer |

### Sliding window: config says 512, model card says 1024

The published model card table lists a sliding window of 1024 tokens. The
checkpoint's `config.json` says `"sliding_window": 512`.

The config is what actually runs, so 512 is the number to trust. Flagged here
because it is a real discrepancy in the primary sources, and because it affects
nothing you do with embeddings — it is an efficiency detail of the attention
implementation, not an output you can observe.

---

## The pipeline, end to end

```
"reverse a list in Python"
        │
        ├─ tokenizer ──────────► [4711, 420, 1029, ...]      262,144-entry vocab
        │
        ├─ embedder ───────────► 9 vectors × width 512        140M params
        │
        ├─ transformer ────────► 9 context-aware vectors      130M params
        │                        24 layers, width 512
        │
        ├─ mean pooling ───────► 1 vector × width 512
        │
        ├─ projection ─────────► 1 vector × width 768
        │
        └─ normalise ──────────► length exactly 1.0
                                 │
                                 ▼
                       your 768-number embedding
```

The only step that changes the width is the projection: 512 → 768.

The final normalisation is why cosine similarity can be computed as a plain dot
product here — and also why truncating a vector breaks things. Slicing a unit
vector leaves something that is not unit length. See
[L08](../lessons/L08_matryoshka.py).

---

## Vision encoder (170M)

| | |
|---|---|
| Layers | 16 |
| Hidden width | 768 |
| Patch size | 16 × 16 |
| Heads | 12 |
| Default cost per image | **280 tokens** |
| Adjustable range | 70 – 1120 soft tokens |

Each image becomes roughly 280 tokens in the shared context. At the default
budget that is about **29 images**; at a lower budget, up to ~114.

Video is sampled at **1 frame per second** by default and each frame goes
through this same vision encoder, costing ~140 tokens each.

---

## Audio encoder (300M)

| | |
|---|---|
| Layers | 12 |
| Hidden width | 1024 |
| Cost | **25 tokens per second** |
| Required input | mono, 16 kHz |

25 tokens/second means roughly **327 seconds** of audio fits the context
window on its own — a few minutes. Combining audio with anything else draws
from the same budget.

---

## Modular loading

The three encoders are independent. You pay only for what you load, via
`config_kwargs`:

| Active modalities | `config_kwargs` | Params |
|---|---|---|
| Text only | `{"vision_config": None, "audio_config": None}` | 270M |
| Text and image | `{"audio_config": None}` | 440M |
| Text and audio | `{"vision_config": None}` | 570M |
| Full multimodal | `{}` | 740M |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"vision_config": None, "audio_config": None},
)
```

### What this saves — measured, not guessed

| Config | Params | fp32 | bf16 |
|---|---|---|---|
| Text only | 270M | ~1.08 GB | ~540 MB |
| Text and image | 440M | ~1.76 GB | ~880 MB |
| Text and audio | 570M | ~2.28 GB | ~1.14 GB |
| Full multimodal | 740M | ~2.96 GB | ~1.48 GB |

Those are weights only, excluding activations. An earlier version of this
repo's README claimed the text backbone was "~191MB" — that is wrong by a
factor of 3 to 6 depending on dtype.

**The download is the same either way.** The repository ships a single
`model.safetensors` of about 1.5 GB. `config_kwargs` reduces what is *built in
memory*, not what is *fetched from disk*.

---

## Precision

| dtype | Use it? | Why |
|---|---|---|
| `float32` | yes | The safe default, and correct on CPU |
| `bfloat16` | yes, preferred | 8-bit exponent like fp32, half the memory of fp32 |
| `float16` | **never** | Gemma activations overflow its range |

In float16 the model returns NaN or silently degraded embeddings **without
raising**. The failure is easy to miss because nothing looks wrong.

```python
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": dtype})
```

`eg2.load_model()` applies this automatically and rejects `float16` outright.

---

## What makes the output comparable

Two properties, both load-bearing:

1. **Mean pooling** turns per-token vectors into one vector per input.
2. **Normalisation** forces every output to length exactly 1.0.

Together they mean the same concept, phrased two ways, lands at nearly the same
point regardless of length — which is what makes cosine similarity (equivalently
dot product) the correct comparison. Break either one and scores stop meaning
what you think they mean.

---

## Next

- Vocabulary in practice: [L05 — tokenizer peek](../lessons/L05_tokenizer_peek.py)
- Documents vs queries: [L06 — asymmetric search](../lessons/L06_asymmetric_search.py)
- Prefix reference: [06-task-prefixes-reference.md](06-task-prefixes-reference.md)
- Source of truth: [`config.json` on the Hub](https://huggingface.co/google/embeddinggemma-2/blob/main/config.json)