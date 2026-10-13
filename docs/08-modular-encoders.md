# Modular loading

EmbeddingGemma 2 is three networks in one checkpoint. You only have to pay for
the ones you use.

---

## The three encoders

| Encoder | Params | Converts |
|---|---|---|
| Text backbone | 270M | text, code |
| Vision | 170M | images, video frames |
| Audio | 300M | waveforms |

They are **independent**. Nothing in text embedding needs the audio tower.
Building it anyway is pure waste.

---

## The `config_kwargs` table

| Active modalities | `config_kwargs` | Params |
|---|---|---|
| Text only | `{"vision_config": None, "audio_config": None}` | 270M |
| Text and image | `{"audio_config": None}` | 440M |
| Text and audio | `{"vision_config": None}` | 570M |
| Full multimodal | `{}` | 740M |

Passing `None` means "this config section describes a model that does not
exist", so the corresponding module is never constructed.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "google/embeddinggemma-2",
    config_kwargs={"vision_config": None, "audio_config": None},
)
```

Or through the helper, which picks the right set from a modality name:

```python
import eg2

model = eg2.load_model(modalities="text")    # or "image", "audio", "full"
```

---

## What it actually saves

Measured, not guessed. Lesson [L10](../lessons/L10_modular_loading.py) loads each
configuration and reports both the parameter count and the resident-memory
delta. The arithmetic, for checking it yourself:

| Config | Params | bf16 | float32 |
|---|---|---|---|
| Text only | 270M | ~515 MiB | ~1.03 GiB |
| Text and image | 440M | ~839 MiB | ~1.64 GiB |
| Text and audio | 570M | ~1.06 GiB | ~2.13 GiB |
| Full multimodal | 740M | ~1.38 GiB | ~2.76 GiB |

Weights only, excluding activations.

### A correction

An earlier version of this repo's README claimed the 270M text backbone was
**"~191MB"**. That is wrong by roughly **2.7x**.

The error was arithmetic: 270M parameters cannot fit in 191 MB, because that
would be under one byte per parameter. At two bytes each it is 540,000,000
bytes. The README now points at measured numbers instead.

---

## What this does NOT change

**Download size is identical.** The repository ships a single
`model.safetensors` of about 1.5 GB and you fetch all of it whatever you pass.

What `config_kwargs` changes is what ends up **resident in memory** after
loading — which is what decides:

- whether the model fits on your laptop
- how many models you can hold in one process
- how many users you can serve per worker

If you are bandwidth-constrained rather than memory-constrained, this saves you
nothing.

---

## Why the text backbone is still 270M

A large part of it is not the transformer at all:

```
embedding table   262,144 x 512  =  134M parameters
transformer       24 layers       =  130M parameters
                              total = 270M
```

The vocabulary table alone is bigger than the network doing the thinking. This
is normal for models with large vocabularies, and it is why "270M text model"
does not imply "270M of computation".

---

## Choosing

| If you are... | Use |
|---|---|
| indexing documents | `modalities="text"` |
| searching images with a text query | `modalities="image"` |
| searching or matching audio | `modalities="audio"` |
| demonstrating everything | `modalities="full"` |

Note that **video uses the vision encoder** — it is sampled as frames at 1 fps
by default. There is no separate video mode.

---

## Practical notes

- Loading is slower than you might expect; the checkpoint is ~1.5 GB even for
  text-only mode.
- Free between loads with `eg2.empty_cache()` (GPU) and `gc.collect()` (CPU).
- `eg2.compare_configs()` measures all four and prints a comparison table.

---

## Next

[L10 — modular loading](../lessons/L10_modular_loading.py) ·
[architecture overview](04-architecture-of-embeddinggemma2.md) ·
[precision](05-precision-and-dtypes.md)