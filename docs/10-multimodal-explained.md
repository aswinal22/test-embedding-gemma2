# Multimodal explained

How text, images, video and audio end up in one shared vector space — and what
each modality costs.

---

## The core idea

EmbeddingGemma 2 maps **all four modalities** into the same 768-dimensional
space, and normalises them the same way.

```
   text ────┐
  images ───┤
  video ────┼──►  768-dimensional unit vectors  ──►  cosine similarity
  audio ────┘
```

This is what separates it from older stacks, where CLIP handled images and a
separate model handled text: two models, two indexes, two sets of bugs.

Because every modality produces the same shape of vector, **the comparison
formula does not change**. Cosine similarity from
[03-why-cosine-similarity](03-why-cosine-similarity.md) works between an image
and a sentence with no special casing.

---

## The encoders

| Encoder | Params | Input | Produces |
|---|---|---|---|
| Text backbone | 270M | text, code | 768d vector |
| Vision | 170M | images, video frames | 768d vector |
| Audio | 300M | waveforms | 768d vector |

Load only what you need — see [08-modular-encoders](08-modular-encoders.md).

---

## Token costs, and the one budget

| Modality | Cost | Max in the window |
|---|---|---|
| Text | 1 token per subword | 8,192 |
| Image | 280 tokens each | ~29 images |
| Video | 140 tokens per frame (1 fps) | ~58 frames |
| Audio | 25 tokens per second | ~327 seconds |

**All four draw from a single 8,192-token context window.** Interleaving is not
free: every media item spends budget that text would otherwise have used.

Exceeding the window **truncates silently**. No exception, no warning — you get
an embedding that describes only the beginning of your input.

### Input requirements

| Modality | Requirement |
|---|---|
| Audio | **mono, 16 kHz** |
| Video | sampled at 1 fps by default (configurable) |
| Image | default 280 soft tokens; adjustable 70–1120 |

Audio at the wrong sample rate gives wrong results, not an error.

---

## Interleaving

A single input can mix text and media. Placeholder tokens mark the position of
each item:

```python
emb = model.encode({
    "text": "Waterproof running shoes. <|image|> Breathable mesh upper. <|image|> Grip test: <|video|>",
    "image": ["shoe.jpg", "mesh.jpg"],
    "video": "demo.mp4",
})
```

Each placeholder is filled from the matching key, **left to right**. Two
`<|image|>` markers consume the first two entries of the image list, so the
order of the list and the order of the placeholders must agree.

The result is a **single** vector representing the whole thing — which is what
lets a text-only query retrieve a document that *contains* images.

> **Requires `sentence-transformers>=6.1.0`.** Older versions ignore the order
> of multimodal dict inputs and cannot put several images into a single
> embedding.

---

## Prefixes are text-only

Images, audio and video take **no prompt prefix**. They already occupy a shared
space with text, so they need no instruction telling the model what they are.

```python
model.encode("how do I reverse a list", prompt_name="SearchQuery")  # prefixed
model.encode(image)                                                 # no prefix
```

---

## Vision token budget

You can trade latency and token count for quality:

| Setting | Tokens per image | Images in window |
|---|---|---|
| Low | 70 | ~114 |
| Default | 280 | ~29 |
| High | 1120 | ~7 |

More tokens means more detail captured, which helps fine-grained visual
understanding and visual document retrieval. Use the default unless you have a
reason.

---

## Video is not a separate encoder

Video is **sampled as frames** and each frame goes through the **vision**
encoder. That means:

- disabling the vision tower disables video too
- frame rate is the knob that matters, not a video-specific model
- longer clips mean more frames, and a 58-second clip fills the window alone

---

## Practical notes

### Synthetic images are out of distribution

The vision encoder was trained on photographs. Flat colour blocks drawn with
Pillow have no texture, depth, lighting or context, and ranking can be
surprising. Test the feature with a real photo before concluding anything
about it — the lessons accept an image path for exactly this reason.

### Optional dependencies

Audio and video need extras (`torchaudio`, `librosa`, `ffmpeg`). An optional
modality should **skip cleanly**, never crash the lesson. Lesson
[L13](../lessons/L13_audio_and_video.py) demonstrates that pattern.

---

## Lessons

| Lesson | Covers |
|---|---|
| [L11](../lessons/L11_multimodal_image.py) | image ↔ text, image ↔ image |
| [L12](../lessons/L12_interleaved_multimodal.py) | `<\|image\|>` placeholders, one mixed vector |
| [L13](../lessons/L13_audio_and_video.py) | audio, video, token budgets, skipping |

---

## Next

[architecture overview](04-architecture-of-embeddinggemma2.md) ·
[modular loading](08-modular-encoders.md) ·
[why cosine similarity](03-why-cosine-similarity.md)