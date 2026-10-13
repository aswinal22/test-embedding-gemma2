# Precision and dtypes

Why this model must never run in float16, and why bfloat16 is the safe
alternative.

---

## The three dtypes that matter

| dtype | bytes | exponent bits | what it gives up |
|---|---|---|---|
| `float32` | 4 | 8 | nothing, it is the reference |
| `bfloat16` | 2 | **8** | mantissa precision |
| `float16` | 2 | 5 | nothing you want to lose, but range suffers |

### The exponent is the whole story

The exponent field controls **how large or small a number you can represent**.
It has nothing to do with precision. That is the part people get backwards.

```
float16   exponent 5 bits   max ≈ 65,000     smallest normal ≈ 0.00006
bfloat16  exponent 8 bits   max ≈ 3.4e38     smallest normal ≈ 1e-38
float32   exponent 8 bits   max ≈ 3.4e38     smallest normal ≈ 1e-38
```

`bfloat16` spends its 16 bits exactly the way `float32` does — same range,
fewer mantissa bits. `float16` keeps a large range but wastes bits on mantissa
and ends up saturating around 65,004.

For inference on a model with large activations, **range is what you need**.
So bfloat16 is not a compromise; it is the correct choice.

---

## What goes wrong with float16

EmbeddingGemma 2's activation range exceeds float16's. In float16 the model:

- returns **NaN**, or
- returns **silently degraded embeddings** that look entirely plausible

It does **not raise**. Nothing warns you. The failure mode is "search quality
dropped for an unknown reason" — close to the hardest thing to debug in
production.

## What goes right with bfloat16

Safe, carries the same 8-bit exponent as float32, and **halves memory** relative
to float32.

## Which to use

```python
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
model = SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": dtype})
```

Or just let the helper decide:

```python
import eg2
model = eg2.load_model()   # bfloat16 if the GPU supports it, else float32
```

`eg2.load_model()` **raises** if you pass `precision="float16"`. That is
deliberate. It is a hard error rather than a warning because the failure mode of
ignoring the warning is invisible.

---

## Why the guard runs early

```python
eg2.resolve_precision("float16")   # ValueError, before torch is even imported
```

The check needs no third-party code, so it fires even on a machine where the
model could not have loaded anyway. On a broken environment the useful error is
"don't use float16", not `ModuleNotFoundError: No module named 'torch'`.

---

## Why "it worked on my machine" proves nothing

Whether the fp16 overflow actually triggers depends on your GPU, your library
build, and the length of your input. A dtype that passes your laptop test can
corrupt production.

This is the worst possible shape of bug: **it survives testing and fails later**.
Hence the hard error rather than a runtime warning.

Lesson [L09](../lessons/L09_precision.py) measures all three dtypes on your own
machine and reports what it actually observes, including the case where fp16
happens not to break — and explains why that is luck rather than safety.

---

## Memory cost

The text backbone is 270M parameters:

| dtype | bytes/value | 270M weights |
|---|---|---|
| `float32` | 4 | ~1.03 GiB |
| `bfloat16` | 2 | ~515 MiB |
| `float16` | 2 | ~515 MiB — **unusable** |

Same memory as fp16, and it actually works.

---

## Quick reference

| Situation | Use |
|---|---|
| NVIDIA GPU with bf16 (Ampere+) | `bfloat16` |
| Older NVIDIA GPU | `float32` |
| Apple Silicon | `float32` |
| CPU | `float32` |
| Anywhere, no thought | `eg2.load_model()` |

---

## Next

[L09 — precision](../lessons/L09_precision.py) ·
[L10 — modular loading](../lessons/L10_modular_loading.py) ·
[architecture overview](04-architecture-of-embeddinggemma2.md)