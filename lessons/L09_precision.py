"""
L09_precision.py
================

Lesson 09 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to explain why float16 must never be used
with this model, check for the damage yourself, and explain why bfloat16 is the
safe alternative.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 00. You know how the environment doctor picks a dtype for you.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Every floating point number in a neural network is stored in a fixed number of
bits. Three matter here:

    float32   32 bits, 4 bytes    huge range, precise
    bfloat16  16 bits, 2 bytes    SAME EXPONENT RANGE as float32, less precise
    float16   16 bits, 2 bytes    huge range, SAME as float32, barely precise

The subtlety is the EXPONENT. It controls how large or small a number you can
represent. It is not about precision at all.

    float16   exponent:  5 bits   max ~65,000, min normal ~0.00006
    bfloat16  exponent:  8 bits   max ~3.4e38,  min normal ~1e-38

bfloat16 simply spends its 16 bits the same way float32 does: keep the huge
range, give up some precision. That is exactly the trade you want for inference
on a model whose activations get large.

EmbeddingGemma 2's activations exceed float16's range. In float16 the model does
not raise. It returns NaN, or silently degraded vectors that look entirely
plausible. Nothing tells you. You find out weeks later that search quality
dropped and nobody knows why.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This is the most dangerous class of bug in the whole repo: no traceback, no
warning, no NaN check in your pipeline. The failure mode is "quality got worse
for an unknown reason", which is close to the hardest thing to debug in
production.

`eg2.load_model()` refuses float16 outright so you cannot shoot yourself in the
foot by accident.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L09_precision.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 09: Precision and why fp16 breaks
      float32 vs bfloat16 vs float16
    ================================================================

      [1/5] Encoding the same text in three dtypes ...

        dtype       NaN?   norm       vs float32
        --------------------------------------------
        float32     no     1.000000   1.000000
        bfloat16    no     1.000000   0.996...
        float16     YES    nan       nan          <-- BROKEN, silently

      [2/5] Reading that table
      -> float16 returned NaN and did not raise. That is the entire danger.

      [3/5] What eg2.load_model does about it
      -> eg2.load_model() chose float32 on this machine (CPU)
      -> eg2.load_model(precision="float16") raises ValueError, by design

      [4/5] What you should have remembered to do
      -> import torch
         eg2.load_model()                      # picks a safe dtype for you

      [5/5] Memory

    IMPORTANT: whether float16 actually produces NaN depends on your hardware.
    On CPU the overflow may or may not trigger depending on the build. The
    lesson reports what it observes rather than asserting a result, and tells
    you what to check either way.

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import output

PROBES = [
    "Gradient descent minimises loss by following the negative derivative.",
    "Sourdough starter needs regular feeding to stay active.",
    "def binary_search(arr, target): return mid",
]


def inspect(vectors) -> dict:
    """Summarise a batch of embeddings: NaN count, norms, and agreement.

    Returns everything needed to decide whether a dtype broke the model,
    without assuming in advance that it did.
    """
    flat = [float(value) for row in vectors for value in row]
    nan_count = sum(1 for value in flat if math.isnan(value))
    inf_count = sum(1 for value in flat if math.isinf(value))

    if nan_count or inf_count:
        return {
            "ok": False,
            "nan": nan_count,
            "inf": inf_count,
            "total": len(flat),
            "norm": None,
            "vector": None,
        }

    norms = [eg2.vector_norm(row) for row in vectors]
    return {
        "ok": True,
        "nan": 0,
        "inf": 0,
        "total": len(flat),
        "norm": sum(norms) / len(norms),
        "vector": [float(value) for value in vectors[0]],
    }


def agreement(a: list[float] | None, b: list[float] | None) -> float | None:
    """Cosine similarity between two first-vectors, or None if either is broken."""
    if a is None or b is None:
        return None
    return eg2.cosine(a, b)


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 09: Precision, and why fp16 breaks",
        "float32 vs bfloat16 vs float16",
    )

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        output.error(str(exc))
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output.note(f"This machine is running on {device.upper()}. Results below are measured, not assumed.")

    # =====================================================================
    n += 1
    output.section("Encode the same text in three dtypes", n)
    output.step("Loading the model three times, once per dtype...", n, total)

    candidates = ["float32", "bfloat16", "float16"]
    reports = {}

    for label in candidates:
        output.step(f"  {label} ...", n, total)
        model = None
        try:
            # Deliberately bypassing eg2.load_model's float16 guard. That guard
            # exists precisely so you never write this by accident; this lesson
            # exists so you understand why it is there.
            model = SentenceTransformer(
                eg2.MODEL_ID,
                device=device,
                model_kwargs={"torch_dtype": getattr(torch, label)},
                config_kwargs=eg2.TEXT_ONLY,
            )
            reports[label] = inspect(model.encode(PROBES))
        except Exception as exc:
            reports[label] = {
                "ok": False,
                "nan": 0,
                "inf": 0,
                "total": 0,
                "norm": None,
                "vector": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            del model
            eg2.empty_cache()

    # =====================================================================
    n += 1
    output.section("Reading the results", n)
    output.step("Comparing each dtype against float32 as ground truth...", n, total)

    reference = reports.get("float32", {}).get("vector")
    rows = []
    for label in candidates:
        report = reports[label]
        if not report["ok"]:
            detail = "raised" if report.get("error") else f"{report['nan']} NaN / {report['total']} values"
            rows.append([label, "YES" if report["nan"] else "n/a", "-", detail])
            continue
        matches = agreement(reference, report["vector"])
        rows.append(
            [
                label,
                "no",
                f"{report['norm']:.6f}",
                f"{matches:.6f}" if matches is not None else "-",
            ]
        )

    output.table(
        ["dtype", "NaN?", "mean norm", "cosine vs float32"],
        rows,
    )

    broken = [label for label in candidates if not reports[label]["ok"]]
    if broken:
        output.error(
            f"float16 is broken on this machine: {', '.join(broken)} returned NaN or raised.\n"
            "  Nothing warned you. This is the failure mode."
        )
    else:
        output.warn(
            "float16 did NOT break on this machine.\n"
            "  That is luck, not safety. Whether the overflow triggers depends on\n"
            "  your hardware, your library version, and the length of your input.\n"
            "  The rule exists because it breaks SOMEWHERE, not everywhere."
        )
        output.explain(
            "A dtype that works on your laptop and silently corrupts production\n"
            "      is the worst possible outcome: it survives testing and fails later."
        )

    # =====================================================================
    n += 1
    output.section("What eg2.load_model does about it", n)

    chosen = eg2.resolve_precision()
    output.kv(
        [
            ("auto-selected dtype", eg2.precision_label(chosen)),
            ("why", "bfloat16 if the GPU supports it, otherwise float32"),
        ]
    )

    output.step("Attempting to force float16 through the safe path...", n, total)
    try:
        eg2.resolve_precision("float16")
        output.error("float16 was accepted. The guard is not working.")
    except ValueError as exc:
        output.ok("Rejected, as intended:")
        print(f"      {exc}")

    output.explain(
        "The guard runs before torch is imported, so it fires even on a\n"
        "      machine where the model could not have loaded anyway."
    )

    # =====================================================================
    n += 1
    output.section("What you should have written", n)

    output.table(
        ["Instead of", "Write"],
        [
            [
                'SentenceTransformer(id, model_kwargs={"torch_dtype": torch.float16})',
                "eg2.load_model()",
            ],
            [
                'SentenceTransformer(id, model_kwargs={"torch_dtype": torch.bfloat16})',
                "eg2.load_model()",
            ],
            ["hardcoding a device", "eg2.load_model(modalities='text')"],
        ],
    )

    output.note(
        "If you genuinely need a specific dtype, pass precision= explicitly.\n"
        "  eg2.load_model() still refuses float16."
    )

    # =====================================================================
    n += 1
    output.section("What precision actually costs you", n)

    itemsize = 2 if "bfloat16" in str(chosen) else 4
    rows = []
    for label in candidates:
        rows.append(
            [
                label,
                f"{itemsize if label != 'float16' else 2} bytes/value",
                f"{270_000_000 * (itemsize if label != 'float16' else 2) / 1024 ** 2:.0f} MiB",
                "NEVER" if label == "float16" else "safe",
            ]
        )

    output.table(["dtype", "size", "270M text backbone", "verdict"], rows)
    output.explain(
        "bfloat16 halves memory versus float32 at the cost of some mantissa\n"
        "      precision. For embedding similarity scoring, that is a bargain."
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "bfloat16 and float16 are both 16 bits. What is the actual difference?",
        "The exponent field. bfloat16 uses 8 bits for it, the same as float32, so\n"
        "  it covers the same enormous range of magnitudes and gives up mantissa\n"
        "  precision instead. float16 keeps a wide exponent but has so few mantissa\n"
        "  bits that it saturates near 65504. For a model with large activations,\n"
        "  range is what you need, which is exactly why bfloat16 is the right pick.",
    )

    output.checkpoint(
        "Why is 'float16 worked on my machine' not evidence that float16 is fine?",
        "Because the overflow is hardware- and build-dependent. A dtype that\n"
        "  survives your laptop test can silently corrupt production, and the\n"
        "  failure produces plausible-looking vectors rather than an exception.\n"
        "  That asymmetry is why the guard in eg2.load_model() refuses it\n"
        "  outright instead of warning and letting you proceed.",
    )

    output.footer("Next: python lessons/L10_modular_loading.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())