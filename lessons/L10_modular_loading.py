"""
L10_modular_loading.py
======================

Lesson 10 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to load only the encoders you need, and
explain with measured numbers what that saves.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 09. You know which dtype to use, so the memory numbers mean something.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
EmbeddingGemma 2 is three separate networks in one checkpoint:

    text backbone     270M    the 24-layer transformer plus its embedding table
    vision encoder    170M    turns images into tokens
    audio encoder     300M    turns waveforms into tokens

They are independent. Nothing in text embedding needs the audio tower, and
building it anyway is pure waste. `config_kwargs` tells transformers which parts
to construct:

    text only        {"vision_config": None, "audio_config": None}     270M
    text and image   {"audio_config": None}                            440M
    text and audio   {"vision_config": None}                           570M
    everything       {}                                                740M

Passing None means "this config section describes a model that does not exist",
so the corresponding module is never built.

Here is the thing worth internalising:

    THIS CHANGES MEMORY, NOT DOWNLOAD SIZE.

The repository ships one `model.safetensors` of roughly 1.5 GB. You fetch that
file regardless of config_kwargs. What you save is what gets RESIDING in RAM
once loading finishes -- which is what determines whether you can fit on a
laptop, run two models at once, or serve several users per process.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
This lesson exists because this repo's README used to claim the 270M text
backbone was "~191MB". It is not. 270M parameters at two bytes each is 540
million bytes, and no framing changes that. Lesson 10 measures the real number
and replaces the guess.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L10_modular_loading.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 10: Load only what you need
      Measure the real cost of each encoder configuration
    ================================================================

      [1/4] Loading the model four times and measuring each ...

      Configuration      config_kwargs                       Params  Weights
      ----------------------------------------------------------------------------
      Text only          {'vision_config': None, ...}          270M   ~540 MiB
      Text and image     {'audio_config': None}                440M   ~880 MiB
      Text and audio     {'vision_config': None}               570M   ~1.1 GiB
      Full multimodal    {}                                    740M   ~1.4 GiB

      [2/4] Reading those numbers
      -> The old README claimed 191MB for the text backbone. It is ~540 MiB.
      -> The mistake was assuming a 270M-parameter model fits in 191MB, which
         would work out to under one byte per parameter.

      [3/4] The arithmetic, so you can check the measurement
         270M x 2 bytes = 540,000,000 bytes = 504 MiB
         740M x 2 bytes = 1,480,000,000 bytes = 1.38 GiB

      [4/4] Choosing one for your use case

    NOTE: this loads the full 740M checkpoint four times, so it needs roughly
    1.5 GB of disk for the download and several minutes on first run. It is the
    most expensive lesson in the series.

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import memory, output


def main() -> int:
    total = 4
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 10: Load only what you need",
        "Measure the real cost of each encoder configuration",
    )

    output.note(
        "This lesson loads the full 740M checkpoint four times. It is the most\n"
        "  expensive lesson in the series: expect a ~1.5 GB download on first run\n"
        "  and several minutes of loading."
    )

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        output.error(str(exc))
        return 1

    # =====================================================================
    n += 1
    output.section("Measure every configuration", n)
    output.step("Loading four times, freeing between runs...", n, total)

    baseline = memory.current_rss_bytes()
    output.kv([("RSS before any load", memory.format_bytes(baseline))])
    print()

    results = eg2.compare_configs(verbose=False)

    if not results or all(row.get("params", 0) == 0 for row in results):
        output.error("No configuration loaded successfully. Run lesson 00 to check the environment.")
        return 1

    eg2.show_config_comparison(results)

    # =====================================================================
    n += 1
    output.section("Reading those numbers", n)

    text_row = next((r for r in results if r["modalities"] == "text"), None)
    full_row = next((r for r in results if r["modalities"] == "full"), None)

    if text_row and text_row.get("params"):
        output.kv(
            [
                ("measured text-backbone params", f"{text_row['params']:,}"),
                ("dtype", text_row["dtype"]),
                ("expected weights", memory.format_bytes(text_row["expected_bytes"])),
                ("measured RSS delta", memory.format_bytes(text_row["rss_delta_bytes"])),
            ]
        )
        output.error(
            "This repo's README used to claim the text backbone was \"~191MB\".\n"
            f"  Measured: {memory.format_bytes(text_row['expected_bytes'])}.\n"
            "  The mistake was assuming a 270M-parameter model fits in 191MB, which\n"
            "  works out to under one byte per parameter. It is roughly 3x wrong."
        )

    if text_row and full_row and full_row.get("params"):
        ratio = full_row["params"] / text_row["params"]
        output.explain(
            f"Loading everything costs {ratio:.2f}x the parameters of text only.\n"
            "      If you never touch an image or a sound file, that is pure waste."
        )

    # =====================================================================
    n += 1
    output.section("The arithmetic, so you can check the measurement", n)

    dtype_bytes = 2 if any(r["dtype"] == "bfloat16" for r in results) else 4
    rows = []
    for row in results:
        if not row.get("params"):
            continue
        millions = row["params"] / 1_000_000
        rows.append(
            [
                row["label"],
                f"{millions:.0f}M",
                f"{dtype_bytes} bytes",
                f"{row['params'] * dtype_bytes:,}",
                memory.format_bytes(row["params"] * dtype_bytes),
            ]
        )

    output.table(["Configuration", "Params", "per value", "bytes", "human"], rows)
    output.explain(
        "Multiplication. There is no hidden factor, no compression and no\n"
        "      clever trick -- which is exactly why 191MB was never plausible."
    )

    # =====================================================================
    n += 1
    output.section("Choosing one for your use case", n)

    output.table(
        ["If you are...", "Use", "Why"],
        [
            ["indexing documents", 'modalities="text"', "cheapest, 270M"],
            ["searching images with text", 'modalities="image"', "440M, skips audio"],
            ["searching audio", 'modalities="audio"', "570M, skips vision"],
            ["demoing everything", 'modalities="full"', "740M"],
        ],
    )

    output.section("The part that does NOT change", 4)
    output.warn(
        "config_kwargs does not change how much you DOWNLOAD.\n"
        "  The repository ships a single model.safetensors of about 1.5 GB and you\n"
        "  fetch all of it whatever you ask for. What you save is resident memory\n"
        "  AFTER loading -- which is what decides whether you fit on a laptop, can\n"
        "  run two models in one process, or can serve several users at once."
    )

    output.footer("That is Phase 4 complete. Next: Phase 5 -- multimodal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())