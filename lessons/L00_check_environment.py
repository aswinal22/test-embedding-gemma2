"""
L00_check_environment.py
=======================

Lesson 00 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will be able to confirm that this machine can actually
run the model, and read a dependency version error before it becomes a
confusing stack trace from deep inside transformers.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
None. This is the first lesson.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Every model on Hugging Face is just a folder of numbers plus a description of
how those numbers fit together. Loading it needs a chain of libraries:

    PyTorch          does the arithmetic (tensors, GPU, dtypes)
      -> transformers   understands the model's architecture ("embedding_gemma2")
      -> sentence-transformers  wraps it with a nice .encode() method

If any link in that chain is the wrong version, you do not get a clear message.
You get a stack trace forty lines deep, or -- worse -- an error that only shows
up once you try to use the model.

`eg2.doctor` checks each link and tells you which one is broken.

IMPORTANT: you do NOT need a GPU for anything in this series. Every lesson runs
on a plain CPU laptop. A GPU only makes it faster.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
The published checkpoint for google/embeddinggemma-2 requires transformers
5.18.0 or newer. If you install an older one, the library has never heard of
this model's architecture and cannot load it. Checking now costs ten seconds;
discovering it through a confusing error costs twenty minutes.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L00_check_environment.py

    ================================================================
      EmbeddingGemma 2 -- Lesson 00: Can I run anything?
      Checking every dependency before we touch a 1.5 GB download
    ================================================================

      [1/4] Checking that Python is new enough...
          -> Python 3.11.15 is fine (need 3.10 or newer)

      [2/4] Checking installed packages...

        [ ok ]  Python                 3.11.15
        [FAIL]  sentence-transformers  not installed
        [info]  System RAM             16.0 GB
        [info]  Compute device         CPU only

      [3/4] Checking that this transformers knows the model...

      [4/4] Running the full doctor report...

    (your results depend on your machine; the doctor reports whatever it finds)

-----------------------------------------------------------------------------
6. RUN IT
-----------------------------------------------------------------------------
    python lessons/L00_check_environment.py

If every check passes you are ready for L01. If anything fails, the report
tells you the exact pip command to run.
"""

from __future__ import annotations

# --- Setup -----------------------------------------------------------------
# Make the `eg2` package importable when this file is run directly from a clone.
# (`pip install -e .` also works, and START_HERE.md mentions it.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import output


def main() -> int:
    """Run every environment check. Returns a process exit code."""
    total = 4
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 00: Can I run anything?",
        "Checking every dependency before we touch a 1.5 GB download",
    )

    # --- 1. Python version ---
    n += 1
    output.section("Is your Python new enough?", n)
    python_version = ".".join(str(p) for p in eg2.MIN_PYTHON)
    output.step(f"Checking that Python is {python_version} or newer...", n, total)

    import platform

    current = platform.python_version()
    if eg2.at_least(current, python_version):
        output.explain(f"Python {current} is fine (need {python_version} or newer)")
    else:
        output.error(f"Python {current} is too old. You need {python_version} or newer.")
        output.note("Install a newer Python from https://python.org and re-run this lesson.")
        return 1

    # --- 2. Installed packages ---
    n += 1
    output.section("Are the libraries installed?", n)
    output.step("Checking torch, sentence-transformers, transformers, numpy, pillow...", n, total)

    packages = [
        ("torch", "torch"),
        ("sentence_transformers", "sentence-transformers"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
    ]
    installed = {}
    for import_name, label in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            installed[label] = version
            output.step(f"{label:<22} {version}")
        except ImportError:
            installed[label] = None
            output.step(f"{label:<22} {output.paint('NOT INSTALLED', 'red')}")

    # --- 3. Does transformers know this architecture? ---
    n += 1
    output.section("Does this transformers know the model?", n)
    output.step("Asking transformers whether 'embedding_gemma2' is a model type it knows...", n, total)

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        knows_model = "embedding_gemma2" in CONFIG_MAPPING
    except Exception:
        knows_model = False

    if knows_model:
        output.explain("Yes -- transformers has the embedding_gemma2 architecture registered.")
    else:
        output.warn("No -- this transformers build has never heard of 'embedding_gemma2'.")
        output.note("This is the single most common setup failure for this model.")
        output.note("Fix it with:  python -m pip install -U 'transformers>=5.18.0'")

    # --- 4. Full doctor report ---
    n += 1
    output.section("Full environment report", n)
    output.step("Running the complete doctor check (same as: python -m eg2.doctor)...", n, total)

    report = eg2.build_report()
    eg2.print_report(report)

    # --- 7. Read your output ---
    output.section("How to read that report")

    output.table(
        ["Marker", "Meaning", "What to do"],
        [
            ["ok", "This dependency is correct.", "Nothing."],
            ["FAIL", "This dependency is wrong or missing.", "Run the pip command shown."],
            ["info", "Not wrong, just context.", "Read it, then move on."],
        ],
    )

    output.note(
        "'Compute device: CPU only' is NOT a failure.\n"
        "  Every lesson in this series is designed to run on a CPU. A GPU is\n"
        "  roughly 10-30x faster, and changes nothing about what you learn."
    )

    # --- Verdict ---
    if report.ok:
        output.section("Verdict")
        output.ok("Your environment is ready.")
        output.explain("Run the next lesson:  python lessons/L01_first_embedding.py")
        output.footer("Next: python lessons/L01_first_embedding.py")
        return 0

    output.section("Verdict")
    output.error("Your environment is not ready yet.")
    failing = ", ".join(c.name for c in report.failures)
    output.explain(f"Still failing: {failing}")
    output.note("Fix the items marked FAIL above, then re-run this lesson.")
    output.footer("Fix the FAIL lines above, then re-run: python lessons/L00_check_environment.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())