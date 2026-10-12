"""Single source of truth for model identity and minimum dependency versions.

This module deliberately imports **nothing** outside the standard library.
`eg2/__init__.py` loads it eagerly, and the environment doctor must be able to
run even on a machine where torch is not installed yet -- otherwise the tool you
reach for when things are broken is itself broken.
"""

from __future__ import annotations

# --- Model identity -------------------------------------------------------

#: Hugging Face repository id. The 740M-parameter multimodal embedding model.
MODEL_ID = "google/embeddinggemma-2"

#: Output width of every embedding, in floats.
NATIVE_DIM = 768

#: Dimensions this model was trained to survive truncation to (Matryoshka).
MRL_DIMS = (768, 512, 256, 128)

#: Maximum tokens across all modalities sharing one context window.
CONTEXT_TOKENS = 8_192

# --- Minimum supported versions -------------------------------------------

MIN_PYTHON = (3, 10)
MIN_TORCH = "2.2.0"
MIN_SENTENCE_TRANSFORMERS = "6.1.0"

# The published checkpoint's config.json records
# "transformers_version": "5.18.0.dev0". Bumping this should be a deliberate act,
# not an accident of whatever happens to be installed. Note that the doctor
# verifies this empirically -- it checks that the `embedding_gemma2` model type
# is actually registered -- so this constant is a floor, not the real test.
MIN_TRANSFORMERS = "5.18.0"

# --- Modular loading ------------------------------------------------------
# The vision and audio encoders are independent. Dropping the ones you do not
# need is how you load 270M parameters instead of 740M.

TEXT_ONLY = {"vision_config": None, "audio_config": None}
NO_AUDIO = {"audio_config": None}
NO_VISION = {"vision_config": None}
FULL_MULTIMODAL: dict[str, None] = {}

#: Human-readable labels, used by the environment report.
LOAD_CONFIGS = (
    ("Text only", TEXT_ONLY, "270M"),
    ("Text and image", NO_AUDIO, "440M"),
    ("Text and audio", NO_VISION, "570M"),
    ("Full multimodal", FULL_MULTIMODAL, "740M"),
)

# --- Precision ------------------------------------------------------------

#: Never use float16. Gemma-style activations overflow fp16's dynamic range and
#: the model returns NaN or silently degraded vectors rather than raising.
FORBIDDEN_DTYPES = ("float16", "fp16", "half")


def version_tuple(version: str) -> tuple[int, ...]:
    """Turn '5.18.0.dev0' into (5, 18, 0) so versions can be compared.

    Handles pre-release suffixes by ignoring anything after the first
    non-numeric segment, so '2.6.0+cpu' and '6.1.0rc1' both compare sanely.
    """
    cleaned = version.split("+", 1)[0]  # strip local version, e.g. '+cpu'
    parts: list[int] = []
    for chunk in cleaned.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def at_least(version: str, minimum: str) -> bool:
    """True if `version` >= `minimum`, comparing only the numeric prefix."""
    return version_tuple(version) >= version_tuple(minimum)