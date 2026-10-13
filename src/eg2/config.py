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

# --- Task prompt prefixes -------------------------------------------------
# Verbatim copy of the `prompts` table in the model's
# config_sentence_transformers.json on the Hugging Face Hub.
#
# Why this is duplicated here rather than read from the model: lesson L04 has
# to show you what a prompt does to your text, and it should be able to do that
# before you have downloaded 1.5 GB of weights. tests/test_smoke.py asserts
# this table still matches the real model (skipped if the model is absent).
#
# READ THIS LINE CAREFULLY -- it is the single biggest footgun in this repo:
#   "Document" already begins with "title: none | text: "
# So `model.encode("title: X | text: Y", prompt_name="Document")` produces
#   "title: none | text: title: X | text: Y"
# which is not what anyone wants. See docs/06-task-prefixes-reference.md.
PROMPTS = {
    "BitextMining": "task: search result | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Document": "title: none | text: ",
    "FactChecking": "task: fact checking | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "Reranking": "task: search result | query: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Retrieval-query": "task: search result | query: ",
    "STS": "task: sentence similarity | query: ",
    "SearchQuery": "task: search result | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
    "document": "title: none | text: ",
    "query": "task: search result | query: ",
}

#: Prefix used by images, video and audio. None -- media takes no prompt.
NO_PROMPT_MODALITIES = ("image", "audio", "video")

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

#: (modalities key, config_kwargs, advertised size). Used by
#: `loader.compare_configs()` to measure the real footprint of each.
LOAD_CONFIGS = (
    ("text", TEXT_ONLY, "270M"),
    ("image", NO_AUDIO, "440M"),
    ("audio", NO_VISION, "570M"),
    ("full", FULL_MULTIMODAL, "740M"),
)

#: Human-readable names, matching the modalities keys above.
LOAD_CONFIG_LABELS = {
    "text": "Text only",
    "image": "Text and image",
    "audio": "Text and audio",
    "full": "Full multimodal",
}

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