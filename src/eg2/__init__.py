"""Shared helpers for the EmbeddingGemma 2 lesson series.

The package name is ``eg2`` ("embedding gemma 2"). It holds the *mechanics* --
loading the model safely, formatting documents correctly, scoring similarity --
while each lesson in ``lessons/`` holds the *teaching*.

Design rule: **silence is the enemy.** These helpers print what they are doing
by default. A learner should always be able to see what just happened.

Imports are lazy (PEP 562). ``import eg2`` therefore succeeds on a machine with
no third-party packages installed, which is what lets ``python -m eg2.doctor``
run precisely when the environment is broken.

Quick start::

    import eg2

    model = eg2.load_model()                    # text-only, safe dtype
    model  # -> SentenceTransformer
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Attribute name -> the submodule that defines it.
_EXPORTS = {
    # config
    "MODEL_ID": "eg2.config",
    "NATIVE_DIM": "eg2.config",
    "MRL_DIMS": "eg2.config",
    "CONTEXT_TOKENS": "eg2.config",
    "TEXT_ONLY": "eg2.config",
    "NO_AUDIO": "eg2.config",
    "NO_VISION": "eg2.config",
    "FULL_MULTIMODAL": "eg2.config",
    "LOAD_CONFIGS": "eg2.config",
    "MIN_TRANSFORMERS": "eg2.config",
    "MIN_SENTENCE_TRANSFORMERS": "eg2.config",
    "MIN_TORCH": "eg2.config",
    "MIN_PYTHON": "eg2.config",
    "at_least": "eg2.config",
    "version_tuple": "eg2.config",
    # loader
    "load_model": "eg2.loader",
    "describe_hardware": "eg2.loader",
    "resolve_precision": "eg2.loader",
    "precision_label": "eg2.loader",
    "model_cache_path": "eg2.loader",
    # doctor
    "build_report": "eg2.doctor",
    "print_report": "eg2.doctor",
}

__version__ = "0.1.0"

if TYPE_CHECKING:  # pragma: no cover - import-time hints for type checkers only
    from .config import (
        CONTEXT_TOKENS,
        FULL_MULTIMODAL,
        LOAD_CONFIGS,
        MRL_DIMS,
        MODEL_ID,
        MIN_PYTHON,
        MIN_SENTENCE_TRANSFORMERS,
        MIN_TORCH,
        MIN_TRANSFORMERS,
        NATIVE_DIM,
        NO_AUDIO,
        NO_VISION,
        TEXT_ONLY,
        at_least,
        version_tuple,
    )
    from .doctor import build_report, print_report
    from .loader import (
        describe_hardware,
        load_model,
        model_cache_path,
        precision_label,
        resolve_precision,
    )


def __getattr__(name: str):
    """Resolve an exported name on first access (PEP 562).

    This keeps ``import eg2`` free of third-party imports: ``eg2.load_model``
    pulls in torch only when you actually call it.
    """
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'eg2' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so later lookups skip this function
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))