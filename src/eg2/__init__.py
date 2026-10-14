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
    "LOAD_CONFIG_LABELS": "eg2.config",
    "MIN_TRANSFORMERS": "eg2.config",
    "MIN_SENTENCE_TRANSFORMERS": "eg2.config",
    "MIN_TORCH": "eg2.config",
    "MIN_PYTHON": "eg2.config",
    "at_least": "eg2.config",
    "version_tuple": "eg2.config",
    "PROMPTS": "eg2.config",
    # documents
    "format_document": "eg2.documents",
    "encode_documents": "eg2.documents",
    "encode_queries": "eg2.documents",
    "preview_input": "eg2.documents",
    "resolve_prompt": "eg2.documents",
    # similarity
    "cosine": "eg2.similarity",
    "cosine_matrix": "eg2.similarity",
    "similarity_matrix": "eg2.similarity",
    "rank": "eg2.similarity",
    "rank_with_families": "eg2.similarity",
    "show_ranking": "eg2.similarity",
    "show_similarity_matrix": "eg2.similarity",
    # tokenization
    "load_tokenizer": "eg2.tokenization",
    "count_tokens": "eg2.tokenization",
    "show_tokens": "eg2.tokenization",
    "context_report": "eg2.tokenization",
    "fit_to_context": "eg2.tokenization",
    # stats
    "vector_norm": "eg2.stats",
    "dot_product": "eg2.stats",
    "column_means": "eg2.stats",
    "best_separating_dimension": "eg2.stats",
    "column_spread": "eg2.stats",
    "ranks": "eg2.stats",
    "pearson": "eg2.stats",
    "spearman": "eg2.stats",
    "topk_overlap": "eg2.stats",
    "topk_indices": "eg2.stats",
    "top1_accuracy": "eg2.stats",
    "storage_bytes": "eg2.stats",
    "storage_report": "eg2.stats",
    # memory
    "current_rss_bytes": "eg2.memory",
    "format_bytes": "eg2.memory",
    "parameter_count": "eg2.memory",
    # multimodal
    "interleave": "eg2.multimodal",
    "draw_image": "eg2.multimodal",
    "save_images": "eg2.multimodal",
    "budget_for": "eg2.multimodal",
    "write_tone_wav": "eg2.multimodal",
    "write_chord_wav": "eg2.multimodal",
    "audio_requirements": "eg2.multimodal",
    "PLACEHOLDERS": "eg2.multimodal",
    # corpus
    "load_folder": "eg2.corpus",
    "chunk_text": "eg2.corpus",
    "index_folder": "eg2.corpus",
    "summarise": "eg2.corpus",
    # loader
    "load_model": "eg2.loader",
    "describe_hardware": "eg2.loader",
    "compare_configs": "eg2.loader",
    "show_config_comparison": "eg2.loader",
    "empty_cache": "eg2.loader",
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
        LOAD_CONFIG_LABELS,
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
        PROMPTS,
        TEXT_ONLY,
        at_least,
        version_tuple,
    )
    from .doctor import build_report, print_report
    from .corpus import chunk_text, index_folder, load_folder, summarise
    from .documents import (
        encode_documents,
        encode_queries,
        format_document,
        preview_input,
        resolve_prompt,
    )
    from .loader import (
        compare_configs,
        describe_hardware,
        empty_cache,
        load_model,
        model_cache_path,
        precision_label,
        resolve_precision,
        show_config_comparison,
    )
    from .memory import current_rss_bytes, format_bytes, parameter_count
    from .similarity import (
        cosine,
        cosine_matrix,
        rank,
        rank_with_families,
        show_ranking,
        show_similarity_matrix,
        similarity_matrix,
    )
    from .stats import (
        best_separating_dimension,
        column_means,
        column_spread,
        dot_product,
        pearson,
        ranks,
        spearman,
        storage_bytes,
        storage_report,
        topk_indices,
        topk_overlap,
        top1_accuracy,
        vector_norm,
    )
    from .tokenization import (
        context_report,
        count_tokens,
        fit_to_context,
        load_tokenizer,
        show_tokens,
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