"""Loading EmbeddingGemma 2 safely.

Two rules are encoded here so that no lesson has to repeat them:

1. **Never float16.** Gemma-style activations overflow fp16's dynamic range.
   The model returns NaN or silently degraded embeddings rather than raising,
   so the failure is easy to miss.
2. **Load only the encoders you need.** The vision and audio towers are
   independent; dropping the unused ones is how you load 270M parameters
   instead of 740M.

torch and sentence-transformers are imported *inside* the functions, not at
module scope, so that `python -m eg2.doctor` still runs on a machine where the
runtime is not installed.
"""

from __future__ import annotations

from typing import Any

from . import output
from .config import (
    FORBIDDEN_DTYPES,
    MODEL_ID,
    NO_AUDIO,
    NO_VISION,
    TEXT_ONLY,
)

_MISSING_ST_HINT = (
    "sentence-transformers is not installed. "
    "Run: python -m pip install -r requirements.txt "
    "(then verify with: python -m eg2.doctor)"
)

_MISSING_TORCH_HINT = (
    "torch is not installed, so the model cannot be loaded. "
    "Run: python -m pip install -r requirements.txt "
    "(then verify with: python -m eg2.doctor)"
)

#: Which config_kwargs to pass for a given modality request.
_LOAD_MODES = {
    "text": TEXT_ONLY,
    "image": NO_AUDIO,
    "audio": NO_VISION,
    "full": {},
}


def _sentence_transformer_class():
    """Import SentenceTransformer lazily, with an actionable error message."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_MISSING_ST_HINT) from exc
    return SentenceTransformer


def resolve_precision(precision: str | None = None) -> Any:
    """Pick a safe torch dtype.

    Never returns float16 -- see the module docstring. Callers who explicitly
    ask for float16 get an exception rather than silent NaN vectors.
    """
    # Check the forbidden dtype BEFORE importing torch. The guard needs no
    # third-party code, and on a broken environment this is the error the
    # learner should see rather than a bare ModuleNotFoundError.
    if precision is not None and str(precision).lower() in FORBIDDEN_DTYPES:
        raise ValueError(
            "float16 is not safe for EmbeddingGemma 2. Its activation range "
            "overflows fp16, producing NaN or degraded embeddings with no error. "
            "Use bfloat16 on GPUs that support it, or float32 everywhere else."
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_MISSING_TORCH_HINT) from exc

    if precision is not None:
        return getattr(torch, precision)

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def describe_hardware() -> dict[str, Any]:
    """Collect a machine report. Safe to call even if torch is missing."""
    import platform

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
    }

    try:
        import torch
    except ImportError:
        report["torch"] = None
        return report

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        report["gpu"] = None
        report["bf16_supported"] = False

    return report


def precision_label(dtype: Any) -> str:
    """Human-readable dtype name, e.g. 'bfloat16'."""
    return str(dtype).replace("torch.", "")


def load_model(
    modalities: str = "text",
    device: str | None = None,
    precision: str | None = None,
    model_id: str = MODEL_ID,
    verbose: bool = True,
    **overrides: Any,
):
    """Load EmbeddingGemma 2 with a safe dtype and only the encoders you asked for.

    Args:
        modalities: One of ``"text"``, ``"image"``, ``"audio"`` or ``"full"``.
            Selects which config_kwargs are passed, which is what controls
            whether the vision and audio towers are built at all.
        device: ``"cuda"``, ``"cpu"``, or None to autodetect.
        precision: ``"bfloat16"``, ``"float32"``, or None to autodetect.
            float16 is rejected -- it silently breaks this model.
        model_id: Override the repository id.
        verbose: Print what is being loaded, and the memory it costs.
        **overrides: Extra keyword arguments forwarded to SentenceTransformer.

    Returns:
        A loaded ``SentenceTransformer``.

    Raises:
        ImportError: If torch or sentence-transformers is missing, with the
            command needed to fix it.
        ValueError: If float16 is requested, or `modalities` is unknown.
    """
    # Validate arguments before touching any heavy dependency, so a typo
    # reports itself instead of first failing on a missing torch.
    if modalities not in _LOAD_MODES:
        raise ValueError(
            f"Unknown modalities {modalities!r}. Choose one of: "
            + ", ".join(repr(k) for k in _LOAD_MODES)
        )
    dtype = resolve_precision(precision)  # also rejects float16

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_MISSING_TORCH_HINT) from exc

    SentenceTransformer = _sentence_transformer_class()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config_kwargs = _LOAD_MODES[modalities]

    if verbose:
        output.step(f"Loading '{model_id}' from Hugging Face")
        output.step(f"  device     : {device}")
        output.step(f"  precision  : {precision_label(dtype)}")
        output.step(f"  encoders   : {modalities} mode {config_kwargs or '(all)'}")

    kwargs: dict[str, Any] = {
        "model_id_or_path": model_id,
        "device": device,
        "model_kwargs": {"torch_dtype": dtype},
        # config_kwargs is what makes encoder loading modular.
        "config_kwargs": config_kwargs,
        **overrides,
    }

    return SentenceTransformer(**kwargs)


def model_cache_path(model_id: str = MODEL_ID):
    """Where this model is cached on disk, or None if we cannot tell."""
    try:
        from huggingface_hub import constants as hf_constants
    except ImportError:
        return None
    folder = "models--" + model_id.replace("/", "--")
    import os.path

    return os.path.join(hf_constants.HF_HUB_CACHE, folder)