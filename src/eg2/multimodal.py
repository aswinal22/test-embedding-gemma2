"""Helpers for the non-text modalities.

Images, audio and video all encode into the SAME 768-dimensional space as
text, which is the whole point of a natively multimodal model. These helpers
cover the awkward parts: drawing test images without downloading anything,
building the interleaved placeholder syntax, and generating audio without
depending on an audio library.

Standard library plus Pillow. torch/transformers are imported only where
genuinely needed, so the image drawing works on a machine with no model.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Sequence

#: Placeholder tokens that mark where media belongs in the text stream.
#: Order matters: each is filled from the correspondingly named key.
PLACEHOLDERS = {
    "image": "<|image|>",
    "video": "<|video|>",
    "audio": "<|audio|>",
}

#: Published token budgets, from the model card. All modalities draw from the
#: single shared 8,192-token context window.
TOKENS_PER_IMAGE = 280
TOKENS_PER_VIDEO_FRAME = 140
TOKENS_PER_AUDIO_SECOND = 25
CONTEXT_TOKENS = 8_192


def interleave(segments: Sequence[tuple[str, str]]) -> str:
    """Build an interleaved text stream.

    Args:
        segments: Ordered ``(text, modality)`` pairs, e.g.
            ``[("Shoes. ", "text"), ("<|image|>", "image")]``.

    Returns:
        The combined string, ready to pass as the ``text`` key alongside a
        dict of media lists.

    The model reads placeholders left to right and fills each from the matching
    key, in order. Two ``<|image|>`` markers consume the first two entries of
    the ``image`` list.
    """
    parts = []
    for text, modality in segments:
        if modality == "text":
            parts.append(text)
        elif modality in PLACEHOLDERS:
            parts.append(PLACEHOLDERS[modality])
        else:
            raise ValueError(
                f"Unknown modality {modality!r}. Use 'text' or one of: "
                + ", ".join(sorted(PLACEHOLDERS))
            )
    return " ".join(parts)


def budget_for(counts: dict[str, int]) -> tuple[int, int]:
    """Return ``(total_tokens, percent_of_context)`` for a media mix."""
    total = counts.get("text_tokens", 0)
    total += counts.get("image", 0) * TOKENS_PER_IMAGE
    total += counts.get("video", 0) * TOKENS_PER_VIDEO_FRAME
    total += counts.get("audio", 0) * TOKENS_PER_AUDIO_SECOND
    return total, total / CONTEXT_TOKENS * 100


# --- Synthetic images -----------------------------------------------------
# Drawn with Pillow primitives so a lesson can run with no network and no
# bundled binary assets.


def draw_image(kind: str, size: int = 256) -> "Image.Image":  # noqa: F821
    """Draw a simple, deliberately distinctive test image.

    These are flat colour blocks, not photographs. The vision encoder was
    trained on photographs, so these are OUT OF DISTRIBUTION and results can
    be surprising -- which is itself worth teaching. Lesson L11 says so and
    shows how to test with a real photo instead.
    """
    from PIL import Image, ImageDraw

    if kind == "sunflower":
        image = Image.new("RGB", (size, size), (135, 206, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle([size * 0.48, size * 0.55, size * 0.52, size], fill=(34, 139, 34))
        draw.ellipse(
            [size * 0.28, size * 0.25, size * 0.72, size * 0.69],
            fill=(255, 215, 0),
        )
        draw.ellipse(
            [size * 0.39, size * 0.36, size * 0.61, size * 0.58],
            fill=(101, 67, 33),
        )

    elif kind == "pizza":
        image = Image.new("RGB", (size, size), (60, 42, 30))
        draw = ImageDraw.Draw(image)
        draw.ellipse([size * 0.1, size * 0.1, size * 0.9, size * 0.9], fill=(222, 184, 135))
        draw.ellipse([size * 0.18, size * 0.18, size * 0.82, size * 0.82], fill=(255, 217, 102))
        for offset in (0.34, 0.5, 0.66):
            draw.ellipse(
                [
                    size * (offset - 0.06),
                    size * (offset - 0.06),
                    size * (offset + 0.06),
                    size * (offset + 0.06),
                ],
                fill=(196, 30, 30),
            )

    elif kind == "forest":
        image = Image.new("RGB", (size, size), (186, 220, 240))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, size * 0.82, size, size], fill=(58, 110, 40))
        for offset in (0.28, 0.5, 0.72):
            top = size * 0.15
            base = size * 0.88
            half = size * 0.13
            centre = size * offset
            draw.polygon(
                [(centre, top), (centre - half, base), (centre + half, base)],
                fill=(20, 90, 40),
            )

    else:
        raise ValueError(f"Unknown image kind {kind!r}. Try: sunflower, pizza, forest")

    return image


def save_images(directory: str = ".", kinds: Sequence[str] = ("sunflower", "pizza", "forest")) -> list[str]:
    """Draw and save the test images, returning their paths."""
    paths = []
    for kind in kinds:
        path = str(Path(directory) / f"eg2_{kind}.png")
        draw_image(kind).save(path)
        paths.append(path)
    return paths


# --- Synthetic audio ------------------------------------------------------
# Written with the stdlib `wave` module so no audio library is needed to
# CREATE the file. Reading it back through the model may still require
# torchaudio, which `audio_requirements()` checks for.


def write_tone_wav(
    path: str = "eg2_tone.wav",
    seconds: float = 3.0,
    frequency: float = 440.0,
    sample_rate: int = 16_000,
    decay: bool = True,
) -> str:
    """Write a mono 16 kHz WAV containing a tone.

    16 kHz mono is exactly what the model card asks for. A decaying envelope
    gives the audio encoder something with onset and offset to work with,
    which a perfectly flat sine does not.
    """
    import math as _math

    frame_count = int(seconds * sample_rate)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            envelope = 1.0
            if decay:
                envelope = math.exp(-2.5 * index / frame_count)
            value = int(32767 * 0.4 * envelope * math.sin(2 * _math.pi * frequency * index / sample_rate))
            frames += struct.pack("<h", value)
        handle.writeframes(bytes(frames))
    return path


def write_chord_wav(path: str = "eg2_chord.wav", seconds: float = 3.0, sample_rate: int = 16_000) -> str:
    """Write a three-note chord. Stands in for 'music' in the audio lesson."""
    frequencies = (261.63, 329.63, 392.00)  # C4, E4, G4 -- a C major triad
    frame_count = int(seconds * sample_rate)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(frame_count):
            envelope = math.exp(-1.5 * index / frame_count)
            value = sum(math.sin(2 * math.pi * f * index / sample_rate) for f in frequencies) / 3.0
            frames += struct.pack("<h", int(32767 * 0.35 * envelope * value))
        handle.writeframes(bytes(frames))
    return path


def audio_requirements() -> tuple[bool, list[str]]:
    """Check whether audio can be encoded on this machine.

    Returns ``(ok, missing)``. Callers should skip rather than fail when audio
    is unavailable -- an optional modality should not block a lesson.
    """
    missing = []
    for module in ("torchaudio", "librosa"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")

    return (not missing), missing


def ffmpeg_available() -> bool:
    """Whether ffmpeg is on PATH. Video extraction needs it."""
    import shutil

    return shutil.which("ffmpeg") is not None