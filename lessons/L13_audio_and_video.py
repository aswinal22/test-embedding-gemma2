"""
L13_audio_and_video.py
======================

Lesson 13 of the EmbeddingGemma 2 series.

-----------------------------------------------------------------------------
1. OBJECTIVE
-----------------------------------------------------------------------------
After this lesson you will know what audio and video embeddings look like, how
much context they consume, and how to run the audio part on your machine.

-----------------------------------------------------------------------------
2. PREREQUISITE
-----------------------------------------------------------------------------
Lesson 12. You know media can be interleaved into text.

-----------------------------------------------------------------------------
3. WHAT IS THIS?
-----------------------------------------------------------------------------
Text, images, video and audio all encode into the same 768-dimensional space.
Lessons 11 and 12 covered images. This one covers the other two.

AUDIO
The audio encoder runs at 25 tokens per second. A three-minute recording is
about 4,500 tokens -- more than half the entire 8,192-token window. Input must
be **mono at 16 kHz**.

VIDEO
There is no separate video encoder. Video is sampled as frames (1 frame per
second by default) and each frame goes through the VISION encoder at about 140
tokens. So a 58-second clip fills the context window on its own.

All four modalities draw from ONE shared budget. Ten seconds of audio plus
four images and you are already near the ceiling.

-----------------------------------------------------------------------------
4. WHY IT MATTERS
-----------------------------------------------------------------------------
The arithmetic is the practical lesson. People budget embeddings as if each
modality had its own allowance, and then discover at query time that their
two-minute video silently truncated.

-----------------------------------------------------------------------------
5. EXPECTED OUTPUT
-----------------------------------------------------------------------------
Run with:  python lessons/L13_audio_and_video.py

This lesson SKIPS itself cleanly if the optional audio dependencies are not
installed. It never blocks on an optional modality.

    ================================================================
      EmbeddingGemma 2 -- Lesson 13: Audio and video
      The other two modalities, and their context cost
    ================================================================

      [1/5] What this machine can do
      -> torchaudio: MISSING
      -> Skipping the audio encoding section. See the install hint below.

      [2/5] The context arithmetic (always runs)
      -> audio   : 25 tokens/second
      -> video   : 140 tokens/frame at 1 fps
      -> image   : 280 tokens each
         5s audio + 10 video frames + 2 images
         = 125 + 1400 + 560 = 2085 of 8192 tokens (25.4%)

      [3/5] Generating test audio (always runs, no dependencies)
      -> wrote eg2_tone.wav and eg2_chord.wav, mono 16 kHz

      [4/5] Video is images
      -> ffmpeg found on PATH / not found

      [5/5] Rules to remember

    NOTE: audio and video are the modalities this repo's README advertised
    but never demonstrated. This lesson is that demonstration -- and it skips
    rather than failing when the optional dependencies are absent.

-----------------------------------------------------------------------------
6. THE CODE
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eg2
from eg2 import multimodal, output


def main() -> int:
    total = 5
    n = 0

    output.banner(
        "EmbeddingGemma 2 -- Lesson 13: Audio and video",
        "The other two modalities, and what they cost",
    )

    # --- 1. Capabilities ---
    n += 1
    output.section("What this machine can do", n)

    audio_ok, missing = multimodal.audio_requirements()
    ffmpeg = multimodal.ffmpeg_available()

    output.kv(
        [
            ("audio encoding", "available" if audio_ok else "unavailable"),
            ("missing", ", ".join(missing) if missing else "nothing"),
            ("ffmpeg (for video frames)", "found" if ffmpeg else "not found"),
        ]
    )

    if not audio_ok:
        output.warn(
            "Audio encoding needs optional dependencies that are not installed.\n"
            "  Install with:  python -m pip install torchaudio librosa"
        )
        output.explain(
            "Skipping the audio section rather than failing. An optional modality\n"
            "      should not block the lesson -- and the arithmetic below is the part\n"
            "      that actually matters for planning."
        )

    # --- 2. The arithmetic (always runs) ---
    n += 1
    output.section("The context arithmetic", n)
    output.step("This section needs no dependencies at all...", n, total)

    output.table(
        ["Modality", "Cost", "Budget"],
        [
            ["text", "1 token per subword", "8,192 tokens"],
            ["image", f"{multimodal.TOKENS_PER_IMAGE} tokens each", "~29 images"],
            ["video", f"{multimodal.TOKENS_PER_VIDEO_FRAME} tokens per frame (1 fps)", "~58 frames"],
            ["audio", f"{multimodal.TOKENS_PER_AUDIO_SECOND} tokens per second", "~327 seconds"],
        ],
    )

    # A realistic mixed document, which is where budgeting actually goes wrong.
    mixed = {"text_tokens": 30, "image": 2, "video": 10, "audio": 5}
    tokens, percent = multimodal.budget_for(mixed)
    audio_tokens = mixed["audio"] * multimodal.TOKENS_PER_AUDIO_SECOND
    video_tokens = mixed["video"] * multimodal.TOKENS_PER_VIDEO_FRAME
    image_tokens = mixed["image"] * multimodal.TOKENS_PER_IMAGE

    output.table(
        ["Input", "Tokens", "Share"],
        [
            ["5 seconds of audio", audio_tokens, f"{audio_tokens / tokens * 100:.1f}%"],
            ["10 video frames", video_tokens, f"{video_tokens / tokens * 100:.1f}%"],
            ["2 images", image_tokens, f"{image_tokens / tokens * 100:.1f}%"],
            ["~30 tokens of text", mixed["text_tokens"], f"{mixed['text_tokens'] / tokens * 100:.1f}%"],
            ["TOTAL", tokens, f"{percent:.1f}% of 8192"],
        ],
    )

    output.explain(
        "One budget for everything. Interleaving is not free -- every media item\n"
        "      spends context that text would otherwise have had."
    )

    over_budget, over_pct = multimodal.budget_for({"video": 90})
    output.warn(
        f"90 video frames would need {over_budget} tokens ({over_pct:.0f}% of the window).\n"
        "  Anything past the limit is SILENTLY TRUNCATED -- no exception, and the\n"
        "  embedding then describes only the first part of your video."
    )

    # --- 3. Generate test audio ---
    n += 1
    output.section("Generating test audio", n)
    output.step("Writing WAV files with the standard library...", n, total)

    tone = multimodal.write_tone_wav("eg2_tone.wav", seconds=3.0)
    chord = multimodal.write_chord_wav("eg2_chord.wav", seconds=3.0)

    output.kv(
        [
            (tone, "mono 16 kHz, 3 seconds"),
            (chord, "C major triad, mono 16 kHz, 3 seconds"),
            ("cost", f"~{3 * multimodal.TOKENS_PER_AUDIO_SECOND} tokens each"),
        ]
    )
    output.explain(
        "Generated with Python's built-in `wave` module, so creating them needs\n"
        "      no audio library at all."
    )

    # --- 4. Audio encoding, if possible ---
    if audio_ok:
        n += 1
        output.section("Encoding audio", n)
        output.step("Loading the full multimodal model...", n, total)
        try:
            model = eg2.load_model(modalities="audio")
            audio_embeddings = model.encode([chord])
            output.kv([("shape", f"{audio_embeddings.shape}   <- audio in the same 768d space")])

            probes = ["a musical chord", "a dog barking", "a car engine"]
            probe_embeddings = eg2.encode_queries(model, probes, verbose=False)
            scores = [eg2.cosine(probe_embeddings[i], audio_embeddings[0]) for i in range(len(probes))]
            best = max(range(len(probes)), key=lambda i: scores[i])
            output.table(["Text probe", "score"], [[probes[i], f"{scores[i]:.4f}"] for i in sorted(range(len(probes)), key=lambda i: scores[i], reverse=True)])
            output.explain(f"Highest: {probes[best]!r}")
        except Exception as exc:
            output.warn(f"Audio encoding failed: {type(exc).__name__}: {exc}")

    # --- 5. Video ---
    n += 1
    output.section("Video is images", n)

    output.explain(
        "There is no video encoder. Video is sampled as frames -- 1 frame per\n"
        "      second by default -- and each frame goes through the VISION encoder.\n"
        "      So the 'video' path in config_kwargs is the same 170M vision tower."
    )

    if ffmpeg:
        output.kv([("ffmpeg", "found -- extract frames with: ffmpeg -i clip.mp4 -vf fps=1 frame_%03d.png")])
        output.note(
            "Video quality is governed by the frame sampling rate and by the vision\n"
            "  token budget, which is adjustable from 70 to 1120 soft tokens per frame."
        )
    else:
        output.warn(
            "ffmpeg is not on PATH, so frames cannot be extracted.\n"
            "  Install ffmpeg, or skip video and encode frames as images directly --\n"
            "  which is the same code path."
        )

    output.section("Rules to remember", 5)
    output.bullet(
        [
            "Audio must be mono at 16 kHz. Anything else gives wrong results.",
            "Budget video in FRAMES, not seconds: 1 fps by default.",
            "All modalities share one 8,192-token budget.",
            "Exceeding it truncates silently -- check before encoding.",
            "Audio and video are optional dependencies; skip cleanly, never crash.",
        ]
    )

    # --- 9. Checkpoint ---
    output.section("Checkpoint")

    output.checkpoint(
        "How much of the 8,192-token window does 30 seconds of audio take?",
        "About 750 tokens, at 25 per second -- roughly 9% of the window. But it is\n"
        "  5,400 tokens for a full three minutes, which is most of the window. Audio\n"
        "  is the modality most likely to overflow, because seconds accumulate fast.",
    )

    output.checkpoint(
        "Is there a separate video encoder?",
        "No. Video is sampled as frames at 1 frame per second by default and each\n"
        "  frame goes through the vision encoder at about 140 tokens. Disabling the\n"
        "  vision tower disables video too, since there is nothing else running it.",
    )

    output.footer("Next: python lessons/L14_cross_lingual.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())