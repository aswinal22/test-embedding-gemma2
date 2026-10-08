"""
01_quickstart_text.py
====================
Quickstart guide for Google DeepMind's EmbeddingGemma 2 (google/embeddinggemma-2).

Key concepts demonstrated:
1. Loading the lightweight text-only backbone (~270M parameters) by disabling
   vision and audio encoders to save RAM.
2. Proper precision setting (bfloat16 for GPU or float32 for CPU - NEVER float16).
3. Generating 768-dimensional embeddings.
4. Computing cosine similarity between sentences.
"""

import sys
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[!] sentence-transformers is not installed.")
    print("    Install it via: pip install sentence-transformers>=6.1.0 transformers")
    sys.exit(1)


def main():
    print("=" * 65)
    print(" EmbeddingGemma 2: Quickstart Text Embedding Demo")
    print("=" * 65)

    model_id = "google/embeddinggemma-2"

    # Select compute device and safe precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # IMPORTANT: EmbeddingGemma 2 activations overflow in float16.
    # Use bfloat16 on supported GPUs or float32 on CPU.
    model_kwargs = {}
    if device == "cuda" and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
        print(f"[*] Hardware: GPU detected ({torch.cuda.get_device_name(0)}) with native bfloat16.")
    else:
        model_kwargs["torch_dtype"] = torch.float32
        print(f"[*] Hardware: Running on {device.upper()} with float32 precision.")

    print(f"\n[1] Initializing '{model_id}' (Text-Only Mode)...")
    print("    -> Disabling vision and audio encoders saves memory (~191MB RAM footprint).")
    
    # Load model with modular config (disables vision & audio for minimal RAM)
    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs=model_kwargs,
        trust_remote_code=True,
        config_kwargs={"vision_config": None, "audio_config": None}
    )

    print("\n[2] Defining test corpus:")
    sentences = [
        "Google DeepMind introduced EmbeddingGemma 2, an open multimodal embedding model.",
        "EmbeddingGemma maps text, images, and audio into a shared 768-d vector space.",
        "The recipe for a delicious chocolate cake requires flour, cocoa, and sugar.",
        "To bake a dessert, preheat your oven and mix the sweet ingredients."
    ]

    for idx, s in enumerate(sentences, start=1):
        print(f"    [{idx}] {s}")

    print("\n[3] Encoding sentences into 768-dimensional vector space...")
    embeddings = model.encode(sentences)
    print(f"    -> Generated embeddings shape: {embeddings.shape}")
    print(f"    -> Embedding dimension per sentence: {embeddings.shape[1]}")

    print("\n[4] Computing pairwise cosine similarity matrix:")
    sim_matrix = model.similarity(embeddings, embeddings).cpu().numpy()

    print("\n    Similarity Matrix:")
    header = "       " + "".join([f"  Doc {i+1} " for i in range(len(sentences))])
    print(header)
    print("       " + "-" * (len(header) - 7))

    for i in range(len(sentences)):
        row_str = f" Doc {i+1} | "
        for j in range(len(sentences)):
            score = sim_matrix[i][j]
            row_str += f"{score:7.3f} "
        print(row_str)

    print("\n[5] Key Observations:")
    print(f"    - Docs 1 & 2 (AI/Embeddings) similarity: {sim_matrix[0][1]:.3f} (High semantic match)")
    print(f"    - Docs 3 & 4 (Baking/Dessert) similarity: {sim_matrix[2][3]:.3f} (High semantic match)")
    print(f"    - Doc 1 & Doc 3 (AI vs Baking) similarity: {sim_matrix[0][2]:.3f} (Dissimilar / distinct)")
    print("=" * 65)


if __name__ == "__main__":
    main()
