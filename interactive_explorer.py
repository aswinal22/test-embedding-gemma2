"""
interactive_explorer.py
=======================
Interactive CLI search tool for testing queries with EmbeddingGemma 2.

Features:
- Live semantic search on a knowledge base of tech, code, and science topics.
- Toggle Matryoshka dimensions (768 vs 256) on the fly.
- Measures retrieval latency and displays similarity scores.
"""

import sys
import time
import torch
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[!] sentence-transformers is not installed.")
    print("    Install it via: pip install sentence-transformers>=6.1.0 transformers")
    sys.exit(1)


KNOWLEDGE_BASE = [
    {
        "title": "EmbeddingGemma 2 Overview",
        "text": "EmbeddingGemma 2 is Google DeepMind's open multimodal embedding model with 740M parameters and 768 output dimensions under Apache 2.0."
    },
    {
        "title": "Matryoshka Representation Learning",
        "text": "MRL allows truncating embeddings to 512, 256, or 128 dimensions to dramatically reduce memory while preserving ranking quality."
    },
    {
        "title": "Python Dictionary Comprehension",
        "text": "def invert_dict(d):\n    '''Inverts key-value pairs of a dictionary.'''\n    return {v: k for k, v in d.items()}"
    },
    {
        "title": "Quicksort Algorithm in Python",
        "text": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)"
    },
    {
        "title": "Photosynthesis in Plants",
        "text": "Plants convert sunlight, carbon dioxide, and water into glucose and oxygen through chlorophyll reactions in chloroplasts."
    },
    {
        "title": "CRISPR-Cas9 Gene Editing",
        "text": "CRISPR-Cas9 uses guide RNA molecules to target specific DNA sequences and introduce precise cuts for genetic modification."
    },
    {
        "title": "Authentic Espresso Brewing",
        "text": "Pulling espresso requires 18g finely ground coffee, 9 bars of pressure at 93C water temperature for a 36g liquid yield in 28 seconds."
    }
]


def format_doc(title: str, text: str) -> str:
    return f"title: {title} | text: {text}"


def main():
    print("=" * 70)
    print(" EmbeddingGemma 2: Live Interactive Semantic Search Explorer")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32}

    print(f"[*] Initializing model on {device.upper()} (Text-Only Mode)...")
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device=device,
        model_kwargs=model_kwargs,
        config_kwargs={"vision_config": None, "audio_config": None}
    )

    formatted_docs = [format_doc(d["title"], d["text"]) for d in KNOWLEDGE_BASE]
    print(f"[*] Pre-indexing {len(formatted_docs)} documents...")

    # Pre-index at 768d and 256d
    doc_embs_768 = model.encode(formatted_docs, prompt_name="Document", truncate_dim=768)
    doc_embs_256 = model.encode(formatted_docs, prompt_name="Document", truncate_dim=256)
    print("[*] Knowledge base indexed and ready.")
    print("\nCommands:")
    print("  Type your search query (e.g., 'how to sort numbers in python' or 'coffee brewing')")
    print("  Type 'dim 768' or 'dim 256' to change Matryoshka dimension.")
    print("  Type 'quit' or 'exit' to exit.")
    print("-" * 70)

    current_dim = 768
    active_doc_embs = doc_embs_768

    while True:
        try:
            user_input = input(f"\n[Search ({current_dim}d)] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "dim 256":
            current_dim = 256
            active_doc_embs = doc_embs_256
            print("[*] Switched Matryoshka dimension to 256 (66.7% memory savings).")
            continue
        elif user_input.lower() == "dim 768":
            current_dim = 768
            active_doc_embs = doc_embs_768
            print("[*] Switched Matryoshka dimension to 768 (full resolution).")
            continue

        # Detect code intent
        is_code = any(kw in user_input.lower() for kw in ["code", "python", "def ", "algorithm", "sort", "dict", "function"])
        prompt_name = "CodeRetrieval" if is_code else "SearchQuery"

        t0 = time.perf_counter()
        query_emb = model.encode([user_input], prompt_name=prompt_name, truncate_dim=current_dim)
        scores = model.similarity(query_emb, active_doc_embs).cpu().numpy()[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ranked_indices = np.argsort(scores)[::-1]

        print(f"\nTop Matches (using prompt '{prompt_name}' @ {current_dim}d, took {elapsed_ms:.1f}ms):")
        for rank, idx in enumerate(ranked_indices[:3], start=1):
            score = scores[idx]
            doc = KNOWLEDGE_BASE[idx]
            preview = doc["text"].replace("\n", " ")[:90] + "..."
            print(f"  #{rank} [Score: {score:.4f}] {doc['title']}")
            print(f"      {preview}")


if __name__ == "__main__":
    main()
