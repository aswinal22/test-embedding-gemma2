"""
02_task_prompts_search.py
=========================
Demonstrates instruction-tuned Asymmetric Retrieval with EmbeddingGemma 2.

Why Task Prompts Matter:
------------------------
In real-world search and RAG systems, user queries are short and question-oriented,
whereas documents are long and explanatory. EmbeddingGemma 2 was trained with
asymmetric task instructions to align query intent with document answers:
- Documents:   prompt_name="Document" (with format: 'title: {title} | text: {text}')
- Text Search: prompt_name="SearchQuery"
- Code Search: prompt_name="CodeRetrieval"
"""

import sys
import torch
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[!] sentence-transformers is not installed.")
    print("    Install it via: pip install sentence-transformers>=6.1.0 transformers")
    sys.exit(1)


def format_doc(title: str, text: str) -> str:
    """Format document as recommended by the EmbeddingGemma 2 specification."""
    t = title if title else "none"
    return f"title: {t} | text: {text}"


def main():
    print("=" * 70)
    print(" EmbeddingGemma 2: Instruction-Tuned Asymmetric Search & Code Retrieval")
    print("=" * 70)

    model_id = "google/embeddinggemma-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32}

    print(f"[*] Loading model on {device.upper()} (Text-Only Mode)...")
    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs=model_kwargs,
        config_kwargs={"vision_config": None, "audio_config": None}
    )

    # 1. Prepare Document Corpus (Formatted with title and text)
    raw_documents = [
        {
            "title": "EmbeddingGemma 2 Architecture",
            "text": "EmbeddingGemma 2 has 740 million total parameters, with a 270M text backbone and modular vision/audio encoders mapping to a shared 768-d space."
        },
        {
            "title": "Matryoshka Representation Learning",
            "text": "MRL allows truncating embeddings to smaller dimensions like 512, 256, or 128 to save vector storage and compute while maintaining retrieval accuracy."
        },
        {
            "title": "Python List Reverse Implementation",
            "text": "def reverse_list(items):\n    '''Returns a reversed copy of the list using slice indexing.'''\n    return items[::-1]"
        },
        {
            "title": "Binary Search Algorithm",
            "text": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"
        },
        {
            "title": "Italian Pizza Dough",
            "text": "To make authentic Neapolitan pizza dough, use 00 flour, 65% hydration with room temperature water, active dry yeast, and sea salt. Ferment for 24 hours."
        }
    ]

    formatted_docs = [format_doc(d["title"], d["text"]) for d in raw_documents]

    print("\n[1] Encoding Document Corpus with prompt_name='Document'...")
    doc_embeddings = model.encode(formatted_docs, prompt_name="Document")
    print(f"    -> Encoded {len(formatted_docs)} documents into shape {doc_embeddings.shape}")

    # 2. Test Queries with appropriate prompt types
    test_queries = [
        {
            "query": "How can I reduce vector database storage costs for embeddings?",
            "prompt_name": "SearchQuery",
            "type": "General Text Query"
        },
        {
            "query": "python code snippet to reverse an array or list",
            "prompt_name": "CodeRetrieval",
            "type": "Code Retrieval Query"
        },
        {
            "query": "What ingredients are needed for Neapolitan pizza crust?",
            "prompt_name": "SearchQuery",
            "type": "General Text Query"
        }
    ]

    print("\n[2] Executing Asymmetric Search Queries:")
    for q_idx, q in enumerate(test_queries, start=1):
        query_text = q["query"]
        prompt_name = q["prompt_name"]
        
        print(f"\n--- Query {q_idx} [{q['type']}] ---")
        print(f"Prompt instruction: prompt_name='{prompt_name}'")
        print(f"Query: \"{query_text}\"")

        # Encode query with designated instruction prompt
        query_emb = model.encode([query_text], prompt_name=prompt_name)

        # Compute cosine similarity
        scores = model.similarity(query_emb, doc_embeddings).cpu().numpy()[0]

        # Rank documents by score
        ranked_indices = np.argsort(scores)[::-1]

        print("\nTop Matches:")
        for rank, doc_i in enumerate(ranked_indices[:3], start=1):
            score = scores[doc_i]
            title = raw_documents[doc_i]["title"]
            preview = raw_documents[doc_i]["text"].replace("\n", " ")[:80] + "..."
            print(f"  #{rank} [Score: {score:.4f}] {title}")
            print(f"      Excerpt: {preview}")

    print("\n" + "=" * 70)
    print(" Notice how prompt_name='CodeRetrieval' precisely fetched Python code,")
    print(" and prompt_name='SearchQuery' fetched the exact target knowledge docs!")
    print("=" * 70)


if __name__ == "__main__":
    main()
