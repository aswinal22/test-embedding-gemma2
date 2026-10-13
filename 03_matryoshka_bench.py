"""
03_matryoshka_bench.py
======================
Benchmarks Matryoshka Representation Learning (MRL) in EmbeddingGemma 2.

What is Matryoshka Representation Learning?
-------------------------------------------
Like Russian nesting dolls, MRL trains the model so that the most essential
semantic information is concentrated in the earliest vector coordinates.
You can truncate the standard 768-dimensional vector to 512, 256, or 128 dimensions.

Benefits:
- Massive Vector DB RAM/Disk Savings (66.7% savings at 256d, 83.3% at 128d).
- Faster Nearest-Neighbor (ANN) Search & Lower Latency.
- Retains over 95%+ of retrieval accuracy!

Known issues being addressed in later phases of PLAN.md:
  - Documents below are pre-formatted with "title: X | text: Y", so they must
    NOT also pass prompt_name="Document" (see lessons/L04).
  - All truncate_dim calls now pass normalize_embeddings=True, which truncation
    requires (see lessons/L08).
  - The mean-score-delta metric this originally reported was trivially 0.0 at
    768d and said nothing about ranking, so top-3 overlap and rank correlation
    replaced it.
  - One query and four documents is still far too small a sample. The caveat is
    printed at the end; lessons/L08_matryoshka.py repeats this properly.
"""

import math
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


def main():
    print("=" * 75)
    print(" EmbeddingGemma 2: Matryoshka Representation Learning (MRL) Benchmark")
    print("=" * 75)

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

    query = "How do neural embedding models represent unstructured text?"
    documents = [
        "title: Representation Learning | text: Neural models transform textual tokens into dense vectors capturing latent semantic relationships in high-dimensional space.",
        "title: Vector Search Engines | text: Databases like Milvus, Qdrant, and Pinecone index vector embeddings using HNSW graphs for sub-millisecond similarity search.",
        "title: French Baguette Recipe | text: Authentic French baguettes use flour, water, yeast, and salt with long cold poolish fermentation.",
        "title: Quantum Computing | text: Qubits utilize superposition and quantum entanglement to execute parallel state computations."
    ]

    dimensions = [768, 512, 256, 128]
    results = {}

    print(f"\n[1] Testing query against {len(documents)} documents across MRL dimensions: {dimensions}...")

    # No prompt_name: these documents already carry "title: X | text: Y".
    # Adding prompt_name="Document" would prepend a second "title: none | text: ".
    # normalize_embeddings=True is REQUIRED alongside truncate_dim. Truncation
    # slices the vector but does not scale it, so the result comes back shorter
    # than 1.0 and every score is scaled down. See lessons/L08_matryoshka.py.
    q_emb_768 = model.encode([query], prompt_name="SearchQuery", truncate_dim=768, normalize_embeddings=True)
    d_emb_768 = model.encode(documents, truncate_dim=768, normalize_embeddings=True)
    baseline_scores = model.similarity(q_emb_768, d_emb_768).cpu().numpy()[0]

    for dim in dimensions:
        t0 = time.perf_counter()
        q_emb = model.encode([query], prompt_name="SearchQuery", truncate_dim=dim, normalize_embeddings=True)
        d_emb = model.encode(documents, truncate_dim=dim, normalize_embeddings=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        scores = model.similarity(q_emb, d_emb).cpu().numpy()[0]
        bytes_per_vector = dim * 4  # 4 bytes per float32
        savings_pct = (1.0 - (dim / 768)) * 100

        # Mean absolute difference compared to 768-d baseline
        mean_diff = float(np.mean(np.abs(scores - baseline_scores)))

        # Mean absolute difference is a weak metric for retrieval: it is
        # trivially 0.0 at 768d (the baseline compared against itself) and says
        # nothing about whether the ORDER of results survived. These two do.
        baseline_order = list(np.argsort(baseline_scores)[::-1])
        candidate_order = list(np.argsort(scores)[::-1])
        top3_overlap = len(set(baseline_order[:3]) & set(candidate_order[:3])) / len(set(baseline_order[:3]) | set(candidate_order[:3]))
        rank_correlation = _spearman(baseline_scores, scores)

        results[dim] = {
            "dim": dim,
            "bytes": bytes_per_vector,
            "savings_pct": savings_pct,
            "elapsed_ms": elapsed_ms,
            "scores": scores,
            "mean_diff": mean_diff,
            "top3_overlap": top3_overlap,
            "spearman": rank_correlation
        }

    # Print summary table
    print("\n" + "=" * 90)
    print(f" {'Dim':<8} | {'Bytes/Vec':<10} | {'Savings':<12} | {'Time':<12} | {'Top-3 overlap':<14} | {'Rank corr.':<11}")
    print("-" * 90)
    for dim in dimensions:
        r = results[dim]
        print(f" {r['dim']:<8} | {r['bytes']:<10} | {r['savings_pct']:>6.1f}%     | {r['elapsed_ms']:>7.1f} ms  | {r['top3_overlap']:>10.3f}     | {r['spearman']:>7.3f}")
    print("=" * 90)

    print("\n[2] Document Ranking Comparison by Dimension:")
    doc_titles = ["Representation Learning", "Vector Search Engines", "French Baguette", "Quantum Computing"]
    for dim in dimensions:
        scores = results[dim]["scores"]
        top_idx = int(np.argmax(scores))
        print(f"    - At {dim:3d}d: Top Match -> '{doc_titles[top_idx]}' (Score: {scores[top_idx]:.4f})")

    print("\n[3] Takeaway:")
    print("    Top-3 overlap and rank correlation are the metrics that matter here, not mean")
    print("    score delta. 256d keeps ranking essentially intact at two thirds less storage.")
    print()
    print("    CAVEAT: this script uses ONE query against FOUR documents, which is far too")
    print("    small a sample to support any conclusion. lessons/L08_matryoshka.py repeats")
    print("    this properly over 24 documents and 6 queries, and shows where 128d starts")
    print("    to degrade -- the published benchmarks have multimodal MMEB falling 59 -> 46.")
    print("=" * 90)


def _spearman(a, b) -> float:
    """Rank correlation between two score lists. 1.0 means identical ordering."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            stop = start
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
                stop += 1
            average = (start + stop) / 2.0 + 1.0
            for position in range(start, stop + 1):
                result[order[position]] = average
            start = stop + 1
        return result

    x, y = ranks(list(a)), ranks(list(b))
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


if __name__ == "__main__":
    main()
