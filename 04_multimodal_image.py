"""
04_multimodal_image.py
======================
Demonstrates Multimodal Cross-Modal Embeddings with EmbeddingGemma 2.

Unified Vector Space:
---------------------
Unlike legacy systems that required separate models (e.g., CLIP for vision and
BERT for text), EmbeddingGemma 2 embeds text, code, images, and audio into a single,
shared 768-dimensional vector space.

This script:
1. Creates a sample image (or loads an existing image).
2. Encodes the image and multiple text candidate queries into the shared 768-d space.
3. Finds the text query that best matches the image via cosine similarity.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("[!] sentence-transformers is not installed.")
    print("    Install it via: pip install sentence-transformers>=6.1.0 transformers pillow")
    sys.exit(1)


def generate_sample_image(output_path: str = "sample_sunflower.png") -> str:
    """Generate a clean synthetic test image (sunflower in blue sky)."""
    if os.path.exists(output_path):
        return output_path

    img = Image.new("RGB", (256, 256), color=(135, 206, 235))  # Sky blue background
    draw = ImageDraw.Draw(img)

    # Draw green stem
    draw.rectangle([123, 140, 133, 256], fill=(34, 139, 34))

    # Draw yellow flower petals
    center = (128, 120)
    radius = 55
    draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius], fill=(255, 215, 0))

    # Draw dark brown center disc
    disc_radius = 28
    draw.ellipse([center[0]-disc_radius, center[1]-disc_radius, center[0]+disc_radius, center[1]+disc_radius], fill=(101, 67, 33))

    img.save(output_path)
    print(f"[*] Generated test image: '{output_path}'")
    return output_path


def main():
    print("=" * 70)
    print(" EmbeddingGemma 2: Multimodal Cross-Modal Image Retrieval Demo")
    print("=" * 70)

    model_id = "google/embeddinggemma-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32}

    # Generate or locate sample image
    image_path = generate_sample_image("sample_sunflower.png")
    test_img = Image.open(image_path)

    print(f"\n[*] Loading full multimodal model '{model_id}' on {device.upper()}...")
    print("    (Vision encoder ~170M + Text backbone ~270M enabled)")
    
    # Notice: Here we do NOT disable vision_config, allowing multimodal processing
    model = SentenceTransformer(
        model_id,
        device=device,
        model_kwargs=model_kwargs,
        trust_remote_code=True
    )

    candidate_texts = [
        "A bright yellow sunflower standing tall under a clear blue sky",
        "A sleek silver sports car racing on a nighttime highway",
        "A steaming hot slice of pepperoni pizza with melted cheese",
        "A snowy mountain peak covered in pine trees during winter"
    ]

    print("\n[1] Encoding candidate text queries:")
    for idx, text in enumerate(candidate_texts, start=1):
        print(f"    [{idx}] \"{text}\"")

    # Text queries use SearchQuery prompt
    text_embeddings = model.encode(candidate_texts, prompt_name="SearchQuery")
    print(f"    -> Text embeddings shape: {text_embeddings.shape}")

    print("\n[2] Encoding image into the unified 768-d vector space:")
    # Images are passed directly without prompt prefix
    image_embedding = model.encode([test_img])
    print(f"    -> Image embedding shape: {image_embedding.shape}")

    print("\n[3] Calculating Cross-Modal Cosine Similarity:")
    similarities = model.similarity(image_embedding, text_embeddings).cpu().numpy()[0]

    ranked_indices = np.argsort(similarities)[::-1]
    print("\nRanked Text Descriptions Matching the Image:")
    for rank, idx in enumerate(ranked_indices, start=1):
        score = similarities[idx]
        print(f"  #{rank} [Score: {score:.4f}] {candidate_texts[idx]}")

    best_match = candidate_texts[ranked_indices[0]]
    print("\n" + "=" * 70)
    print(f" Top Match: \"{best_match}\" (Score: {similarities[ranked_indices[0]]:.4f})")
    print(" Notice how the visual features directly map to the corresponding textual description!")
    print("=" * 70)


if __name__ == "__main__":
    main()
