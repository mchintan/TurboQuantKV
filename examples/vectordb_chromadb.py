"""Example: Using TurboQuantKV with ChromaDB for compressed embedding storage.

Demonstrates how TurboQuant's quantization can compress embeddings before
storing them in ChromaDB, reducing memory footprint for large vector databases.

Requirements:
    pip install chromadb torch transformers

Author: mchintan
"""

import numpy as np
import torch

from turboquantkv.config import TurboQuantConfig
from turboquantkv.core.rotation import RotationManager
from turboquantkv.core.quantizer import PolarQuantEncoder


def create_quantizer(embedding_dim: int, n_bits: int = 4) -> PolarQuantEncoder:
    """Create a PolarQuant encoder for embedding compression.

    Args:
        embedding_dim: Dimension of embeddings (must be divisible by block_size).
        n_bits: Quantization bits (2, 3, or 4).

    Returns:
        Configured PolarQuantEncoder.
    """
    block_size = min(32, embedding_dim)
    assert embedding_dim % block_size == 0, (
        f"embedding_dim {embedding_dim} must be divisible by block_size {block_size}"
    )
    rotation = RotationManager("wht", block_size=block_size, head_dim=embedding_dim)
    return PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)


def compress_embeddings(
    embeddings: np.ndarray,
    encoder: PolarQuantEncoder,
) -> tuple[np.ndarray, dict]:
    """Compress embeddings using TurboQuant PolarQuant.

    Args:
        embeddings: (n, d) float32 array of embeddings.
        encoder: PolarQuantEncoder instance.

    Returns:
        reconstructed: (n, d) float32 array of reconstructed embeddings (for ChromaDB storage).
        stats: Dict with compression statistics.
    """
    x = torch.from_numpy(embeddings).float()

    # Quantize
    qt = encoder.encode(x)

    # Dequantize (ChromaDB needs float embeddings, but these are compressed-then-decompressed)
    x_hat = encoder.decode(qt, dtype=torch.float32)

    # Compute stats
    original_bytes = embeddings.nbytes
    compressed_bytes = (
        qt.packed_indices.nelement() * qt.packed_indices.element_size()
        + qt.norms.nelement() * qt.norms.element_size()
    )
    cosine_sim = torch.nn.functional.cosine_similarity(
        x, x_hat, dim=-1
    ).mean().item()

    stats = {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": round(original_bytes / compressed_bytes, 2),
        "cosine_similarity": round(cosine_sim, 6),
        "n_vectors": len(embeddings),
        "dim": embeddings.shape[1],
    }

    return x_hat.numpy(), stats


def main():
    try:
        import chromadb
    except ImportError:
        print("ChromaDB not installed. Install with: pip install chromadb")
        print("Showing compressed embedding workflow without ChromaDB...\n")
        chromadb = None

    # --- Configuration ---
    EMBEDDING_DIM = 128  # Common embedding dimension (e.g., from sentence-transformers)
    N_DOCUMENTS = 1000
    N_BITS = 4  # Quantization bits

    # --- Generate synthetic embeddings (replace with real model embeddings) ---
    print(f"Generating {N_DOCUMENTS} synthetic embeddings (dim={EMBEDDING_DIM})...")
    np.random.seed(42)
    embeddings = np.random.randn(N_DOCUMENTS, EMBEDDING_DIM).astype(np.float32)
    # Normalize (typical for sentence embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    documents = [f"Document {i}: This is a sample text for vector search." for i in range(N_DOCUMENTS)]
    ids = [f"doc_{i}" for i in range(N_DOCUMENTS)]

    # --- Compress embeddings with TurboQuant ---
    print(f"Compressing with TurboQuant ({N_BITS}-bit)...")
    encoder = create_quantizer(EMBEDDING_DIM, n_bits=N_BITS)
    compressed_embeddings, stats = compress_embeddings(embeddings, encoder)

    print(f"\n--- Compression Stats ---")
    print(f"  Original size:      {stats['original_bytes'] / 1024:.1f} KB")
    print(f"  Compressed size:    {stats['compressed_bytes'] / 1024:.1f} KB")
    print(f"  Compression ratio:  {stats['compression_ratio']}x")
    print(f"  Cosine similarity:  {stats['cosine_similarity']:.4f}")

    if chromadb is not None:
        # --- Store in ChromaDB ---
        print("\n--- ChromaDB Integration ---")
        client = chromadb.Client()

        # Collection with original embeddings
        original_collection = client.create_collection(
            name="original_embeddings",
            metadata={"description": "Original FP32 embeddings"},
        )
        original_collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
        )

        # Collection with compressed embeddings
        compressed_collection = client.create_collection(
            name="compressed_embeddings",
            metadata={
                "description": f"TurboQuant {N_BITS}-bit compressed embeddings",
                "compression_ratio": str(stats["compression_ratio"]),
            },
        )
        compressed_collection.add(
            embeddings=compressed_embeddings.tolist(),
            documents=documents,
            ids=ids,
        )

        # --- Query comparison ---
        query_embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compress query too for consistency
        query_compressed, _ = compress_embeddings(
            query_embedding.reshape(1, -1), encoder
        )

        print("\nQuerying both collections (top 5 results)...")
        original_results = original_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
        )
        compressed_results = compressed_collection.query(
            query_embeddings=[query_compressed[0].tolist()],
            n_results=5,
        )

        print(f"\n  Original top-5 IDs:    {original_results['ids'][0]}")
        print(f"  Compressed top-5 IDs:  {compressed_results['ids'][0]}")

        # Measure retrieval overlap
        orig_set = set(original_results["ids"][0])
        comp_set = set(compressed_results["ids"][0])
        overlap = len(orig_set & comp_set) / len(orig_set)
        print(f"  Retrieval overlap:     {overlap:.0%}")
    else:
        # Without ChromaDB, still demonstrate the compression
        print("\n--- Nearest Neighbor Search (Manual) ---")
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)

        # Search with original embeddings
        orig_scores = embeddings @ query
        orig_top5 = np.argsort(orig_scores)[-5:][::-1]

        # Search with compressed embeddings
        comp_scores = compressed_embeddings @ query
        comp_top5 = np.argsort(comp_scores)[-5:][::-1]

        print(f"  Original top-5 indices:    {orig_top5.tolist()}")
        print(f"  Compressed top-5 indices:  {comp_top5.tolist()}")
        overlap = len(set(orig_top5) & set(comp_top5)) / 5
        print(f"  Retrieval overlap:         {overlap:.0%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
