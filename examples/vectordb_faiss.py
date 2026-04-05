"""Example: Using TurboQuantKV with FAISS for compressed similarity search.

Demonstrates how TurboQuant can compress embeddings before indexing in FAISS,
maintaining high retrieval accuracy while reducing the memory footprint.

Requirements:
    pip install faiss-cpu torch

Author: mchintan
"""

import time

import numpy as np
import torch

from turboquantkv.core.rotation import RotationManager
from turboquantkv.core.quantizer import PolarQuantEncoder


def create_quantizer(dim: int, n_bits: int = 4) -> PolarQuantEncoder:
    """Create a PolarQuant encoder for embedding compression."""
    block_size = min(32, dim)
    rotation = RotationManager("wht", block_size=block_size, head_dim=dim)
    return PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)


def compress_batch(embeddings: np.ndarray, encoder: PolarQuantEncoder) -> np.ndarray:
    """Compress a batch of embeddings through TurboQuant encode/decode."""
    x = torch.from_numpy(embeddings).float()
    qt = encoder.encode(x)
    x_hat = encoder.decode(qt, dtype=torch.float32)
    return x_hat.numpy()


def main():
    try:
        import faiss
    except ImportError:
        print("FAISS not installed. Install with: pip install faiss-cpu")
        print("Showing compression workflow without FAISS...\n")
        faiss = None

    # --- Configuration ---
    DIM = 128
    N_VECTORS = 10_000
    N_QUERIES = 100
    TOP_K = 10

    # --- Generate data ---
    print(f"Generating {N_VECTORS} vectors (dim={DIM})...")
    np.random.seed(42)
    data = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    # L2 normalize
    data = data / np.linalg.norm(data, axis=1, keepdims=True)

    queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # --- Compress with TurboQuant ---
    for n_bits in [2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"  TurboQuant {n_bits}-bit Compression")
        print(f"{'='*60}")

        encoder = create_quantizer(DIM, n_bits=n_bits)

        # Compress database vectors
        t0 = time.perf_counter()
        compressed_data = compress_batch(data, encoder)
        compress_time = time.perf_counter() - t0

        # Measure reconstruction quality
        cosine_sims = np.sum(data * compressed_data, axis=1)  # already normalized
        mse = np.mean((data - compressed_data) ** 2)

        print(f"  Compression time:     {compress_time:.2f}s ({N_VECTORS / compress_time:.0f} vec/s)")
        print(f"  Mean cosine sim:      {np.mean(cosine_sims):.4f}")
        print(f"  MSE:                  {mse:.6f}")

        # Memory calculation
        x_t = torch.from_numpy(data[:1]).float()
        qt = encoder.encode(x_t)
        bytes_per_vec_compressed = (
            qt.packed_indices.nelement() * qt.packed_indices.element_size()
            + qt.norms.nelement() * qt.norms.element_size()
        )
        bytes_per_vec_fp32 = DIM * 4
        print(f"  Bytes/vector (FP32):  {bytes_per_vec_fp32}")
        print(f"  Bytes/vector (TQ):    {bytes_per_vec_compressed}")
        print(f"  Compression ratio:    {bytes_per_vec_fp32 / bytes_per_vec_compressed:.1f}x")

        if faiss is not None:
            # --- Build FAISS indexes ---

            # Index with original vectors
            index_original = faiss.IndexFlatIP(DIM)
            index_original.add(data)

            # Index with compressed vectors
            index_compressed = faiss.IndexFlatIP(DIM)
            index_compressed.add(compressed_data)

            # --- Search both ---
            t0 = time.perf_counter()
            D_orig, I_orig = index_original.search(queries, TOP_K)
            search_time_orig = time.perf_counter() - t0

            t0 = time.perf_counter()
            D_comp, I_comp = index_compressed.search(queries, TOP_K)
            search_time_comp = time.perf_counter() - t0

            # --- Measure recall ---
            recalls = []
            for i in range(N_QUERIES):
                orig_set = set(I_orig[i])
                comp_set = set(I_comp[i])
                recalls.append(len(orig_set & comp_set) / TOP_K)

            mean_recall = np.mean(recalls)
            print(f"\n  FAISS Results (top-{TOP_K}):")
            print(f"  Original search time: {search_time_orig * 1000:.1f}ms")
            print(f"  Compressed search:    {search_time_comp * 1000:.1f}ms")
            print(f"  Recall@{TOP_K}:            {mean_recall:.1%}")
        else:
            # Manual brute-force search without FAISS
            # Original search
            scores_orig = queries @ data.T  # (N_QUERIES, N_VECTORS)
            I_orig = np.argsort(scores_orig, axis=1)[:, -TOP_K:][:, ::-1]

            # Compressed search
            scores_comp = queries @ compressed_data.T
            I_comp = np.argsort(scores_comp, axis=1)[:, -TOP_K:][:, ::-1]

            recalls = []
            for i in range(N_QUERIES):
                orig_set = set(I_orig[i])
                comp_set = set(I_comp[i])
                recalls.append(len(orig_set & comp_set) / TOP_K)

            mean_recall = np.mean(recalls)
            print(f"\n  Brute-force Search Results (top-{TOP_K}):")
            print(f"  Recall@{TOP_K}: {mean_recall:.1%}")

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
