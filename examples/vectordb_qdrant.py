"""Example: Using TurboQuantKV with Qdrant for compressed vector search.

Demonstrates how TurboQuant quantization can be used as a pre-processing
step before storing embeddings in Qdrant, enabling memory-efficient
vector similarity search with minimal accuracy loss.

Requirements:
    pip install qdrant-client torch

Author: mchintan
"""

import numpy as np
import torch

from turboquantkv.core.rotation import RotationManager
from turboquantkv.core.quantizer import PolarQuantEncoder


def create_quantizer(dim: int, n_bits: int = 4) -> PolarQuantEncoder:
    """Create a PolarQuant encoder for embedding compression."""
    block_size = min(32, dim)
    rotation = RotationManager("wht", block_size=block_size, head_dim=dim)
    return PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)


def compress_embeddings(embeddings: np.ndarray, encoder: PolarQuantEncoder) -> np.ndarray:
    """Compress embeddings through TurboQuant encode/decode cycle."""
    x = torch.from_numpy(embeddings).float()
    qt = encoder.encode(x)
    x_hat = encoder.decode(qt, dtype=torch.float32)
    return x_hat.numpy()


def main():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        print("qdrant-client not installed. Install with: pip install qdrant-client")
        print("Showing compression workflow without Qdrant...\n")
        QdrantClient = None

    # --- Configuration ---
    DIM = 128
    N_DOCUMENTS = 5000
    N_BITS = 4
    COLLECTION_ORIGINAL = "documents_original"
    COLLECTION_COMPRESSED = "documents_compressed"

    # --- Generate synthetic embeddings ---
    print(f"Generating {N_DOCUMENTS} embeddings (dim={DIM})...")
    np.random.seed(42)
    embeddings = np.random.randn(N_DOCUMENTS, DIM).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # --- Compress with TurboQuant ---
    print(f"Compressing with TurboQuant ({N_BITS}-bit)...")
    encoder = create_quantizer(DIM, n_bits=N_BITS)
    compressed = compress_embeddings(embeddings, encoder)

    # Stats
    cosine_sim = np.mean(np.sum(embeddings * compressed, axis=1))
    mse = np.mean((embeddings - compressed) ** 2)
    print(f"  Cosine similarity: {cosine_sim:.4f}")
    print(f"  MSE: {mse:.6f}")

    if QdrantClient is not None:
        # --- Qdrant Integration ---
        print("\n--- Qdrant Integration ---")
        client = QdrantClient(":memory:")  # In-memory for demo

        # Create collections
        for name in [COLLECTION_ORIGINAL, COLLECTION_COMPRESSED]:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
            )

        # Upload vectors
        print("Uploading to Qdrant...")
        batch_size = 500
        for start in range(0, N_DOCUMENTS, batch_size):
            end = min(start + batch_size, N_DOCUMENTS)
            points_orig = [
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload={"doc_id": i, "text": f"Document {i}"},
                )
                for i in range(start, end)
            ]
            points_comp = [
                PointStruct(
                    id=i,
                    vector=compressed[i].tolist(),
                    payload={"doc_id": i, "text": f"Document {i}", "quantized": True},
                )
                for i in range(start, end)
            ]
            client.upsert(collection_name=COLLECTION_ORIGINAL, points=points_orig)
            client.upsert(collection_name=COLLECTION_COMPRESSED, points=points_comp)

        # --- Query comparison ---
        print("\nRunning queries...")
        n_queries = 50
        top_k = 10
        recalls = []

        for q in range(n_queries):
            query_vec = np.random.randn(DIM).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)

            results_orig = client.search(
                collection_name=COLLECTION_ORIGINAL,
                query_vector=query_vec.tolist(),
                limit=top_k,
            )
            results_comp = client.search(
                collection_name=COLLECTION_COMPRESSED,
                query_vector=query_vec.tolist(),
                limit=top_k,
            )

            orig_ids = {r.id for r in results_orig}
            comp_ids = {r.id for r in results_comp}
            recalls.append(len(orig_ids & comp_ids) / top_k)

        mean_recall = np.mean(recalls)
        print(f"\n  Recall@{top_k}: {mean_recall:.1%}")
        print(f"  (How often compressed results match original top-{top_k})")
    else:
        # Manual search without Qdrant
        print("\n--- Manual Search Comparison ---")
        n_queries = 50
        top_k = 10
        recalls = []

        for q in range(n_queries):
            query_vec = np.random.randn(DIM).astype(np.float32)
            query_vec = query_vec / np.linalg.norm(query_vec)

            scores_orig = embeddings @ query_vec
            scores_comp = compressed @ query_vec

            top_orig = set(np.argsort(scores_orig)[-top_k:])
            top_comp = set(np.argsort(scores_comp)[-top_k:])
            recalls.append(len(top_orig & top_comp) / top_k)

        mean_recall = np.mean(recalls)
        print(f"  Recall@{top_k}: {mean_recall:.1%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
