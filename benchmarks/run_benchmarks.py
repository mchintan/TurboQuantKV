"""Comprehensive benchmark suite for TurboQuantKV.

Collects comparison stats across bit widths, head dimensions, and sequence lengths.
Outputs JSON results and generates matplotlib graphs for the README.

Author: mchintan
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import numpy as np

from turboquantkv.config import TurboQuantConfig
from turboquantkv.core.rotation import RotationManager
from turboquantkv.core.quantizer import PolarQuantEncoder, TurboQuantizer
from turboquantkv.core.codebook import compute_lloyd_max_centroids
from turboquantkv.core.bitpack import pack_bits, unpack_bits


def benchmark_mse_vs_bits(head_dims=(64, 128), n_vectors=500, seed=42):
    """Measure MSE across bit widths and head dimensions."""
    torch.manual_seed(seed)
    results = []
    for head_dim in head_dims:
        for n_bits in [2, 3, 4]:
            block_size = min(32, head_dim)
            rotation = RotationManager("wht", block_size=block_size, head_dim=head_dim)
            encoder = PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)

            x = torch.randn(n_vectors, head_dim)
            qt = encoder.encode(x)
            x_hat = encoder.decode(qt, dtype=torch.float32)

            mse = ((x - x_hat) ** 2).mean().item()
            signal_power = (x ** 2).mean().item()
            relative_mse = mse / signal_power
            snr_db = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')

            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

            results.append({
                "head_dim": head_dim,
                "n_bits": n_bits,
                "mse": round(mse, 6),
                "relative_mse": round(relative_mse, 6),
                "snr_db": round(snr_db, 2),
                "cosine_similarity": round(cos_sim, 6),
            })
    return results


def benchmark_compression_ratio(head_dims=(64, 128), seq_lengths=(32, 64, 128, 256, 512)):
    """Measure actual compression ratios for different configs."""
    results = []
    for head_dim in head_dims:
        block_size = min(32, head_dim)
        for n_bits in [2, 3, 4]:
            for seq_len in seq_lengths:
                rotation = RotationManager("wht", block_size=block_size, head_dim=head_dim)
                encoder = PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)

                # Simulate KV cache: (batch=1, heads=8, seq_len, head_dim)
                x = torch.randn(1, 8, seq_len, head_dim)
                baseline_bytes = x.nelement() * x.element_size()  # fp32

                qt = encoder.encode(x)
                compressed_bytes = (
                    qt.packed_indices.nelement() * qt.packed_indices.element_size()
                    + qt.norms.nelement() * qt.norms.element_size()
                )

                ratio = baseline_bytes / compressed_bytes if compressed_bytes > 0 else 0

                # Also compute fp16 baseline
                baseline_fp16_bytes = x.nelement() * 2  # 2 bytes per fp16
                ratio_vs_fp16 = baseline_fp16_bytes / compressed_bytes if compressed_bytes > 0 else 0

                results.append({
                    "head_dim": head_dim,
                    "n_bits": n_bits,
                    "seq_len": seq_len,
                    "baseline_fp32_bytes": baseline_bytes,
                    "baseline_fp16_bytes": baseline_fp16_bytes,
                    "compressed_bytes": compressed_bytes,
                    "ratio_vs_fp32": round(ratio, 2),
                    "ratio_vs_fp16": round(ratio_vs_fp16, 2),
                })
    return results


def benchmark_throughput(head_dims=(64, 128), n_iterations=50, seed=42):
    """Measure encode/decode throughput (vectors per second)."""
    torch.manual_seed(seed)
    results = []
    for head_dim in head_dims:
        block_size = min(32, head_dim)
        for n_bits in [2, 3, 4]:
            rotation = RotationManager("wht", block_size=block_size, head_dim=head_dim)
            encoder = PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)

            # Warm up
            x = torch.randn(64, head_dim)
            for _ in range(5):
                qt = encoder.encode(x)
                encoder.decode(qt)

            # Benchmark encode
            x = torch.randn(256, head_dim)
            start = time.perf_counter()
            for _ in range(n_iterations):
                qt = encoder.encode(x)
            encode_time = (time.perf_counter() - start) / n_iterations

            # Benchmark decode
            start = time.perf_counter()
            for _ in range(n_iterations):
                encoder.decode(qt)
            decode_time = (time.perf_counter() - start) / n_iterations

            vectors_per_sec_encode = 256 / encode_time
            vectors_per_sec_decode = 256 / decode_time

            results.append({
                "head_dim": head_dim,
                "n_bits": n_bits,
                "encode_time_ms": round(encode_time * 1000, 3),
                "decode_time_ms": round(decode_time * 1000, 3),
                "encode_vectors_per_sec": round(vectors_per_sec_encode, 0),
                "decode_vectors_per_sec": round(vectors_per_sec_decode, 0),
            })
    return results


def benchmark_wht_vs_random(head_dims=(64, 128), n_vectors=500, seed=42):
    """Compare WHT vs random rotation quality."""
    torch.manual_seed(seed)
    results = []
    for head_dim in head_dims:
        block_size = min(32, head_dim)
        for n_bits in [2, 3, 4]:
            for rotation_type in ["wht", "random"]:
                rotation = RotationManager(rotation_type, block_size=block_size, head_dim=head_dim, seed=seed)
                encoder = PolarQuantEncoder(n_bits=n_bits, rotation_manager=rotation, block_size=block_size)

                x = torch.randn(n_vectors, head_dim)
                qt = encoder.encode(x)
                x_hat = encoder.decode(qt, dtype=torch.float32)

                mse = ((x - x_hat) ** 2).mean().item()
                signal_power = (x ** 2).mean().item()
                cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

                results.append({
                    "head_dim": head_dim,
                    "n_bits": n_bits,
                    "rotation_type": rotation_type,
                    "mse": round(mse, 6),
                    "relative_mse": round(mse / signal_power, 6),
                    "cosine_similarity": round(cos_sim, 6),
                })
    return results


def benchmark_qjl_effect(head_dims=(64, 128), n_vectors=500, seed=42):
    """Compare with and without QJL correction."""
    torch.manual_seed(seed)
    results = []
    for head_dim in head_dims:
        block_size = min(32, head_dim)
        for n_bits in [2, 3, 4]:
            rotation = RotationManager("wht", block_size=block_size, head_dim=head_dim)

            for use_qjl in [False, True]:
                tq = TurboQuantizer(
                    n_bits=n_bits, rotation_manager=rotation, block_size=block_size,
                    use_qjl=use_qjl, qjl_dim=64, head_dim=head_dim, seed=seed,
                )
                x = torch.randn(n_vectors, head_dim)
                qt = tq.quantize(x)
                x_hat = tq.dequantize(qt, dtype=torch.float32)

                mse = ((x - x_hat) ** 2).mean().item()
                signal_power = (x ** 2).mean().item()
                cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

                compressed_bytes = (
                    qt.packed_indices.nelement() * qt.packed_indices.element_size()
                    + qt.norms.nelement() * qt.norms.element_size()
                )
                if qt.qjl_signs is not None:
                    compressed_bytes += qt.qjl_signs.nelement() * qt.qjl_signs.element_size()

                results.append({
                    "head_dim": head_dim,
                    "n_bits": n_bits,
                    "use_qjl": use_qjl,
                    "mse": round(mse, 6),
                    "relative_mse": round(mse / signal_power, 6),
                    "cosine_similarity": round(cos_sim, 6),
                    "compressed_bytes": compressed_bytes,
                })
    return results


def benchmark_bitpack_throughput(sizes=(64, 128, 256, 512, 1024), n_iterations=100):
    """Measure bitpacking throughput for different sizes."""
    results = []
    for size in sizes:
        for n_bits in [2, 3, 4]:
            max_val = 2 ** n_bits
            indices = torch.randint(0, max_val, (256, size), dtype=torch.uint8)

            # Warm up
            for _ in range(5):
                packed = pack_bits(indices, n_bits)
                unpack_bits(packed, n_bits, size)

            # Pack
            start = time.perf_counter()
            for _ in range(n_iterations):
                packed = pack_bits(indices, n_bits)
            pack_time = (time.perf_counter() - start) / n_iterations

            # Unpack
            start = time.perf_counter()
            for _ in range(n_iterations):
                unpack_bits(packed, n_bits, size)
            unpack_time = (time.perf_counter() - start) / n_iterations

            results.append({
                "size": size,
                "n_bits": n_bits,
                "pack_time_us": round(pack_time * 1e6, 1),
                "unpack_time_us": round(unpack_time * 1e6, 1),
                "elements_per_sec_pack": round(256 * size / pack_time, 0),
                "elements_per_sec_unpack": round(256 * size / unpack_time, 0),
            })
    return results


def run_all_benchmarks():
    """Run all benchmarks and return combined results."""
    print("Running MSE vs Bits benchmark...")
    mse_results = benchmark_mse_vs_bits()

    print("Running Compression Ratio benchmark...")
    compression_results = benchmark_compression_ratio()

    print("Running Throughput benchmark...")
    throughput_results = benchmark_throughput()

    print("Running WHT vs Random benchmark...")
    wht_vs_random_results = benchmark_wht_vs_random()

    print("Running QJL Effect benchmark...")
    qjl_results = benchmark_qjl_effect()

    print("Running Bitpack Throughput benchmark...")
    bitpack_results = benchmark_bitpack_throughput()

    all_results = {
        "mse_vs_bits": mse_results,
        "compression_ratio": compression_results,
        "throughput": throughput_results,
        "wht_vs_random": wht_vs_random_results,
        "qjl_effect": qjl_results,
        "bitpack_throughput": bitpack_results,
    }

    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()
    output_path = Path(__file__).parent / "benchmark_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")
