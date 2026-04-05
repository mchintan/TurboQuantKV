"""Generate publication-quality graphs from benchmark results.

Reads benchmark_results.json and produces PNG graphs for the README.

Author: mchintan
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


COLORS = {
    2: "#e94560",
    3: "#0f3460",
    4: "#16213e",
}
HATCHES = {
    2: "//",
    3: "xx",
    4: "..",
}

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_mse_and_cosine(data, output_dir: Path):
    """Create a 2-panel figure: MSE (bar) and Cosine Similarity (bar) by bit width & head_dim."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    head_dims = sorted(set(d["head_dim"] for d in data))
    bits_list = sorted(set(d["n_bits"] for d in data))

    x = np.arange(len(head_dims))
    width = 0.25

    # MSE plot
    for i, bits in enumerate(bits_list):
        vals = [next(d["relative_mse"] for d in data if d["head_dim"] == hd and d["n_bits"] == bits)
                for hd in head_dims]
        bars = ax1.bar(x + i * width - width, vals, width, label=f"{bits}-bit",
                       color=COLORS[bits], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"d={hd}" for hd in head_dims])
    _style_ax(ax1, "Relative MSE by Bit Width", "Head Dimension", "Relative MSE (lower is better)")
    ax1.legend(fontsize=10)

    # Cosine Similarity plot
    for i, bits in enumerate(bits_list):
        vals = [next(d["cosine_similarity"] for d in data if d["head_dim"] == hd and d["n_bits"] == bits)
                for hd in head_dims]
        bars = ax2.bar(x + i * width - width, vals, width, label=f"{bits}-bit",
                       color=COLORS[bits], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"d={hd}" for hd in head_dims])
    ax2.set_ylim(0.85, 1.005)
    _style_ax(ax2, "Cosine Similarity by Bit Width", "Head Dimension", "Cosine Similarity (higher is better)")
    ax2.legend(fontsize=10, loc="lower right")

    fig.tight_layout(pad=2.0)
    path = output_dir / "mse_cosine_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_compression_ratio(data, output_dir: Path):
    """Create compression ratio comparison chart (vs fp16 baseline)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    head_dims = sorted(set(d["head_dim"] for d in data))
    bits_list = sorted(set(d["n_bits"] for d in data))

    # Use seq_len=256 as representative
    target_seq = 256
    x = np.arange(len(head_dims))
    width = 0.25

    for i, bits in enumerate(bits_list):
        vals = [next(d["ratio_vs_fp16"] for d in data
                     if d["head_dim"] == hd and d["n_bits"] == bits and d["seq_len"] == target_seq)
                for hd in head_dims]
        bars = ax.bar(x + i * width - width, vals, width, label=f"{bits}-bit",
                      color=COLORS[bits], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"d={hd}" for hd in head_dims])
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="FP16 baseline (1x)")
    _style_ax(ax, "Compression Ratio vs FP16 Baseline", "Head Dimension", "Compression Ratio (higher is better)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = output_dir / "compression_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_memory_savings(data, output_dir: Path):
    """Stacked bar chart showing memory breakdown for different configs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter for head_dim=128, seq_len=512 (realistic scenario)
    target_hd = 128
    target_seq = 512

    baseline_entry = next(d for d in data
                          if d["head_dim"] == target_hd and d["seq_len"] == target_seq and d["n_bits"] == 4)
    baseline_fp16_kb = baseline_entry["baseline_fp16_bytes"] / 1024

    configs = []
    for bits in [2, 3, 4]:
        entry = next(d for d in data
                     if d["head_dim"] == target_hd and d["seq_len"] == target_seq and d["n_bits"] == bits)
        configs.append({
            "label": f"{bits}-bit TurboQuant",
            "compressed_kb": entry["compressed_bytes"] / 1024,
            "bits": bits,
        })

    labels = ["FP16 Baseline"] + [c["label"] for c in configs]
    values = [baseline_fp16_kb] + [c["compressed_kb"] for c in configs]
    colors = ["#888888"] + [COLORS[c["bits"]] for c in configs]

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f} KB", va="center", fontsize=11, fontweight="bold")

    _style_ax(ax, f"KV Cache Memory Usage (head_dim={target_hd}, seq_len={target_seq}, 8 heads)",
              "Memory (KB)", "")
    ax.invert_yaxis()

    fig.tight_layout()
    path = output_dir / "memory_savings.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_throughput(data, output_dir: Path):
    """Create throughput comparison chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    head_dims = sorted(set(d["head_dim"] for d in data))
    bits_list = sorted(set(d["n_bits"] for d in data))

    x = np.arange(len(head_dims))
    width = 0.25

    # Encode throughput
    for i, bits in enumerate(bits_list):
        vals = [next(d["encode_vectors_per_sec"] / 1000 for d in data
                     if d["head_dim"] == hd and d["n_bits"] == bits)
                for hd in head_dims]
        ax1.bar(x + i * width - width, vals, width, label=f"{bits}-bit",
                color=COLORS[bits], edgecolor="white", linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"d={hd}" for hd in head_dims])
    _style_ax(ax1, "Encode Throughput", "Head Dimension", "Thousand Vectors / sec")
    ax1.legend(fontsize=10)

    # Decode throughput
    for i, bits in enumerate(bits_list):
        vals = [next(d["decode_vectors_per_sec"] / 1000 for d in data
                     if d["head_dim"] == hd and d["n_bits"] == bits)
                for hd in head_dims]
        ax2.bar(x + i * width - width, vals, width, label=f"{bits}-bit",
                color=COLORS[bits], edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"d={hd}" for hd in head_dims])
    _style_ax(ax2, "Decode Throughput", "Head Dimension", "Thousand Vectors / sec")
    ax2.legend(fontsize=10)

    fig.tight_layout(pad=2.0)
    path = output_dir / "throughput.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_snr_comparison(data, output_dir: Path):
    """Create Signal-to-Noise Ratio chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    head_dims = sorted(set(d["head_dim"] for d in data))
    bits_list = sorted(set(d["n_bits"] for d in data))

    x = np.arange(len(bits_list))
    width = 0.3

    for i, hd in enumerate(head_dims):
        vals = [next(d["snr_db"] for d in data if d["head_dim"] == hd and d["n_bits"] == bits)
                for bits in bits_list]
        offset = (i - (len(head_dims) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"d={hd}",
                      color=["#0f3460", "#e94560"][i], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}-bit" for b in bits_list])
    _style_ax(ax, "Signal-to-Noise Ratio (SNR)", "Quantization Bit Width", "SNR (dB, higher is better)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = output_dir / "snr_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_wht_vs_random(data, output_dir: Path):
    """Compare WHT vs Random rotation quality."""
    fig, ax = plt.subplots(figsize=(10, 5))

    head_dims = sorted(set(d["head_dim"] for d in data))
    bits_list = sorted(set(d["n_bits"] for d in data))

    categories = []
    wht_vals = []
    random_vals = []

    for hd in head_dims:
        for bits in bits_list:
            categories.append(f"{bits}-bit\nd={hd}")
            wht_vals.append(next(d["cosine_similarity"] for d in data
                                  if d["head_dim"] == hd and d["n_bits"] == bits and d["rotation_type"] == "wht"))
            random_vals.append(next(d["cosine_similarity"] for d in data
                                     if d["head_dim"] == hd and d["n_bits"] == bits and d["rotation_type"] == "random"))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, wht_vals, width, label="WHT (Walsh-Hadamard)",
                   color="#0f3460", edgecolor="white")
    bars2 = ax.bar(x + width / 2, random_vals, width, label="Random Orthogonal",
                   color="#e94560", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0.85, 1.005)
    _style_ax(ax, "WHT vs Random Rotation: Cosine Similarity",
              "Configuration", "Cosine Similarity (higher is better)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    path = output_dir / "wht_vs_random.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_quality_vs_compression(mse_data, compression_data, output_dir: Path):
    """Scatter plot: compression ratio vs reconstruction quality trade-off."""
    fig, ax = plt.subplots(figsize=(9, 6))

    head_dims = sorted(set(d["head_dim"] for d in mse_data))
    bits_list = sorted(set(d["n_bits"] for d in mse_data))
    markers = {64: "o", 128: "s"}

    for hd in head_dims:
        for bits in bits_list:
            cos_sim = next(d["cosine_similarity"] for d in mse_data
                           if d["head_dim"] == hd and d["n_bits"] == bits)
            ratio = next(d["ratio_vs_fp16"] for d in compression_data
                         if d["head_dim"] == hd and d["n_bits"] == bits and d["seq_len"] == 256)

            ax.scatter(ratio, cos_sim, s=150, c=COLORS[bits], marker=markers[hd],
                       edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(f"{bits}b, d={hd}", (ratio, cos_sim),
                        textcoords="offset points", xytext=(8, 5), fontsize=8)

    # Add legend entries manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="d=64"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="d=128"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[2], markersize=10, label="2-bit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[3], markersize=10, label="3-bit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[4], markersize=10, label="4-bit"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    _style_ax(ax, "Quality vs Compression Trade-off",
              "Compression Ratio vs FP16 (higher = more compressed)",
              "Cosine Similarity (higher = better quality)")

    fig.tight_layout()
    path = output_dir / "quality_vs_compression.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main():
    results_path = Path(__file__).parent / "benchmark_results.json"
    if not results_path.exists():
        print("No benchmark_results.json found. Run run_benchmarks.py first.")
        return

    results = json.loads(results_path.read_text())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating graphs...")
    plot_mse_and_cosine(results["mse_vs_bits"], OUTPUT_DIR)
    plot_compression_ratio(results["compression_ratio"], OUTPUT_DIR)
    plot_memory_savings(results["compression_ratio"], OUTPUT_DIR)
    plot_throughput(results["throughput"], OUTPUT_DIR)
    plot_snr_comparison(results["mse_vs_bits"], OUTPUT_DIR)
    plot_wht_vs_random(results["wht_vs_random"], OUTPUT_DIR)
    plot_quality_vs_compression(results["mse_vs_bits"], results["compression_ratio"], OUTPUT_DIR)

    print(f"\nAll graphs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
