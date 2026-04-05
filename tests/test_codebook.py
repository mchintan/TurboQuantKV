"""Tests for Lloyd-Max codebook computation."""

import torch
import pytest

from turboquantkv.core.codebook import compute_lloyd_max_centroids, get_codebook, CodebookManager


class TestLloydMax:
    def test_centroids_sorted(self):
        """Centroids should be sorted ascending."""
        for bits in [2, 3, 4]:
            centroids, boundaries = compute_lloyd_max_centroids(bits, block_size=32)
            assert (centroids[1:] > centroids[:-1]).all(), f"Centroids not sorted for {bits}-bit"

    def test_boundaries_sorted(self):
        """Boundaries should be sorted ascending, spanning [-1, 1]."""
        centroids, boundaries = compute_lloyd_max_centroids(3, block_size=32)
        assert boundaries[0] == -1.0
        assert boundaries[-1] == 1.0
        assert (boundaries[1:] > boundaries[:-1]).all()

    def test_correct_count(self):
        """Should produce 2^n_bits centroids and 2^n_bits+1 boundaries."""
        for bits in [2, 3, 4]:
            centroids, boundaries = compute_lloyd_max_centroids(bits, block_size=32)
            assert centroids.shape[0] == 2 ** bits
            assert boundaries.shape[0] == 2 ** bits + 1

    def test_centroids_within_range(self):
        """All centroids should be within [-1, 1]."""
        for bits in [2, 3, 4]:
            centroids, _ = compute_lloyd_max_centroids(bits, block_size=32)
            assert (centroids >= -1.0).all() and (centroids <= 1.0).all()

    def test_symmetry(self):
        """Centroids should be approximately symmetric around 0."""
        centroids, _ = compute_lloyd_max_centroids(3, block_size=32)
        # c[i] ≈ -c[n-1-i]
        n = len(centroids)
        for i in range(n // 2):
            assert abs(centroids[i].item() + centroids[n - 1 - i].item()) < 0.01

    def test_higher_bits_finer(self):
        """More bits should produce finer quantization (closer centroids)."""
        c2, _ = compute_lloyd_max_centroids(2, block_size=32)
        c3, _ = compute_lloyd_max_centroids(3, block_size=32)
        c4, _ = compute_lloyd_max_centroids(4, block_size=32)
        # Max gap should decrease with more bits
        gap2 = (c2[1:] - c2[:-1]).max().item()
        gap3 = (c3[1:] - c3[:-1]).max().item()
        gap4 = (c4[1:] - c4[:-1]).max().item()
        assert gap2 > gap3 > gap4

    def test_different_block_sizes(self):
        """Should produce valid centroids for different block sizes."""
        for bs in [8, 16, 32, 64]:
            centroids, boundaries = compute_lloyd_max_centroids(3, block_size=bs)
            assert centroids.shape[0] == 8
            assert (centroids >= -1).all() and (centroids <= 1).all()


class TestCodebookManager:
    def test_caching(self):
        """Same (bits, block_size) should return cached result."""
        mgr = CodebookManager()
        c1, b1 = mgr.get(3, 32)
        c2, b2 = mgr.get(3, 32)
        torch.testing.assert_close(c1, c2)

    def test_global_getter(self):
        """get_codebook should work."""
        centroids, boundaries = get_codebook(4, 32)
        assert centroids.shape[0] == 16
        assert boundaries.shape[0] == 17
