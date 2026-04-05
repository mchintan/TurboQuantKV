"""Tests for Walsh-Hadamard Transform and random orthogonal rotation."""

import torch
import pytest

from turboquantkv.core.rotation import (
    walsh_hadamard_transform,
    generate_rotation_matrix,
    random_orthogonal_rotation,
    RotationManager,
)


class TestWHT:
    def test_self_inverse(self):
        """WHT(WHT(x)) should return x."""
        x = torch.randn(4, 32)
        y = walsh_hadamard_transform(x, block_size=32)
        x_reconstructed = walsh_hadamard_transform(y, block_size=32)
        torch.testing.assert_close(x_reconstructed, x, atol=1e-5, rtol=1e-5)

    def test_norm_preservation(self):
        """WHT should preserve vector norms."""
        x = torch.randn(8, 64)
        y = walsh_hadamard_transform(x, block_size=32)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        torch.testing.assert_close(x_norms, y_norms, atol=1e-4, rtol=1e-4)

    def test_different_block_sizes(self):
        """WHT should work with various power-of-2 block sizes."""
        x = torch.randn(2, 64)
        for bs in [8, 16, 32, 64]:
            y = walsh_hadamard_transform(x, block_size=bs)
            x_rec = walsh_hadamard_transform(y, block_size=bs)
            torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_batch_dims(self):
        """WHT should handle arbitrary batch dimensions."""
        x = torch.randn(2, 4, 8, 64)
        y = walsh_hadamard_transform(x, block_size=32)
        assert y.shape == x.shape
        x_rec = walsh_hadamard_transform(y, block_size=32)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_fp16_input(self):
        """WHT should handle fp16 input and return fp16."""
        x = torch.randn(4, 32, dtype=torch.float16)
        y = walsh_hadamard_transform(x, block_size=32)
        assert y.dtype == torch.float16
        x_rec = walsh_hadamard_transform(y, block_size=32)
        torch.testing.assert_close(x_rec, x, atol=1e-2, rtol=1e-2)

    def test_block_size_must_divide_dim(self):
        """Should raise on incompatible dimensions."""
        x = torch.randn(4, 30)
        with pytest.raises(AssertionError):
            walsh_hadamard_transform(x, block_size=32)


class TestRandomRotation:
    def test_orthogonality(self):
        """Q^T Q should be identity."""
        Q = generate_rotation_matrix(64, seed=42)
        eye = Q.T @ Q
        torch.testing.assert_close(eye, torch.eye(64), atol=1e-5, rtol=1e-5)

    def test_norm_preservation(self):
        """Random rotation should preserve norms."""
        Q = generate_rotation_matrix(64, seed=42)
        x = torch.randn(8, 64)
        y = random_orthogonal_rotation(x, Q)
        torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5, rtol=1e-5)

    def test_round_trip(self):
        """Rotating and un-rotating should return original."""
        Q = generate_rotation_matrix(64, seed=42)
        x = torch.randn(8, 64)
        y = x @ Q.T
        x_rec = y @ Q
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_deterministic(self):
        """Same seed should produce same matrix."""
        Q1 = generate_rotation_matrix(32, seed=123)
        Q2 = generate_rotation_matrix(32, seed=123)
        torch.testing.assert_close(Q1, Q2)


class TestRotationManager:
    def test_wht_round_trip(self):
        mgr = RotationManager("wht", block_size=32, head_dim=64)
        x = torch.randn(4, 64)
        y = mgr.rotate(x)
        x_rec = mgr.unrotate(y)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_random_round_trip(self):
        mgr = RotationManager("random", block_size=32, head_dim=64, seed=42)
        x = torch.randn(4, 64)
        y = mgr.rotate(x)
        x_rec = mgr.unrotate(y)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)
