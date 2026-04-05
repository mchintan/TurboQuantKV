"""Tests for TurboQuantLayer and TurboQuantCache."""

import torch
import pytest

from turboquantkv.config import TurboQuantConfig
from turboquantkv.core.rotation import RotationManager
from turboquantkv.core.quantizer import TurboQuantizer
from turboquantkv.cache.turboquant_layer import TurboQuantLayer


def _make_layer(key_bits=4, value_bits=4, head_dim=64):
    rotation = RotationManager("wht", block_size=32, head_dim=head_dim)
    key_q = TurboQuantizer(n_bits=key_bits, rotation_manager=rotation, block_size=32, head_dim=head_dim)
    val_q = TurboQuantizer(n_bits=value_bits, rotation_manager=rotation, block_size=32, head_dim=head_dim)
    return TurboQuantLayer(key_q, val_q)


class TestTurboQuantLayer:
    def test_single_update(self):
        layer = _make_layer()
        k = torch.randn(1, 2, 4, 64)
        v = torch.randn(1, 2, 4, 64)
        k_out, v_out = layer.update(k, v)
        assert k_out.shape == (1, 2, 4, 64)
        assert v_out.shape == (1, 2, 4, 64)
        assert layer.get_seq_length() == 4

    def test_multiple_updates(self):
        """Simulates prefill + decode steps."""
        layer = _make_layer()
        # Prefill
        k1 = torch.randn(1, 2, 8, 64)
        v1 = torch.randn(1, 2, 8, 64)
        k_out, v_out = layer.update(k1, v1)
        assert k_out.shape == (1, 2, 8, 64)
        assert layer.get_seq_length() == 8

        # Decode step 1
        k2 = torch.randn(1, 2, 1, 64)
        v2 = torch.randn(1, 2, 1, 64)
        k_out, v_out = layer.update(k2, v2)
        assert k_out.shape == (1, 2, 9, 64)
        assert layer.get_seq_length() == 9

        # Decode step 2
        k3 = torch.randn(1, 2, 1, 64)
        v3 = torch.randn(1, 2, 1, 64)
        k_out, v_out = layer.update(k3, v3)
        assert k_out.shape == (1, 2, 10, 64)
        assert layer.get_seq_length() == 10

    def test_mask_sizes(self):
        layer = _make_layer()
        k = torch.randn(1, 2, 5, 64)
        v = torch.randn(1, 2, 5, 64)
        layer.update(k, v)
        kv_len, kv_offset = layer.get_mask_sizes(query_length=1)
        # kv_length = past_seq_len + query_length = 5 + 1 = 6
        assert kv_len == 6
        assert kv_offset == 0

    def test_max_cache_shape(self):
        layer = _make_layer()
        assert layer.get_max_cache_shape() == -1

    def test_compressed_memory(self):
        layer = _make_layer(key_bits=3, value_bits=3)
        k = torch.randn(1, 2, 100, 64)
        v = torch.randn(1, 2, 100, 64)
        layer.update(k, v)
        compressed_bytes = layer.get_compressed_memory_bytes()
        # Baseline would be 2 * 1 * 2 * 100 * 64 * 2 = 51200 bytes (fp16)
        baseline_bytes = 2 * 1 * 2 * 100 * 64 * 2
        assert compressed_bytes < baseline_bytes, (
            f"Compressed {compressed_bytes} should be less than baseline {baseline_bytes}"
        )
