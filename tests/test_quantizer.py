"""Tests for PolarQuant encoder, QJL corrector, and TurboQuantizer."""

import torch
import pytest

from turboquantkv.config import TurboQuantConfig
from turboquantkv.core.rotation import RotationManager
from turboquantkv.core.quantizer import PolarQuantEncoder, QJLCorrector, TurboQuantizer


@pytest.fixture
def rotation_mgr():
    return RotationManager("wht", block_size=32, head_dim=64)


class TestPolarQuantEncoder:
    def test_encode_decode_shape(self, rotation_mgr):
        enc = PolarQuantEncoder(n_bits=4, rotation_manager=rotation_mgr, block_size=32)
        x = torch.randn(2, 4, 8, 64)
        qt = enc.encode(x)
        x_hat = enc.decode(qt)
        assert x_hat.shape == x.shape

    def test_encode_decode_mse(self, rotation_mgr):
        """Reconstruction error should be reasonable."""
        enc = PolarQuantEncoder(n_bits=4, rotation_manager=rotation_mgr, block_size=32)
        x = torch.randn(100, 64)
        qt = enc.encode(x)
        x_hat = enc.decode(qt, dtype=torch.float32)
        # Relative MSE should be small for 4-bit
        mse = ((x - x_hat) ** 2).mean()
        signal = (x ** 2).mean()
        relative_mse = mse / signal
        assert relative_mse < 0.15, f"Relative MSE too high: {relative_mse:.4f}"

    def test_lower_bits_higher_error(self, rotation_mgr):
        """2-bit should have higher error than 4-bit."""
        x = torch.randn(100, 64)
        errors = {}
        for bits in [2, 3, 4]:
            enc = PolarQuantEncoder(n_bits=bits, rotation_manager=rotation_mgr, block_size=32)
            qt = enc.encode(x)
            x_hat = enc.decode(qt, dtype=torch.float32)
            errors[bits] = ((x - x_hat) ** 2).mean().item()
        assert errors[2] > errors[3] > errors[4]

    def test_norms_stored_float32(self, rotation_mgr):
        """Norms should be float32 to avoid fp16 overflow."""
        enc = PolarQuantEncoder(n_bits=4, rotation_manager=rotation_mgr, block_size=32)
        x = torch.randn(8, 64) * 1000  # large norms
        qt = enc.encode(x)
        assert qt.norms.dtype == torch.float32

    def test_fp16_input(self, rotation_mgr):
        """Should handle fp16 input."""
        enc = PolarQuantEncoder(n_bits=4, rotation_manager=rotation_mgr, block_size=32)
        x = torch.randn(8, 64, dtype=torch.float16)
        qt = enc.encode(x)
        x_hat = enc.decode(qt, dtype=torch.float16)
        assert x_hat.dtype == torch.float16


class TestQJLCorrector:
    def test_encode_shape(self):
        qjl = QJLCorrector(projection_dim=64, head_dim=64, seed=42)
        residual = torch.randn(8, 64)
        signs = qjl.encode_residual(residual)
        assert signs.shape == (8, 8)  # 64 bits / 8 = 8 bytes

    def test_decode_signs_shape(self):
        qjl = QJLCorrector(projection_dim=64, head_dim=64, seed=42)
        packed = torch.randint(0, 256, (8, 8), dtype=torch.uint8)
        signs = qjl.decode_signs(packed)
        assert signs.shape == (8, 64)
        assert ((signs == 1) | (signs == -1)).all()

    def test_correction_shape(self):
        qjl = QJLCorrector(projection_dim=64, head_dim=64, seed=42)
        query = torch.randn(8, 64)
        packed = torch.randint(0, 256, (8, 8), dtype=torch.uint8)
        corr = qjl.correction(query, packed)
        assert corr.shape == (8,)


class TestTurboQuantizer:
    def test_quantize_dequantize(self, rotation_mgr):
        tq = TurboQuantizer(
            n_bits=4, rotation_manager=rotation_mgr, block_size=32,
            use_qjl=False, head_dim=64,
        )
        x = torch.randn(2, 4, 8, 64)
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_with_qjl(self, rotation_mgr):
        tq = TurboQuantizer(
            n_bits=4, rotation_manager=rotation_mgr, block_size=32,
            use_qjl=True, qjl_dim=64, head_dim=64,
        )
        x = torch.randn(8, 64)
        qt = tq.quantize(x)
        assert qt.qjl_signs is not None
        x_hat = tq.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_kv_cache_shapes(self, rotation_mgr):
        """Test with typical KV cache tensor shapes: (batch, heads, seq, head_dim)."""
        tq = TurboQuantizer(
            n_bits=3, rotation_manager=rotation_mgr, block_size=32,
            use_qjl=False, head_dim=64,
        )
        # Prefill: batch=1, heads=4, seq=32, head_dim=64
        x = torch.randn(1, 4, 32, 64)
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        assert x_hat.shape == (1, 4, 32, 64)

        # Decode step: batch=1, heads=4, seq=1, head_dim=64
        x = torch.randn(1, 4, 1, 64)
        qt = tq.quantize(x)
        x_hat = tq.dequantize(qt)
        assert x_hat.shape == (1, 4, 1, 64)
