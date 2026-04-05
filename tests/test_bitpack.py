"""Tests for bit packing/unpacking utilities."""

import torch
import pytest

from turboquantkv.core.bitpack import (
    pack_2bit, unpack_2bit,
    pack_3bit, unpack_3bit,
    pack_4bit, unpack_4bit,
    pack_bits, unpack_bits,
)


class TestPack2Bit:
    def test_round_trip(self):
        indices = torch.randint(0, 4, (8, 64), dtype=torch.uint8)
        packed = pack_2bit(indices)
        assert packed.shape == (8, 16)
        unpacked = unpack_2bit(packed, 64)
        torch.testing.assert_close(unpacked, indices)

    def test_specific_values(self):
        indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)
        packed = pack_2bit(indices)
        unpacked = unpack_2bit(packed, 4)
        torch.testing.assert_close(unpacked, indices)

    def test_batch_dims(self):
        indices = torch.randint(0, 4, (2, 4, 8, 32), dtype=torch.uint8)
        packed = pack_2bit(indices)
        unpacked = unpack_2bit(packed, 32)
        torch.testing.assert_close(unpacked, indices)


class TestPack3Bit:
    def test_round_trip(self):
        indices = torch.randint(0, 8, (8, 64), dtype=torch.uint8)
        packed = pack_3bit(indices)
        assert packed.shape == (8, 24)  # 64 * 3 / 8 = 24
        unpacked = unpack_3bit(packed, 64)
        torch.testing.assert_close(unpacked, indices)

    def test_all_values(self):
        """Test all possible 3-bit values."""
        indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.uint8)
        packed = pack_3bit(indices)
        unpacked = unpack_3bit(packed, 8)
        torch.testing.assert_close(unpacked, indices)

    def test_batch_dims(self):
        indices = torch.randint(0, 8, (2, 4, 8, 32), dtype=torch.uint8)
        packed = pack_3bit(indices)
        unpacked = unpack_3bit(packed, 32)
        torch.testing.assert_close(unpacked, indices)


class TestPack4Bit:
    def test_round_trip(self):
        indices = torch.randint(0, 16, (8, 64), dtype=torch.uint8)
        packed = pack_4bit(indices)
        assert packed.shape == (8, 32)
        unpacked = unpack_4bit(packed, 64)
        torch.testing.assert_close(unpacked, indices)

    def test_all_values(self):
        indices = torch.arange(16, dtype=torch.uint8).unsqueeze(0)
        packed = pack_4bit(indices)
        unpacked = unpack_4bit(packed, 16)
        torch.testing.assert_close(unpacked, indices)


class TestGenericInterface:
    @pytest.mark.parametrize("n_bits,max_val", [(2, 4), (3, 8), (4, 16)])
    def test_pack_unpack_round_trip(self, n_bits, max_val):
        indices = torch.randint(0, max_val, (4, 64), dtype=torch.uint8)
        packed = pack_bits(indices, n_bits)
        unpacked = unpack_bits(packed, n_bits, 64)
        torch.testing.assert_close(unpacked, indices)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            pack_bits(torch.zeros(4, 8, dtype=torch.uint8), 5)
