"""Bit packing and unpacking utilities for sub-byte quantization indices."""

import torch


def pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit indices into uint8 bytes (4 values per byte).

    Args:
        indices: (..., d) tensor with values in [0, 3]. d must be divisible by 4.

    Returns:
        (..., d // 4) uint8 tensor.
    """
    d = indices.shape[-1]
    assert d % 4 == 0, f"dimension {d} must be divisible by 4 for 2-bit packing"
    indices = indices.to(torch.uint8)
    reshaped = indices.view(*indices.shape[:-1], d // 4, 4)
    packed = (
        reshaped[..., 0]
        | (reshaped[..., 1] << 2)
        | (reshaped[..., 2] << 4)
        | (reshaped[..., 3] << 6)
    )
    return packed


def unpack_2bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack uint8 bytes into 2-bit indices.

    Args:
        packed: (..., d // 4) uint8 tensor.
        num_elements: Original number of elements d.

    Returns:
        (..., d) uint8 tensor with values in [0, 3].
    """
    v0 = packed & 0x03
    v1 = (packed >> 2) & 0x03
    v2 = (packed >> 4) & 0x03
    v3 = (packed >> 6) & 0x03
    return torch.stack([v0, v1, v2, v3], dim=-1).view(*packed.shape[:-1], num_elements)


def pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 3-bit indices into uint8 bytes (8 values in 3 bytes).

    Args:
        indices: (..., d) tensor with values in [0, 7]. d must be divisible by 8.

    Returns:
        (..., d * 3 // 8) uint8 tensor.
    """
    d = indices.shape[-1]
    assert d % 8 == 0, f"dimension {d} must be divisible by 8 for 3-bit packing"
    # Use int32 for bitshift compatibility (uint16 doesn't support lshift on CPU)
    v = indices.to(torch.int32).view(*indices.shape[:-1], d // 8, 8)

    # Pack 8 x 3-bit values into 3 bytes (24 bits)
    b0 = (v[..., 0] | (v[..., 1] << 3) | (v[..., 2] << 6)).to(torch.uint8)
    b1 = ((v[..., 2] >> 2) | (v[..., 3] << 1) | (v[..., 4] << 4) | (v[..., 5] << 7)).to(torch.uint8)
    b2 = ((v[..., 5] >> 1) | (v[..., 6] << 2) | (v[..., 7] << 5)).to(torch.uint8)

    return torch.stack([b0, b1, b2], dim=-1).view(*indices.shape[:-1], d * 3 // 8)


def unpack_3bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack uint8 bytes into 3-bit indices.

    Args:
        packed: (..., d * 3 // 8) uint8 tensor.
        num_elements: Original number of elements d.

    Returns:
        (..., d) uint8 tensor with values in [0, 7].
    """
    packed = packed.to(torch.int32)
    reshaped = packed.view(*packed.shape[:-1], num_elements // 8, 3)
    b0, b1, b2 = reshaped[..., 0], reshaped[..., 1], reshaped[..., 2]

    v0 = b0 & 0x07
    v1 = (b0 >> 3) & 0x07
    v2 = ((b0 >> 6) | (b1 << 2)) & 0x07
    v3 = (b1 >> 1) & 0x07
    v4 = (b1 >> 4) & 0x07
    v5 = ((b1 >> 7) | (b2 << 1)) & 0x07
    v6 = (b2 >> 2) & 0x07
    v7 = (b2 >> 5) & 0x07

    result = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1)
    return result.view(*packed.shape[:-1], num_elements).to(torch.uint8)


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices into uint8 bytes (2 values per byte, nibble packing).

    Args:
        indices: (..., d) tensor with values in [0, 15]. d must be divisible by 2.

    Returns:
        (..., d // 2) uint8 tensor.
    """
    d = indices.shape[-1]
    assert d % 2 == 0, f"dimension {d} must be divisible by 2 for 4-bit packing"
    indices = indices.to(torch.uint8)
    reshaped = indices.view(*indices.shape[:-1], d // 2, 2)
    packed = reshaped[..., 0] | (reshaped[..., 1] << 4)
    return packed


def unpack_4bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    """Unpack uint8 bytes into 4-bit indices.

    Args:
        packed: (..., d // 2) uint8 tensor.
        num_elements: Original number of elements d.

    Returns:
        (..., d) uint8 tensor with values in [0, 15].
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).view(*packed.shape[:-1], num_elements)


def pack_bits(indices: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Pack indices into bytes for the given bit width."""
    if n_bits == 2:
        return pack_2bit(indices)
    elif n_bits == 3:
        return pack_3bit(indices)
    elif n_bits == 4:
        return pack_4bit(indices)
    else:
        raise ValueError(f"Unsupported bit width: {n_bits}")


def unpack_bits(packed: torch.Tensor, n_bits: int, num_elements: int) -> torch.Tensor:
    """Unpack bytes into indices for the given bit width."""
    if n_bits == 2:
        return unpack_2bit(packed, num_elements)
    elif n_bits == 3:
        return unpack_3bit(packed, num_elements)
    elif n_bits == 4:
        return unpack_4bit(packed, num_elements)
    else:
        raise ValueError(f"Unsupported bit width: {n_bits}")
