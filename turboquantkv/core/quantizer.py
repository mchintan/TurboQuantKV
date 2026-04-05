"""TurboQuant quantizer: PolarQuant + optional QJL residual correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from turboquantkv.core.bitpack import pack_bits, unpack_bits
from turboquantkv.core.codebook import get_codebook
from turboquantkv.core.rotation import RotationManager


@dataclass
class QuantizedTensor:
    """Compressed representation of a tensor."""

    packed_indices: torch.Tensor  # uint8, bit-packed quantization indices
    norms: torch.Tensor  # float32, per-vector norms
    n_bits: int  # bit width used
    num_elements: int  # original last-dim size (head_dim)
    qjl_signs: Optional[torch.Tensor] = None  # uint8, bit-packed QJL sign bits

    @property
    def shape_info(self) -> str:
        return (
            f"packed={self.packed_indices.shape}, norms={self.norms.shape}, "
            f"bits={self.n_bits}, d={self.num_elements}"
        )


class PolarQuantEncoder:
    """Stage 1: Rotation + Lloyd-Max scalar quantization."""

    def __init__(self, n_bits: int, rotation_manager: RotationManager, block_size: int):
        self.n_bits = n_bits
        self.rotation = rotation_manager
        self.block_size = block_size
        self._centroids = None
        self._boundaries = None

    def _ensure_codebook(self, device: str):
        if self._centroids is None or self._centroids.device != torch.device(device):
            self._centroids, self._boundaries = get_codebook(
                self.n_bits, self.block_size, device
            )

    def encode(self, x: torch.Tensor) -> QuantizedTensor:
        """Quantize: normalize -> rotate -> scalar quantize per coordinate.

        Args:
            x: (..., head_dim) tensor in fp16/bf16/fp32.

        Returns:
            QuantizedTensor with packed indices and norms.
        """
        self._ensure_codebook(str(x.device))

        # Compute and store norms (float32 to avoid fp16 overflow)
        norms = x.float().norm(dim=-1, keepdim=True)  # (..., 1)

        # Normalize to unit vectors
        x_unit = x / norms.clamp(min=1e-8).to(x.dtype)

        # Rotate
        x_rot = self.rotation.rotate(x_unit)  # (..., head_dim)

        # Quantize: find nearest centroid per coordinate using boundaries
        boundaries = self._boundaries  # (n_levels + 1,)
        # Use bucketize to find which bin each value falls into
        indices = torch.bucketize(x_rot.float(), boundaries[1:-1])  # (..., head_dim)
        indices = indices.clamp(0, 2**self.n_bits - 1)

        # Pack indices
        packed = pack_bits(indices.to(torch.uint8), self.n_bits)

        return QuantizedTensor(
            packed_indices=packed,
            norms=norms.squeeze(-1),  # (...,)
            n_bits=self.n_bits,
            num_elements=x.shape[-1],
        )

    def decode(self, qt: QuantizedTensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize: lookup centroids -> unrotate -> rescale.

        Args:
            qt: QuantizedTensor to decode.
            dtype: Output dtype.

        Returns:
            (..., head_dim) reconstructed tensor.
        """
        device = qt.packed_indices.device
        self._ensure_codebook(str(device))

        # Unpack indices
        indices = unpack_bits(qt.packed_indices, qt.n_bits, qt.num_elements)

        # Lookup centroids
        centroids = self._centroids  # (n_levels,)
        x_rot = centroids[indices.long()]  # (..., head_dim)

        # Unrotate
        x_unit = self.rotation.unrotate(x_rot)

        # Rescale by norms
        norms = qt.norms.unsqueeze(-1).to(x_unit.dtype)  # (..., 1)
        return (x_unit * norms).to(dtype)


class QJLCorrector:
    """Stage 2: Quantized Johnson-Lindenstrauss residual correction.

    Projects quantization residual through a random sign matrix and stores
    only the signs. Provides unbiased correction for attention score estimation.
    """

    def __init__(self, projection_dim: int, head_dim: int, seed: int = 42,
                 device: str = "cpu"):
        self.projection_dim = projection_dim
        self.head_dim = head_dim
        # Generate random Rademacher matrix (+1/-1)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self._S = (
            torch.randint(0, 2, (projection_dim, head_dim), generator=gen).float() * 2 - 1
        ).to(device)

    def encode_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Project residual and store signs as packed bits.

        Args:
            residual: (..., head_dim) quantization error.

        Returns:
            (..., ceil(projection_dim / 8)) packed uint8 sign bits.
        """
        S = self._S.to(residual.device)
        # Project: (..., head_dim) @ (head_dim, projection_dim) -> (..., projection_dim)
        z = residual.float() @ S.T
        # Store signs: +1 -> 1, -1 -> 0
        signs = (z > 0).to(torch.uint8)

        # Pack bits: 8 signs per byte
        m = self.projection_dim
        padded = m + (8 - m % 8) % 8  # pad to multiple of 8
        if padded > m:
            pad_zeros = torch.zeros(*signs.shape[:-1], padded - m,
                                    dtype=torch.uint8, device=signs.device)
            signs = torch.cat([signs, pad_zeros], dim=-1)

        # Pack 8 bits per byte
        signs = signs.view(*signs.shape[:-1], -1, 8)
        packed = torch.zeros(*signs.shape[:-1], dtype=torch.uint8, device=signs.device)
        for i in range(8):
            packed = packed | (signs[..., i] << i)

        return packed

    def decode_signs(self, packed_signs: torch.Tensor) -> torch.Tensor:
        """Unpack sign bits back to +1/-1 tensor.

        Returns:
            (..., projection_dim) tensor with values +1 or -1.
        """
        unpacked = []
        for i in range(8):
            unpacked.append((packed_signs >> i) & 1)
        signs = torch.stack(unpacked, dim=-1).view(*packed_signs.shape[:-1], -1)
        # Trim to projection_dim and convert 0/1 to -1/+1
        signs = signs[..., :self.projection_dim]
        return signs.float() * 2 - 1

    def correction(self, query_rotated: torch.Tensor,
                   packed_signs: torch.Tensor) -> torch.Tensor:
        """Compute QJL bias correction for attention scores.

        Args:
            query_rotated: (..., head_dim) rotated query (full precision).
            packed_signs: (..., ceil(m/8)) packed sign bits from encode_residual.

        Returns:
            (...,) correction term to add to attention scores.
        """
        S = self._S.to(query_rotated.device)
        signs = self.decode_signs(packed_signs)  # (..., projection_dim)

        # Project query through same random matrix
        q_proj = query_rotated.float() @ S.T  # (..., projection_dim)

        # Correction = dot(q_proj, signs) / m
        correction = (q_proj * signs).sum(dim=-1) / self.projection_dim
        return correction


class TurboQuantizer:
    """Full TurboQuant pipeline: PolarQuant + optional QJL."""

    def __init__(self, n_bits: int, rotation_manager: RotationManager,
                 block_size: int, use_qjl: bool = False, qjl_dim: int = 64,
                 head_dim: int = 64, seed: int = 42, device: str = "cpu"):
        self.polar = PolarQuantEncoder(n_bits, rotation_manager, block_size)
        self.use_qjl = use_qjl
        self.qjl = None
        self.rotation = rotation_manager

        if use_qjl:
            self.qjl = QJLCorrector(qjl_dim, head_dim, seed=seed + 1000, device=device)

    def quantize(self, x: torch.Tensor) -> QuantizedTensor:
        """Quantize a tensor.

        Args:
            x: (..., head_dim) input tensor.

        Returns:
            QuantizedTensor with packed indices, norms, and optional QJL signs.
        """
        qt = self.polar.encode(x)

        if self.use_qjl and self.qjl is not None:
            # Compute residual
            x_hat = self.polar.decode(qt, dtype=x.dtype)
            residual = x - x_hat
            qt.qjl_signs = self.qjl.encode_residual(residual)

        return qt

    def dequantize(self, qt: QuantizedTensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize back to a full tensor (PolarQuant decode only, no QJL)."""
        return self.polar.decode(qt, dtype=dtype)
