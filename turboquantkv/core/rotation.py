"""Walsh-Hadamard Transform and random orthogonal rotation for TurboQuant."""

import math

import torch


def walsh_hadamard_transform(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform on the last dimension.

    Applies the WHT in blocks of `block_size` using the butterfly algorithm.
    O(d log d) per vector. The WHT is its own inverse (self-inverse / involutory)
    when properly normalized.

    Args:
        x: (..., d) tensor where d is divisible by block_size.
        block_size: Size of WHT blocks. Must be a power of 2.

    Returns:
        Transformed tensor, same shape as input.
    """
    d = x.shape[-1]
    assert d % block_size == 0, f"head_dim {d} must be divisible by block_size {block_size}"

    orig_dtype = x.dtype
    orig_shape = x.shape
    x = x.float().contiguous().clone()

    # Reshape so the last dim is split into (num_blocks, block_size)
    num_blocks = d // block_size
    x = x.view(-1, num_blocks, block_size)
    flat_batch = x.shape[0]

    # Butterfly stages
    h = 1
    while h < block_size:
        # Reshape to isolate pairs: (batch, num_blocks, block_size/(2h), 2, h)
        x = x.view(flat_batch, num_blocks, -1, 2, h)
        a = x[:, :, :, 0, :].clone()
        b = x[:, :, :, 1, :].clone()
        x[:, :, :, 0, :] = a + b
        x[:, :, :, 1, :] = a - b
        x = x.view(flat_batch, num_blocks, block_size)
        h *= 2

    x = x / math.sqrt(block_size)
    return x.view(orig_shape).to(orig_dtype)


def generate_rotation_matrix(dim: int, seed: int = 42, device: str = "cpu") -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    Args:
        dim: Matrix dimension.
        seed: Random seed for reproducibility.
        device: Target device.

    Returns:
        (dim, dim) orthogonal matrix Q where Q^T Q = I.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(dim, dim, generator=gen, device="cpu")
    Q, R = torch.linalg.qr(G)
    # Ensure deterministic sign (Haar measure): multiply by sign of diagonal of R
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
    return Q.to(device)


def random_orthogonal_rotation(x: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Apply precomputed orthogonal matrix to last dimension.

    Args:
        x: (..., d) tensor.
        Q: (d, d) orthogonal matrix.

    Returns:
        x @ Q^T, same shape as x.
    """
    return x @ Q.T


class RotationManager:
    """Manages rotation transforms for a given config.

    Caches the rotation matrix (for random mode) and provides
    consistent rotate/unrotate operations.
    """

    def __init__(self, rotation_type: str, block_size: int, head_dim: int,
                 seed: int = 42, device: str = "cpu"):
        self.rotation_type = rotation_type
        self.block_size = block_size
        self.head_dim = head_dim
        self._Q = None

        assert head_dim % block_size == 0, (
            f"head_dim {head_dim} must be divisible by block_size {block_size}"
        )

        if rotation_type == "random":
            self._Q = generate_rotation_matrix(head_dim, seed=seed, device=device)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation_type == "wht":
            return walsh_hadamard_transform(x, self.block_size)
        else:
            return random_orthogonal_rotation(x, self._Q.to(x.device))

    def unrotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation_type == "wht":
            # WHT is self-inverse (with normalization)
            return walsh_hadamard_transform(x, self.block_size)
        else:
            return x @ self._Q.to(x.device)  # Q @ Q^T = I, so inverse is Q (not Q^T)
