from dataclasses import dataclass


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""

    key_bits: int = 4
    value_bits: int = 4
    rotation_type: str = "wht"  # "wht" or "random"
    block_size: int = 32  # must be power of 2, must divide head_dim
    use_qjl: bool = False  # QJL residual correction (off by default)
    qjl_dim: int = 64  # number of random projections for QJL
    seed: int = 42

    def __post_init__(self):
        assert self.key_bits in (2, 3, 4), f"key_bits must be 2, 3, or 4, got {self.key_bits}"
        assert self.value_bits in (2, 3, 4), f"value_bits must be 2, 3, or 4, got {self.value_bits}"
        assert self.rotation_type in ("wht", "random"), f"rotation_type must be 'wht' or 'random'"
        assert self.block_size > 0 and (self.block_size & (self.block_size - 1)) == 0, (
            f"block_size must be a power of 2, got {self.block_size}"
        )
