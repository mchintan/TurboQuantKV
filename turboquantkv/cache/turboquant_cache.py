"""TurboQuantCache: drop-in Cache subclass for HuggingFace transformers."""

from __future__ import annotations

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

from turboquantkv.cache.turboquant_layer import TurboQuantLayer
from turboquantkv.config import TurboQuantConfig
from turboquantkv.core.quantizer import TurboQuantizer
from turboquantkv.core.rotation import RotationManager


class TurboQuantCache(Cache):
    """Drop-in replacement for DynamicCache with TurboQuant KV compression.

    Usage:
        config = TurboQuantConfig(key_bits=4, value_bits=3)
        cache = TurboQuantCache(config=model.config, quant_config=config)
        outputs = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: TurboQuantConfig | None = None,
    ):
        if quant_config is None:
            quant_config = TurboQuantConfig()

        self.quant_config = quant_config

        # Extract model dimensions
        num_layers = config.num_hidden_layers
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads

        # Determine device
        device = "cpu"

        # Validate block_size divides head_dim
        assert head_dim % quant_config.block_size == 0, (
            f"head_dim {head_dim} must be divisible by block_size {quant_config.block_size}"
        )

        # Create shared rotation manager (same rotation for all layers)
        key_rotation = RotationManager(
            rotation_type=quant_config.rotation_type,
            block_size=quant_config.block_size,
            head_dim=head_dim,
            seed=quant_config.seed,
            device=device,
        )
        value_rotation = RotationManager(
            rotation_type=quant_config.rotation_type,
            block_size=quant_config.block_size,
            head_dim=head_dim,
            seed=quant_config.seed + 500,
            device=device,
        )

        # Create per-layer quantizers and cache layers
        layers = []
        for layer_idx in range(num_layers):
            key_quantizer = TurboQuantizer(
                n_bits=quant_config.key_bits,
                rotation_manager=key_rotation,
                block_size=quant_config.block_size,
                use_qjl=quant_config.use_qjl,
                qjl_dim=quant_config.qjl_dim,
                head_dim=head_dim,
                seed=quant_config.seed + layer_idx,
                device=device,
            )
            value_quantizer = TurboQuantizer(
                n_bits=quant_config.value_bits,
                rotation_manager=value_rotation,
                block_size=quant_config.block_size,
                use_qjl=False,  # QJL only for keys (values don't need bias correction)
                head_dim=head_dim,
                seed=quant_config.seed + layer_idx + num_layers,
                device=device,
            )
            layers.append(TurboQuantLayer(key_quantizer, value_quantizer))

        super().__init__(layers=layers)
        self._num_layers = num_layers
        self._head_dim = head_dim

    def get_memory_bytes(self) -> dict[str, int]:
        """Get memory usage breakdown.

        Returns:
            Dict with 'compressed' (actual storage) and 'dequantized' (temp tensors) sizes.
        """
        compressed = 0
        dequantized = 0
        for layer in self.layers:
            if isinstance(layer, TurboQuantLayer):
                compressed += layer.get_compressed_memory_bytes()
                if layer._cached_keys is not None:
                    dequantized += (
                        layer._cached_keys.nelement() * layer._cached_keys.element_size()
                        + layer._cached_values.nelement() * layer._cached_values.element_size()
                    )
        return {"compressed": compressed, "dequantized": dequantized}
