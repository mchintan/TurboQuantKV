"""TurboQuant cache layer implementing CacheLayerMixin for HuggingFace transformers."""

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import CacheLayerMixin

from turboquantkv.core.quantizer import QuantizedTensor, TurboQuantizer


class TurboQuantLayer(CacheLayerMixin):
    """Drop-in replacement for DynamicLayer that stores KV in TurboQuant format.

    On update(), quantizes incoming key/value states, stores them compressed,
    and returns dequantized full-sequence tensors for the attention layer.
    """

    is_compileable = False

    def __init__(self, key_quantizer: TurboQuantizer, value_quantizer: TurboQuantizer):
        super().__init__()
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer

        # Compressed storage: lists of QuantizedTensor (one per update call)
        self._key_chunks: list[QuantizedTensor] = []
        self._value_chunks: list[QuantizedTensor] = []
        self._seq_length = 0

        # Cache dequantized tensors for incremental concat optimization
        self._cached_keys: Optional[torch.Tensor] = None
        self._cached_values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """Initialize on first call. No pre-allocation needed for dynamic storage."""
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize new KV, append to storage, return full dequantized cache.

        Args:
            key_states: (batch, num_kv_heads, new_seq_len, head_dim)
            value_states: (batch, num_kv_heads, new_seq_len, head_dim)

        Returns:
            (full_keys, full_values) dequantized, shape (batch, heads, total_seq, head_dim)
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Quantize new tokens
        key_qt = self.key_quantizer.quantize(key_states)
        value_qt = self.value_quantizer.quantize(value_states)

        self._key_chunks.append(key_qt)
        self._value_chunks.append(value_qt)
        self._seq_length += key_states.shape[-2]

        # Dequantize new chunk
        new_keys = self.key_quantizer.dequantize(key_qt, dtype=key_states.dtype)
        new_values = self.value_quantizer.dequantize(value_qt, dtype=value_states.dtype)

        # Incremental concat: append to cached dequantized tensors
        if self._cached_keys is None:
            self._cached_keys = new_keys
            self._cached_values = new_values
        else:
            self._cached_keys = torch.cat([self._cached_keys, new_keys], dim=-2)
            self._cached_values = torch.cat([self._cached_values, new_values], dim=-2)

        return self._cached_keys, self._cached_values

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return (self._seq_length + query_length, 0)

    def get_max_cache_shape(self) -> int:
        return -1  # unbounded

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder cache for beam search."""
        if self._cached_keys is not None:
            self._cached_keys = self._cached_keys.index_select(0, beam_idx)
            self._cached_values = self._cached_values.index_select(0, beam_idx)

        # Also reorder compressed chunks
        new_key_chunks = []
        new_value_chunks = []
        for kc, vc in zip(self._key_chunks, self._value_chunks):
            new_kc = QuantizedTensor(
                packed_indices=kc.packed_indices.index_select(0, beam_idx),
                norms=kc.norms.index_select(0, beam_idx),
                n_bits=kc.n_bits,
                num_elements=kc.num_elements,
                qjl_signs=kc.qjl_signs.index_select(0, beam_idx) if kc.qjl_signs is not None else None,
            )
            new_vc = QuantizedTensor(
                packed_indices=vc.packed_indices.index_select(0, beam_idx),
                norms=vc.norms.index_select(0, beam_idx),
                n_bits=vc.n_bits,
                num_elements=vc.num_elements,
                qjl_signs=vc.qjl_signs.index_select(0, beam_idx) if vc.qjl_signs is not None else None,
            )
            new_key_chunks.append(new_kc)
            new_value_chunks.append(new_vc)
        self._key_chunks = new_key_chunks
        self._value_chunks = new_value_chunks

    def get_compressed_memory_bytes(self) -> int:
        """Estimate compressed storage size in bytes."""
        total = 0
        for qt in self._key_chunks + self._value_chunks:
            total += qt.packed_indices.nelement() * qt.packed_indices.element_size()
            total += qt.norms.nelement() * qt.norms.element_size()
            if qt.qjl_signs is not None:
                total += qt.qjl_signs.nelement() * qt.qjl_signs.element_size()
        return total
