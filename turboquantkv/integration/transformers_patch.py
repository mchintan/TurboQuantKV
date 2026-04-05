"""Integration helpers for using TurboQuant with HuggingFace transformers models."""

from __future__ import annotations

from typing import Optional

import torch
from transformers import PreTrainedModel

from turboquantkv.cache.turboquant_cache import TurboQuantCache
from turboquantkv.config import TurboQuantConfig


def generate_with_turboquant(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    quant_config: Optional[TurboQuantConfig] = None,
    **generate_kwargs,
) -> torch.Tensor:
    """Generate text using TurboQuant KV cache compression.

    Creates a TurboQuantCache and passes it to model.generate().

    Args:
        model: A HuggingFace causal LM model.
        input_ids: (batch, seq_len) input token IDs.
        quant_config: TurboQuant configuration. Uses defaults if None.
        **generate_kwargs: Additional args passed to model.generate().

    Returns:
        Generated token IDs tensor.
    """
    if quant_config is None:
        quant_config = TurboQuantConfig()

    cache = TurboQuantCache(config=model.config, quant_config=quant_config)
    return model.generate(input_ids, past_key_values=cache, **generate_kwargs)
