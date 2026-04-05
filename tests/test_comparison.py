"""Pre vs Post TurboQuant comparison tests.

Runs a small model with baseline DynamicCache and TurboQuantCache,
comparing output quality, token agreement, and memory usage.
"""

import pytest
import torch

pytestmark = pytest.mark.slow


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "openai-community/gpt2"
    device = _get_device()
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


PROMPTS = [
    "The meaning of life is",
    "In the year 2050, artificial intelligence will",
    "The quick brown fox jumps over the lazy dog. Then,",
    "Python is a programming language that",
]


class TestPrePostComparison:
    """Compare baseline (DynamicCache) vs TurboQuant output quality."""

    def _generate(self, model, tokenizer, device, prompt, cache=None, max_new_tokens=30):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return outputs[0]

    def test_token_agreement_4bit(self, model_and_tokenizer):
        """4-bit quantization should have high token agreement with baseline."""
        from transformers.cache_utils import DynamicCache
        from turboquantkv import TurboQuantCache, TurboQuantConfig

        model, tokenizer, device = model_and_tokenizer
        config = TurboQuantConfig(key_bits=4, value_bits=4)

        total_tokens = 0
        matching_tokens = 0

        for prompt in PROMPTS:
            baseline_ids = self._generate(model, tokenizer, device, prompt)
            tq_cache = TurboQuantCache(config=model.config, quant_config=config)
            tq_ids = self._generate(model, tokenizer, device, prompt, cache=tq_cache)

            # Compare generated tokens (skip prompt)
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
            min_len = min(len(baseline_ids), len(tq_ids))
            gen_baseline = baseline_ids[prompt_len:min_len]
            gen_tq = tq_ids[prompt_len:min_len]

            total_tokens += len(gen_baseline)
            matching_tokens += (gen_baseline == gen_tq).sum().item()

        agreement = matching_tokens / max(total_tokens, 1)
        print(f"\n4-bit token agreement: {agreement:.1%} ({matching_tokens}/{total_tokens})")
        # GPT-2 has small head_dim=64, so quantization error is higher than larger models.
        # Larger models (Qwen2, Mistral) will have much higher agreement.
        assert agreement >= 0.40, f"4-bit token agreement {agreement:.1%} below 40% threshold"

    def test_token_agreement_3bit(self, model_and_tokenizer):
        """3-bit quantization should still have reasonable token agreement."""
        from turboquantkv import TurboQuantCache, TurboQuantConfig

        model, tokenizer, device = model_and_tokenizer
        config = TurboQuantConfig(key_bits=3, value_bits=3)

        total_tokens = 0
        matching_tokens = 0

        for prompt in PROMPTS:
            baseline_ids = self._generate(model, tokenizer, device, prompt)
            tq_cache = TurboQuantCache(config=model.config, quant_config=config)
            tq_ids = self._generate(model, tokenizer, device, prompt, cache=tq_cache)

            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
            min_len = min(len(baseline_ids), len(tq_ids))
            gen_baseline = baseline_ids[prompt_len:min_len]
            gen_tq = tq_ids[prompt_len:min_len]

            total_tokens += len(gen_baseline)
            matching_tokens += (gen_baseline == gen_tq).sum().item()

        agreement = matching_tokens / max(total_tokens, 1)
        print(f"\n3-bit token agreement: {agreement:.1%} ({matching_tokens}/{total_tokens})")
        # Lower threshold for 3-bit: GPT-2 small head_dim makes this harder
        assert agreement >= 0.20, f"3-bit token agreement {agreement:.1%} below 20% threshold"

    def test_memory_compression_ratio(self, model_and_tokenizer):
        """TurboQuant should achieve meaningful compression."""
        from transformers.cache_utils import DynamicCache
        from turboquantkv import TurboQuantCache, TurboQuantConfig

        model, tokenizer, device = model_and_tokenizer
        prompt = " ".join(["word"] * 50)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline
        baseline_cache = DynamicCache()
        with torch.no_grad():
            model(**inputs, past_key_values=baseline_cache)
        baseline_bytes = sum(
            l.keys.nelement() * l.keys.element_size() + l.values.nelement() * l.values.element_size()
            for l in baseline_cache.layers
            if hasattr(l, "keys") and l.keys is not None
        )

        # TurboQuant 3-bit
        tq_config = TurboQuantConfig(key_bits=3, value_bits=3)
        tq_cache = TurboQuantCache(config=model.config, quant_config=tq_config)
        with torch.no_grad():
            model(**inputs, past_key_values=tq_cache)
        tq_bytes = tq_cache.get_memory_bytes()["compressed"]

        ratio = baseline_bytes / max(tq_bytes, 1)
        print(f"\nCompression ratio: {ratio:.2f}x (baseline={baseline_bytes}, tq={tq_bytes})")
        assert ratio >= 2.0, f"Compression ratio {ratio:.2f}x below 2x minimum"

    def test_output_coherence(self, model_and_tokenizer):
        """TurboQuant output should be coherent (not garbage)."""
        from turboquantkv import TurboQuantCache, TurboQuantConfig

        model, tokenizer, device = model_and_tokenizer
        config = TurboQuantConfig(key_bits=4, value_bits=4)

        for prompt in PROMPTS:
            tq_cache = TurboQuantCache(config=model.config, quant_config=config)
            ids = self._generate(model, tokenizer, device, prompt, cache=tq_cache, max_new_tokens=50)
            text = tokenizer.decode(ids, skip_special_tokens=True)

            # Basic coherence checks
            assert len(text) > len(prompt), "Output should be longer than prompt"
            # Check it's not all the same token repeated
            words = text.split()
            unique_ratio = len(set(words)) / max(len(words), 1)
            assert unique_ratio > 0.15, f"Output too repetitive: {unique_ratio:.1%} unique words"
