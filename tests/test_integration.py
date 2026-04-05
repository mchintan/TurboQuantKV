"""Integration tests with HuggingFace transformers (requires model download)."""

import pytest
import torch

# Mark all tests in this module as slow (require model download)
pytestmark = pytest.mark.slow


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 (small, ~500MB) for integration testing."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "openai-community/gpt2"
    device = _get_device()
    dtype = torch.float32 if device == "cpu" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()
    return model, tokenizer, device


class TestTransformersIntegration:
    def test_generate_with_turboquant_cache(self, gpt2_model):
        """Basic generation should work with TurboQuantCache."""
        from turboquantkv import TurboQuantCache, TurboQuantConfig

        model, tokenizer, device = gpt2_model
        config = TurboQuantConfig(key_bits=4, value_bits=4)
        cache = TurboQuantCache(config=model.config, quant_config=config)

        inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=20,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(text) > len("The capital of France is")

    def test_generate_with_helper(self, gpt2_model):
        """generate_with_turboquant helper should work."""
        from turboquantkv import TurboQuantConfig, generate_with_turboquant

        model, tokenizer, device = gpt2_model
        config = TurboQuantConfig(key_bits=3, value_bits=3)

        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        outputs = generate_with_turboquant(
            model,
            inputs["input_ids"],
            quant_config=config,
            max_new_tokens=10,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(text) > len("Hello, world!")

    def test_memory_savings(self, gpt2_model):
        """TurboQuant cache should use less memory than baseline."""
        from transformers.cache_utils import DynamicCache
        from turboquantkv import TurboQuantCache, TurboQuantConfig

        model, tokenizer, device = gpt2_model
        prompt = "Once upon a time in a land far far away there lived a brave knight"
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

        # TurboQuant
        tq_config = TurboQuantConfig(key_bits=3, value_bits=3)
        tq_cache = TurboQuantCache(config=model.config, quant_config=tq_config)
        with torch.no_grad():
            model(**inputs, past_key_values=tq_cache)
        tq_bytes = tq_cache.get_memory_bytes()["compressed"]

        assert tq_bytes < baseline_bytes, (
            f"TQ compressed {tq_bytes} should be < baseline {baseline_bytes}"
        )

    def test_different_bit_configs(self, gpt2_model):
        """Various bit configurations should all produce valid output."""
        from turboquantkv import TurboQuantConfig, generate_with_turboquant

        model, tokenizer, device = gpt2_model
        inputs = tokenizer("Test prompt", return_tensors="pt").to(device)

        for key_bits, value_bits in [(2, 2), (3, 3), (4, 4), (4, 2)]:
            config = TurboQuantConfig(key_bits=key_bits, value_bits=value_bits)
            outputs = generate_with_turboquant(
                model, inputs["input_ids"], quant_config=config,
                max_new_tokens=5, do_sample=False,
            )
            assert outputs.shape[-1] > inputs["input_ids"].shape[-1]
