"""Basic usage example for TurboQuantKV.

Compares baseline generation vs TurboQuant-compressed KV cache generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from turboquantkv import TurboQuantCache, TurboQuantConfig, generate_with_turboquant


def main():
    # Setup
    model_name = "openai-community/gpt2"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Baseline generation
    print("\n--- Baseline (DynamicCache) ---")
    with torch.no_grad():
        baseline_out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)
    print(baseline_text)

    # 2. TurboQuant 4-bit
    print("\n--- TurboQuant 4-bit ---")
    config_4bit = TurboQuantConfig(key_bits=4, value_bits=4)
    tq_out = generate_with_turboquant(
        model, inputs["input_ids"], quant_config=config_4bit, max_new_tokens=50, do_sample=False
    )
    print(tokenizer.decode(tq_out[0], skip_special_tokens=True))

    # 3. TurboQuant 3-bit
    print("\n--- TurboQuant 3-bit ---")
    config_3bit = TurboQuantConfig(key_bits=3, value_bits=3)
    tq_out_3 = generate_with_turboquant(
        model, inputs["input_ids"], quant_config=config_3bit, max_new_tokens=50, do_sample=False
    )
    print(tokenizer.decode(tq_out_3[0], skip_special_tokens=True))

    # 4. Memory comparison
    print("\n--- Memory Comparison ---")
    baseline_cache = DynamicCache()
    with torch.no_grad():
        model(**inputs, past_key_values=baseline_cache)
    baseline_bytes = sum(
        l.keys.nelement() * l.keys.element_size() + l.values.nelement() * l.values.element_size()
        for l in baseline_cache.layers
        if hasattr(l, "keys") and l.keys is not None
    )

    tq_cache = TurboQuantCache(config=model.config, quant_config=config_3bit)
    with torch.no_grad():
        model(**inputs, past_key_values=tq_cache)
    tq_mem = tq_cache.get_memory_bytes()

    print(f"Baseline KV cache:     {baseline_bytes / 1024:.1f} KB")
    print(f"TurboQuant compressed: {tq_mem['compressed'] / 1024:.1f} KB")
    print(f"Compression ratio:     {baseline_bytes / max(tq_mem['compressed'], 1):.2f}x")


if __name__ == "__main__":
    main()
