# TurboQuantKV

A Python library implementing the [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) paper (Google Research, ICLR 2026) for extreme KV cache compression in LLM inference.

TurboQuantKV compresses the key-value cache to 2-4 bits per value with minimal accuracy loss, achieving **3-5x memory reduction** with no calibration data or fine-tuning required. It works as a drop-in replacement for HuggingFace transformers' `DynamicCache`.

## How It Works

TurboQuant uses a two-stage compression pipeline:

### Stage 1: PolarQuant

1. **Rotate** each KV vector using the Walsh-Hadamard Transform (WHT), a fast orthogonal rotation (O(d log d))
2. After rotation, all coordinates become nearly identically distributed (Beta distribution) - this is a property of high-dimensional geometry
3. **Scalar quantize** each coordinate independently using Lloyd-Max optimal centroids precomputed from the known distribution
4. Store the **quantized indices** (2-4 bits per coordinate) and the **vector norm** (float32)

The key insight: because the post-rotation distribution is mathematically known, the quantizer centroids are optimal without any calibration data.

### Stage 2: QJL (Optional)

The Quantized Johnson-Lindenstrauss step corrects systematic bias in attention scores:

1. Compute the quantization **residual** (difference between original and reconstructed vector)
2. Project through a random sign matrix and store only the **signs** (1 bit each)
3. At inference, use these signs to compute an unbiased correction to attention scores

QJL is disabled by default - empirical results show the MSE-only approach (PolarQuant alone) often performs better at low bit budgets because QJL adds variance that softmax amplifies.

## Quick Start

### Installation

```bash
pip install torch transformers
pip install -e .
```

### Basic Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantkv import TurboQuantCache, TurboQuantConfig

# Load any HuggingFace model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Create TurboQuant compressed cache
config = TurboQuantConfig(key_bits=4, value_bits=3)
cache = TurboQuantCache(config=model.config, quant_config=config)

# Generate - just pass the cache as past_key_values
inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### One-Line Helper

```python
from turboquantkv import TurboQuantConfig, generate_with_turboquant

config = TurboQuantConfig(key_bits=4, value_bits=3)
outputs = generate_with_turboquant(model, input_ids, quant_config=config, max_new_tokens=100)
```

### Measuring Memory Savings

```python
cache = TurboQuantCache(config=model.config, quant_config=TurboQuantConfig(key_bits=3, value_bits=3))
with torch.no_grad():
    model(**inputs, past_key_values=cache)

mem = cache.get_memory_bytes()
print(f"Compressed KV cache: {mem['compressed'] / 1024:.1f} KB")
```

## Configuration

```python
TurboQuantConfig(
    key_bits=4,           # Bits for key quantization: 2, 3, or 4
    value_bits=4,         # Bits for value quantization: 2, 3, or 4
    rotation_type="wht",  # "wht" (Walsh-Hadamard) or "random" (orthogonal matrix)
    block_size=32,        # WHT block size (power of 2, must divide head_dim)
    use_qjl=False,        # Enable QJL residual correction
    qjl_dim=64,           # Number of QJL random projections
    seed=42,              # Random seed for reproducibility
)
```

**Recommended configurations:**

| Use Case | Config | Compression | Notes |
|----------|--------|-------------|-------|
| Quality-first | `key_bits=4, value_bits=4` | ~3.5x | Minimal accuracy loss |
| Balanced | `key_bits=4, value_bits=3` | ~4x | Good quality/compression tradeoff |
| Max compression | `key_bits=3, value_bits=3` | ~4.5x | Some accuracy loss on small models |
| Extreme | `key_bits=2, value_bits=2` | ~6x+ | Best for large models (128+ head_dim) |

## Benchmarks

### Running Reports

Generate a full comparison report for any HuggingFace model:

```bash
# Basic report
python -m benchmarks.report --model "openai-community/gpt2" --bits 3,4

# Full sweep with custom prompts
python -m benchmarks.report \
  --model "Qwen/Qwen2-0.5B" \
  --bits 2,3,4 \
  --max-new-tokens 50 \
  --output-dir ./my_reports

# Different model
python -m benchmarks.report \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --bits 3,4
```

Reports are generated in both JSON (raw metrics) and HTML (formatted with tables) formats:

```
benchmarks/reports/
  openai-community_gpt2_20260405_101549.json
  openai-community_gpt2_20260405_101549.html
```

### Report Metrics

Each report compares baseline `DynamicCache` vs `TurboQuantCache` across:

- **Compression ratio** - how much smaller the KV cache is
- **Token agreement** - % of generated tokens matching baseline
- **Throughput** - tokens per second (baseline vs TurboQuant)
- **Memory usage** - bytes for baseline vs compressed storage
- **Sample outputs** - side-by-side text comparisons

### GPT-2 Results (Apple Silicon, MPS)

| Config | Compression | Token Agreement |
|--------|-------------|-----------------|
| 3-bit K/V | 4.57x | varies by prompt |
| 4-bit K/V | 3.56x | varies by prompt |

Note: GPT-2 has `head_dim=64`, which is small. Larger models with `head_dim=128` (Llama, Qwen, Mistral) benefit significantly more from TurboQuant due to tighter post-rotation coordinate concentration.

## Architecture

```
turboquantkv/
  config.py                    # TurboQuantConfig dataclass
  core/
    rotation.py                # Fast Walsh-Hadamard Transform + random rotation
    codebook.py                # Lloyd-Max centroid precomputation (Beta distribution)
    quantizer.py               # PolarQuant encoder/decoder + QJL corrector
    bitpack.py                 # 2/3/4-bit packing into uint8
  cache/
    turboquant_layer.py        # CacheLayerMixin - per-layer compressed storage
    turboquant_cache.py        # Cache subclass - the main entry point
  integration/
    transformers_patch.py      # generate_with_turboquant() helper
```

### Key Design Decisions

- **Walsh-Hadamard Transform** over random rotation: 15-60x better empirical performance at sub-4-bit compression (community finding from llama.cpp implementations). WHT is also deterministic and O(d log d) vs O(d^2) for matrix multiplication.
- **Block size 32**: Better Flash Attention parallelism than the paper's 128. Divides all common head dims (64, 128, 256).
- **QJL off by default**: MSE-only quantization empirically outperforms MSE+QJL at low bit budgets because softmax amplifies QJL's added variance.
- **Float32 norms**: Key norms can reach 1000+ in some models, exceeding fp16 range (65504). Float32 prevents overflow.
- **Incremental dequantization**: On each decode step, only the new token is dequantized and concatenated, rather than re-dequantizing the entire history.

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (includes model download for integration tests)
pytest tests/ -v

# Run only unit tests (no model download)
pytest tests/test_rotation.py tests/test_codebook.py tests/test_bitpack.py tests/test_quantizer.py tests/test_cache.py -v

# Run pre/post comparison tests
pytest tests/test_comparison.py -v
```

### Test Coverage

| Test File | What It Tests |
|-----------|--------------|
| `test_rotation.py` | WHT self-inverse, norm preservation, batch dims, fp16 |
| `test_codebook.py` | Centroid sorting, symmetry, range, convergence |
| `test_bitpack.py` | 2/3/4-bit pack/unpack round-trips |
| `test_quantizer.py` | Encode/decode shapes, MSE bounds, QJL, KV cache shapes |
| `test_cache.py` | Layer updates, prefill+decode, mask sizes, compression |
| `test_integration.py` | End-to-end generation with GPT-2, memory savings |
| `test_comparison.py` | Baseline vs TurboQuant: token agreement, compression ratio, coherence |

## Compatibility

- **Python**: 3.10+
- **PyTorch**: 2.1+
- **Transformers**: 4.45+
- **Devices**: CPU, CUDA, MPS (Apple Silicon)
- **Models**: Any HuggingFace causal LM with standard attention (GPT-2, Llama, Qwen, Mistral, Gemma, etc.)

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/html/2504.19874v1) (ICLR 2026)
- [TurboQuant blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (Google Research)
- [llama.cpp TurboQuant discussion](https://github.com/ggml-org/llama.cpp/discussions/20969) (community implementations)

## License

MIT
