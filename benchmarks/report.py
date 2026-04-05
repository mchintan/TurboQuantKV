"""CLI report generator for TurboQuant vs baseline comparison.

Usage:
    python -m benchmarks.report --model "openai-community/gpt2" --bits 3,4
    python -m benchmarks.report --model "Qwen/Qwen2-0.5B" --bits 2,3,4 --seq-lengths 512,1024
    python -m benchmarks.report --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --bits 3,4 --output-dir ./my_reports
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from turboquantkv import TurboQuantCache, TurboQuantConfig


DEFAULT_PROMPTS = [
    "The future of artificial intelligence is",
    "In a galaxy far far away, there was a civilization that",
    "The most important scientific discovery of the 21st century will be",
    "Once upon a time in a small village, a young inventor created",
    "The key to understanding quantum mechanics is",
]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure_perplexity(model, tokenizer, text: str, cache=None, device="cpu") -> float:
    """Measure perplexity of model on a text string."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, past_key_values=cache, labels=input_ids)

    return torch.exp(outputs.loss).item()


def generate_and_measure(
    model, tokenizer, prompt: str, cache=None, max_new_tokens: int = 50, device="cpu"
) -> tuple[str, float, int]:
    """Generate text and measure throughput.

    Returns: (generated_text, tokens_per_second, num_new_tokens)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start

    new_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    tps = new_tokens / max(elapsed, 1e-6)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text, tps, new_tokens


def run_comparison(
    model, tokenizer, device: str, bit_config: tuple[int, int],
    prompts: list[str], max_new_tokens: int = 50,
) -> dict:
    """Run baseline vs TurboQuant comparison for a single bit configuration."""
    key_bits, value_bits = bit_config
    tq_config = TurboQuantConfig(key_bits=key_bits, value_bits=value_bits)

    results = {
        "key_bits": key_bits,
        "value_bits": value_bits,
        "prompts": [],
        "baseline_tps_avg": 0,
        "tq_tps_avg": 0,
        "token_agreement_pct": 0,
        "memory_baseline_bytes": 0,
        "memory_tq_compressed_bytes": 0,
    }

    total_tps_baseline = 0
    total_tps_tq = 0
    total_tokens = 0
    matching_tokens = 0

    for prompt in prompts:
        # Baseline
        baseline_text, baseline_tps, n_tokens = generate_and_measure(
            model, tokenizer, prompt, cache=None, max_new_tokens=max_new_tokens, device=device
        )
        total_tps_baseline += baseline_tps

        # TurboQuant
        tq_cache = TurboQuantCache(config=model.config, quant_config=tq_config)
        tq_text, tq_tps, n_tokens_tq = generate_and_measure(
            model, tokenizer, prompt, cache=tq_cache, max_new_tokens=max_new_tokens, device=device
        )
        total_tps_tq += tq_tps

        # Token agreement
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        baseline_ids = tokenizer(baseline_text, return_tensors="pt")["input_ids"][0]
        tq_ids = tokenizer(tq_text, return_tensors="pt")["input_ids"][0]
        prompt_len = inputs["input_ids"].shape[-1]
        min_len = min(len(baseline_ids), len(tq_ids))
        if min_len > prompt_len:
            gen_b = baseline_ids[prompt_len:min_len]
            gen_t = tq_ids[prompt_len:min_len]
            total_tokens += len(gen_b)
            matching_tokens += (gen_b == gen_t).sum().item()

        results["prompts"].append({
            "prompt": prompt,
            "baseline_output": baseline_text,
            "tq_output": tq_text,
            "baseline_tps": round(baseline_tps, 1),
            "tq_tps": round(tq_tps, 1),
        })

    # Memory measurement on last prompt
    inputs = tokenizer(prompts[-1], return_tensors="pt").to(device)

    baseline_cache = DynamicCache()
    with torch.no_grad():
        model(**inputs, past_key_values=baseline_cache)
    baseline_bytes = sum(
        l.keys.nelement() * l.keys.element_size() + l.values.nelement() * l.values.element_size()
        for l in baseline_cache.layers
        if hasattr(l, "keys") and l.keys is not None
    )

    tq_cache2 = TurboQuantCache(config=model.config, quant_config=tq_config)
    with torch.no_grad():
        model(**inputs, past_key_values=tq_cache2)
    tq_mem = tq_cache2.get_memory_bytes()

    n_prompts = len(prompts)
    results["baseline_tps_avg"] = round(total_tps_baseline / n_prompts, 1)
    results["tq_tps_avg"] = round(total_tps_tq / n_prompts, 1)
    results["token_agreement_pct"] = round(100 * matching_tokens / max(total_tokens, 1), 1)
    results["memory_baseline_bytes"] = baseline_bytes
    results["memory_tq_compressed_bytes"] = tq_mem["compressed"]
    results["compression_ratio"] = round(baseline_bytes / max(tq_mem["compressed"], 1), 2)

    return results


def generate_html_report(report: dict, output_path: Path) -> None:
    """Generate a self-contained HTML report."""
    configs = report["configs"]

    rows_html = ""
    for cfg in configs:
        rows_html += f"""
        <tr>
            <td>{cfg['key_bits']}K / {cfg['value_bits']}V</td>
            <td>{cfg['compression_ratio']}x</td>
            <td>{cfg['memory_baseline_bytes'] / 1024:.1f} KB</td>
            <td>{cfg['memory_tq_compressed_bytes'] / 1024:.1f} KB</td>
            <td>{cfg['token_agreement_pct']}%</td>
            <td>{cfg['baseline_tps_avg']}</td>
            <td>{cfg['tq_tps_avg']}</td>
        </tr>"""

    samples_html = ""
    for cfg in configs:
        samples_html += f"<h3>{cfg['key_bits']}-bit keys / {cfg['value_bits']}-bit values</h3>"
        for p in cfg["prompts"][:3]:
            samples_html += f"""
            <div class="sample">
                <p><strong>Prompt:</strong> {p['prompt']}</p>
                <p><strong>Baseline:</strong> {p['baseline_output'][:200]}</p>
                <p><strong>TurboQuant:</strong> {p['tq_output'][:200]}</p>
            </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>TurboQuantKV Report - {report['model']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #333; }}
        h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: center; }}
        th {{ background: #16213e; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .meta {{ background: #f0f0f5; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .sample {{ background: #fafafa; border-left: 3px solid #e94560; padding: 10px 15px; margin: 10px 0; }}
        .sample p {{ margin: 5px 0; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>TurboQuantKV Comparison Report</h1>
    <div class="meta">
        <strong>Model:</strong> {report['model']}<br>
        <strong>Device:</strong> {report['device']}<br>
        <strong>Timestamp:</strong> {report['timestamp']}<br>
        <strong>Prompts tested:</strong> {len(configs[0]['prompts']) if configs else 0}
    </div>

    <h2>Summary</h2>
    <table>
        <tr>
            <th>Bit Config</th>
            <th>Compression</th>
            <th>Baseline Mem</th>
            <th>TQ Mem</th>
            <th>Token Agreement</th>
            <th>Baseline TPS</th>
            <th>TQ TPS</th>
        </tr>
        {rows_html}
    </table>

    <h2>Sample Outputs</h2>
    {samples_html}

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 0.85em;">
        Generated by TurboQuantKV benchmark suite
    </footer>
</body>
</html>"""

    output_path.write_text(html)


def main():
    parser = argparse.ArgumentParser(description="TurboQuantKV comparison report generator")
    parser.add_argument("--model", type=str, default="openai-community/gpt2",
                        help="HuggingFace model ID")
    parser.add_argument("--bits", type=str, default="3,4",
                        help="Comma-separated bit widths to test (applied to both K and V)")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to file with one prompt per line")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reports")

    args = parser.parse_args()

    # Parse bit configs
    bit_widths = [int(b.strip()) for b in args.bits.split(",")]
    bit_configs = [(b, b) for b in bit_widths]

    # Load prompts
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().strip().split("\n")
    else:
        prompts = DEFAULT_PROMPTS

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    device = get_device()
    dtype = torch.float32 if device == "cpu" else torch.float16

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if device != "cpu":
        model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run comparisons
    report = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "configs": [],
    }

    for key_bits, value_bits in bit_configs:
        print(f"\nRunning {key_bits}-bit keys / {value_bits}-bit values...")
        result = run_comparison(
            model, tokenizer, device,
            bit_config=(key_bits, value_bits),
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
        report["configs"].append(result)
        print(f"  Compression: {result['compression_ratio']}x")
        print(f"  Token agreement: {result['token_agreement_pct']}%")
        print(f"  TPS: baseline={result['baseline_tps_avg']}, tq={result['tq_tps_avg']}")

    # Save reports
    model_slug = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{model_slug}_{timestamp}.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nJSON report: {json_path}")

    html_path = output_dir / f"{model_slug}_{timestamp}.html"
    generate_html_report(report, html_path)
    print(f"HTML report: {html_path}")

    # Console summary
    print("\n" + "=" * 70)
    print(f"{'Config':<12} {'Compress':<10} {'Agreement':<12} {'BL TPS':<10} {'TQ TPS':<10}")
    print("-" * 70)
    for cfg in report["configs"]:
        print(
            f"{cfg['key_bits']}K/{cfg['value_bits']}V"
            f"{'':>6} {cfg['compression_ratio']}x"
            f"{'':>6} {cfg['token_agreement_pct']}%"
            f"{'':>7} {cfg['baseline_tps_avg']}"
            f"{'':>6} {cfg['tq_tps_avg']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
