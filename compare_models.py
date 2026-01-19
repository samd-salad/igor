#!/usr/bin/env python3
"""Compare response times across different Ollama models."""

import requests
import time
from config import OLLAMA_URL

MODELS = ["qwen3:30b", "qwen3:8b", "qwen3:4b"]

WARMUP_PROMPT = "Say hi."

TEST_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about coffee.",
]

def test_network():
    """Measure network round-trip."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)

def query_model(model: str, prompt: str, timeout: int = 180):
    """Send a prompt to a model and return timing details."""
    start = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 100}
        },
        timeout=timeout
    )
    total_time = time.perf_counter() - start
    r.raise_for_status()
    data = r.json()

    return {
        "total": total_time,
        "load": data.get("load_duration", 0) / 1e9,
        "prompt_eval": data.get("prompt_eval_duration", 0) / 1e9,
        "generation": data.get("eval_duration", 0) / 1e9,
        "tokens": data.get("eval_count", 0),
        "response": data.get("response", "").strip()
    }

def test_model(model: str):
    """Warm up model then run 3 test prompts."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print('='*60)

    # Warmup
    print(f"  Warming up (prompt: \"{WARMUP_PROMPT}\")...", end=" ", flush=True)
    try:
        warmup = query_model(model, WARMUP_PROMPT)
        print(f"loaded in {warmup['load']:.2f}s")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    # Real tests
    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n  Test {i}: \"{prompt}\"")
        try:
            r = query_model(model, prompt)
            tok_s = r['tokens'] / r['generation'] if r['generation'] > 0 else 0
            print(f"    Response: \"{r['response'][:70]}{'...' if len(r['response']) > 70 else ''}\"")
            print(f"    Time: {r['total']:.2f}s | Tokens: {r['tokens']} | Speed: {tok_s:.1f} tok/s")
            results.append(r)
        except Exception as e:
            print(f"    FAILED: {e}")

    return results

def main():
    print(f"Ollama: {OLLAMA_URL}")

    # Network test
    print(f"\nNetwork latency: {test_network()*1000:.1f}ms")

    # Test each model
    all_results = {}
    for model in MODELS:
        results = test_model(model)
        if results:
            all_results[model] = results

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (averaged across 3 prompts, excluding warmup)")
    print("="*60)
    print(f"{'Model':<15} {'Avg Total':>10} {'Avg Gen':>10} {'Avg Tok/s':>10}")
    print("-"*60)

    for model, results in all_results.items():
        avg_total = sum(r['total'] for r in results) / len(results)
        avg_gen = sum(r['generation'] for r in results) / len(results)
        total_tokens = sum(r['tokens'] for r in results)
        total_gen_time = sum(r['generation'] for r in results)
        avg_tok_s = total_tokens / total_gen_time if total_gen_time > 0 else 0
        print(f"{model:<15} {avg_total:>9.2f}s {avg_gen:>9.2f}s {avg_tok_s:>10.1f}")

    print("-"*60)

if __name__ == "__main__":
    main()
