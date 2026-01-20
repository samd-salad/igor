#!/usr/bin/env python3
"""Compare response times, tool-calling accuracy, and GPU usage across Ollama models."""

import requests
import time
import json
import subprocess
import threading
from server.config import OLLAMA_URL
from prompt import SYSTEM_PROMPT

MODELS = [
    "gpt-oss:20b",
    "qwen3:30b",
    "qwen3:4b"
]

WARMUP_PROMPT = "Say hi."

# Tool definitions for testing
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a timer for a specified duration",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Timer duration in seconds"
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the timer"
                    }
                },
                "required": ["duration_seconds"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Test cases: (prompt, expected_tool, expected_params_substring)
TOOL_TEST_CASES = [
    ("Set a timer for 5 minutes", "set_timer", "300"),
    ("Set a 10 minute pasta timer", "set_timer", "600"),
    ("Timer for 30 seconds", "set_timer", "30"),
    ("What is 256 times 48", "calculate", "256"),
    ("Calculate 25 times 4", "calculate", "25"),
    ("How much is 100 divided by 8", "calculate", "100"),
]

# Replace {persistent_memory} placeholder with empty for testing
TEST_SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{persistent_memory}", "No persistent memories yet.")


class GPUMonitor:
    """Monitor GPU usage during inference."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.samples = []
        self.running = False
        self.thread = None

    def _sample(self):
        """Sample GPU metrics using Windows perf counters, rocm-smi, or nvidia-smi."""
        # Try Windows GPU Engine performance counters (works for AMD on Windows)
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | "
                 "Where-Object {$_.InstanceName -like '*engtype_3D*'} | "
                 "Measure-Object -Property CookedValue -Average | "
                 "Select-Object -ExpandProperty Average"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_util = float(result.stdout.strip())
                return {
                    "gpu_util": gpu_util,
                    "mem_used": 0,
                    "mem_total": 0,
                    "power": 0
                }
        except Exception:
            pass

        # Try rocm-smi (Linux AMD)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--csv"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    for line in lines[1:]:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            return {
                                "gpu_util": float(parts[0].strip().replace('%', '')) if parts[0].strip() else 0,
                                "mem_used": float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0,
                                "mem_total": 0,
                                "power": 0
                            }
        except Exception:
            pass

        # Fall back to NVIDIA
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 4:
                    return {
                        "gpu_util": float(parts[0]),
                        "mem_used": float(parts[1]),
                        "mem_total": float(parts[2]),
                        "power": float(parts[3]) if parts[3] != "[N/A]" else 0
                    }
        except Exception:
            pass
        return None

    def _monitor_loop(self):
        """Continuous monitoring loop."""
        while self.running:
            sample = self._sample()
            if sample:
                self.samples.append(sample)
            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring and return stats."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

        if not self.samples:
            return None

        return {
            "gpu_util_avg": sum(s["gpu_util"] for s in self.samples) / len(self.samples),
            "gpu_util_max": max(s["gpu_util"] for s in self.samples),
            "mem_used_avg": sum(s["mem_used"] for s in self.samples) / len(self.samples),
            "mem_used_max": max(s["mem_used"] for s in self.samples),
            "mem_total": self.samples[0]["mem_total"],
            "power_avg": sum(s["power"] for s in self.samples) / len(self.samples),
            "power_max": max(s["power"] for s in self.samples),
            "sample_count": len(self.samples)
        }


def has_think_mode(model: str) -> bool:
    """Check if model supports think mode."""
    model_lower = model.lower()
    return "qwen" in model_lower or "deepseek" in model_lower


def test_network():
    """Measure network round-trip."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        r.raise_for_status()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def query_with_tools(model: str, prompt: str, tools: list, timeout: int = 180, think: bool = None, gpu_monitor: GPUMonitor = None):
    """Send a prompt with tools using the chat API."""

    options = {"num_predict": 200}

    # For qwen/deepseek: use think option to control reasoning
    if think is not None and has_think_mode(model):
        options["think"] = think

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "tools": tools,
        "stream": False,
        "options": options
    }

    if gpu_monitor:
        gpu_monitor.start()

    start = time.perf_counter()
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=timeout
    )
    total_time = time.perf_counter() - start

    gpu_stats = None
    if gpu_monitor:
        gpu_stats = gpu_monitor.stop()

    r.raise_for_status()
    data = r.json()

    message = data.get("message", {})
    tool_calls = message.get("tool_calls", [])
    content = message.get("content", "")

    return {
        "total": total_time,
        "load": data.get("load_duration", 0) / 1e9,
        "prompt_eval": data.get("prompt_eval_duration", 0) / 1e9,
        "generation": data.get("eval_duration", 0) / 1e9,
        "tokens": data.get("eval_count", 0),
        "tool_calls": tool_calls,
        "content": content.strip(),
        "gpu_stats": gpu_stats
    }


def check_tool_call(result: dict, expected_tool: str, expected_param: str) -> tuple:
    """Check if the model called the correct tool with expected params."""
    tool_calls = result.get("tool_calls", [])

    if not tool_calls:
        return False, "no_tool_call"

    first_call = tool_calls[0]
    func = first_call.get("function", {})
    name = func.get("name", "")
    args = func.get("arguments", {})
    args_str = json.dumps(args) if isinstance(args, dict) else str(args)

    if name != expected_tool:
        return False, f"wrong_tool:{name}"

    if expected_param not in args_str:
        return False, f"wrong_params:{args_str[:50]}"

    return True, f"correct:{name}({args_str[:30]})"


def warmup_model(model: str):
    """Load model into memory and measure GPU."""
    print(f"  Loading {model}...", end=" ", flush=True)
    gpu_monitor = GPUMonitor()
    try:
        gpu_monitor.start()
        start = time.perf_counter()
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": WARMUP_PROMPT}],
                "stream": False
            },
            timeout=180
        )
        total = time.perf_counter() - start
        gpu_stats = gpu_monitor.stop()
        r.raise_for_status()
        data = r.json()
        load_time = data.get("load_duration", 0) / 1e9

        gpu_info = ""
        if gpu_stats:
            gpu_info = f" | VRAM: {gpu_stats['mem_used_max']:.0f}MB"

        print(f"loaded in {load_time:.1f}s (total: {total:.1f}s){gpu_info}")
        return {"load_time": load_time, "gpu_stats": gpu_stats}
    except Exception as e:
        gpu_monitor.stop()
        print(f"FAILED: {e}")
        return None


def test_tool_calling(model: str, think: bool = None):
    """Test tool calling accuracy for a model."""
    think_label = ""
    if has_think_mode(model):
        think_label = " (think=True)" if think else " (think=False)"

    print(f"\n  Tool calling tests{think_label}:")

    results = []
    gpu_monitor = GPUMonitor()

    for prompt, expected_tool, expected_param in TOOL_TEST_CASES:
        try:
            r = query_with_tools(model, prompt, TOOLS, think=think, gpu_monitor=gpu_monitor)
            correct, detail = check_tool_call(r, expected_tool, expected_param)
            status = "OK" if correct else "FAIL"

            gpu_info = ""
            if r.get("gpu_stats"):
                gpu_info = f" | GPU:{r['gpu_stats']['gpu_util_avg']:.0f}%"

            print(f"    [{status}] \"{prompt[:30]}...\" -> {detail} ({r['total']:.2f}s{gpu_info})")
            results.append({
                "prompt": prompt,
                "correct": correct,
                "detail": detail,
                "time": r["total"],
                "tokens": r["tokens"],
                "gpu_stats": r.get("gpu_stats")
            })
        except Exception as e:
            print(f"    [ERR] \"{prompt[:30]}...\" -> {e}")
            results.append({
                "prompt": prompt,
                "correct": False,
                "detail": f"error:{e}",
                "time": 0,
                "tokens": 0,
                "gpu_stats": None
            })

    return results


def main():
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Testing {len(MODELS)} models for tool-calling accuracy")
    print(f"Tools: set_timer, calculate")
    print(f"Test cases: {len(TOOL_TEST_CASES)}")
    print(f"Using full Dr. Butts system prompt")

    # Check GPU availability (PyTorch/ROCm, rocm-smi, or nvidia-smi)
    gpu_available = False
    gpu_name = None

    # Try PyTorch first (works for AMD ROCm on Windows)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            print(f"GPU: {gpu_name} ({mem_total:.0f}GB) via PyTorch/ROCm")
            gpu_available = True
    except Exception:
        pass

    # Try rocm-smi (Linux AMD)
    if not gpu_available:
        try:
            result = subprocess.run(["rocm-smi", "--showproductname"],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "GPU" in line or "Card" in line or "series" in line.lower():
                        gpu_name = line.strip()
                        break
                if not gpu_name:
                    gpu_name = "AMD GPU (ROCm)"
                print(f"GPU: {gpu_name}")
                gpu_available = True
        except Exception:
            pass

    # Fall back to NVIDIA
    if not gpu_available:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                print(f"GPU: {gpu_name}")
                gpu_available = True
        except Exception:
            pass

    if not gpu_available:
        print("GPU: Not detected")

    # Network test
    print(f"\nNetwork latency: {test_network()*1000:.1f}ms")

    all_results = {}

    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model}")
        print('='*70)

        warmup_data = warmup_model(model)
        if warmup_data is None:
            continue

        if has_think_mode(model):
            # Test with thinking enabled
            results_think = test_tool_calling(model, think=True)
            # Test with thinking disabled
            results_no_think = test_tool_calling(model, think=False)
            all_results[model] = {
                "warmup": warmup_data,
                "think_true": results_think,
                "think_false": results_no_think
            }
        else:
            # Non-thinking models
            results = test_tool_calling(model, think=None)
            all_results[model] = {
                "warmup": warmup_data,
                "results": results
            }

    # Summary
    print("\n" + "="*90)
    print("TOOL-CALLING SUMMARY")
    print("="*90)
    header = f"{'Model':<20} {'Mode':<12} {'Accuracy':>8} {'Avg Time':>9} {'Tokens':>7}"
    if gpu_available:
        header += f" {'GPU%':>6} {'VRAM MB':>8}"
    print(header)
    print("-"*90)

    for model, data in all_results.items():
        warmup = data.get("warmup", {})
        vram = warmup.get("gpu_stats", {}).get("mem_used_max", 0) if warmup.get("gpu_stats") else 0

        if has_think_mode(model):
            for mode, results in [("think=True", data.get("think_true", [])),
                                   ("think=False", data.get("think_false", []))]:
                if results:
                    correct = sum(1 for r in results if r["correct"])
                    total = len(results)
                    accuracy = f"{correct}/{total}"
                    avg_time = sum(r["time"] for r in results) / len(results) if results else 0
                    avg_tokens = sum(r["tokens"] for r in results) / len(results) if results else 0

                    gpu_samples = [r["gpu_stats"] for r in results if r.get("gpu_stats")]
                    avg_gpu = sum(s["gpu_util_avg"] for s in gpu_samples) / len(gpu_samples) if gpu_samples else 0

                    line = f"{model:<20} {mode:<12} {accuracy:>8} {avg_time:>8.2f}s {avg_tokens:>7.0f}"
                    if gpu_available:
                        line += f" {avg_gpu:>5.0f}% {vram:>7.0f}"
                    print(line)
        else:
            results = data.get("results", [])
            if results:
                correct = sum(1 for r in results if r["correct"])
                total = len(results)
                accuracy = f"{correct}/{total}"
                avg_time = sum(r["time"] for r in results) / len(results) if results else 0
                avg_tokens = sum(r["tokens"] for r in results) / len(results) if results else 0

                gpu_samples = [r["gpu_stats"] for r in results if r.get("gpu_stats")]
                avg_gpu = sum(s["gpu_util_avg"] for s in gpu_samples) / len(gpu_samples) if gpu_samples else 0

                line = f"{model:<20} {'-':<12} {accuracy:>8} {avg_time:>8.2f}s {avg_tokens:>7.0f}"
                if gpu_available:
                    line += f" {avg_gpu:>5.0f}% {vram:>7.0f}"
                print(line)

    print("-"*90)


if __name__ == "__main__":
    main()
