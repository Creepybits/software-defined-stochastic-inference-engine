import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import math
import threading
import pynvml
import sys
import os
import json
import csv
from datetime import datetime

class NVMLPowerMonitor:
    def __init__(self, device_index=0, poll_rate_hz=100):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval = 1.0 / poll_rate_hz
        self.running = False
        self.lock = threading.Lock()
        self.power_samples = []
        self.thread = None

    def start(self):
        self.running = True
        self.power_samples = []
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()

    def _poll_loop(self):
        while self.running:
            try:
                pwr_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                with self.lock:
                    self.power_samples.append((time.perf_counter(), pwr_mw / 1000.0))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def get_average_power(self, start_time, end_time):
        with self.lock:
            window_samples = [p for t, p in self.power_samples if start_time <= t <= end_time]
        if not window_samples:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except Exception:
                return 0.0
        return sum(window_samples) / len(window_samples)

    def close(self):
        self.stop()
        pynvml.nvmlShutdown()


class SchmittTriggerClutch:
    def __init__(self, theta_low=1.0, theta_high=2.5, h=0.25, window_size=3):
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.h = h
        self.window_size = window_size
        self.entropy_history = []
        self.current_gear = "HIGH_GEAR"

    def update(self, entropy_val: float) -> tuple[str, float]:
        self.entropy_history.append(entropy_val)
        if len(self.entropy_history) > self.window_size:
            self.entropy_history.pop(0)
        rolling_h = sum(self.entropy_history) / len(self.entropy_history)

        if self.current_gear == "HIGH_GEAR":
            if rolling_h > (self.theta_high + self.h):
                self.current_gear = "LOW_GEAR"
        else:
            if rolling_h < (self.theta_low - self.h):
                self.current_gear = "HIGH_GEAR"

        return self.current_gear, rolling_h


def calculate_shannon_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * (log_probs / math.log(2.0)), dim=-1)
    return entropy.item()


def run_cognitive_suite(model_id="meta-llama/Llama-3.1-8B-Instruct"):
    output_dir = os.path.expanduser("~/sdsie/benchmarks")
    os.makedirs(output_dir, exist_ok=True)
    master_csv = os.path.join(output_dir, "master_telemetry.csv")

    test_prompts = [
        {"category": "Reasoning / Math", "prompt": "Solve this step-by-step: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "max_tokens": 75},
        {"category": "Code / Syntax", "prompt": "Write a clean Python function to calculate the Fibonacci sequence up to N using memoization with docstrings.", "max_tokens": 75},
        {"category": "Creative Narrative", "prompt": "Write the opening paragraph of a cyberpunk noir detective story set in a rain-slicked mega-city.", "max_tokens": 75}
    ]

    print("=" * 95)
    print("🧠 SDSIE MULTI-PROMPT COGNITIVE TELEMETRY BENCHMARK")
    print(f"📦 Model: {model_id} | GPU: RTX 5090 32GB (CUDA 13.3)")
    print("=" * 95)

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading weights into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()

    suite_results = []
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, test in enumerate(test_prompts, 1):
        category = test["category"]
        prompt = test["prompt"]
        max_tokens = test["max_tokens"]

        print(f"\n[{idx}/3] RUNNING TEST: {category.upper()}")
        print(f"Prompt: \"{prompt}\"")
        print("-" * 95)

        clutch = SchmittTriggerClutch(theta_low=1.0, theta_high=2.5, h=0.25, window_size=3)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        generated_ids = inputs["input_ids"]

        total_energy_j = 0.0
        total_lat_s = 0.0
        tokens_gen = 0
        gear_counts = {"HIGH_GEAR": 0, "LOW_GEAR": 0}

        with torch.inference_mode():
            for step in range(max_tokens):
                torch.cuda.synchronize()
                t0 = time.perf_counter()

                outputs = model(generated_ids)
                logits = outputs.logits[:, -1, :]

                h_t = calculate_shannon_entropy(logits)
                gear, smoothed_h = clutch.update(h_t)
                gear_counts[gear] += 1

                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
                tok_str = tokenizer.decode(next_tok[0], clean_up_tokenization_spaces=False)

                torch.cuda.synchronize()
                t1 = time.perf_counter()

                step_lat = t1 - t0
                step_pwr = monitor.get_average_power(t0, t1)
                step_energy_mj = step_pwr * (step_lat * 1000.0)

                total_energy_j += (step_energy_mj / 1000.0)
                total_lat_s += step_lat
                tokens_gen += 1
                generated_ids = torch.cat([generated_ids, next_tok], dim=-1)

                if next_tok.item() == tokenizer.eos_token_id:
                    break

        tok_sec = tokens_gen / total_lat_s if total_lat_s > 0 else 0
        avg_w = total_energy_j / total_lat_s if total_lat_s > 0 else 0
        j_per_tok = total_energy_j / tokens_gen if tokens_gen > 0 else 0
        high_pct = (gear_counts["HIGH_GEAR"] / tokens_gen) * 100.0 if tokens_gen > 0 else 0
        low_pct = (gear_counts["LOW_GEAR"] / tokens_gen) * 100.0 if tokens_gen > 0 else 0

        res = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "tokens": tokens_gen,
            "throughput_tok_sec": round(tok_sec, 2),
            "avg_power_watts": round(avg_w, 2),
            "total_energy_joules": round(total_energy_j, 4),
            "joules_per_token": round(j_per_tok, 6),
            "high_gear_pct": round(high_pct, 1),
            "low_gear_pct": round(low_pct, 1)
        }
        suite_results.append(res)

        print(f"Output: {tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)[:100]}...")
        print(f"Result: {tok_sec:.2f} tok/s | {avg_w:.1f} W | {j_per_tok:.4f} J/tok | High Gear: {high_pct:.1f}%")

    monitor.stop()
    monitor.close()

    # Append to Master CSV
    file_exists = os.path.exists(master_csv)
    with open(master_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "category", "tokens", "throughput_tok_sec",
            "avg_power_watts", "total_energy_joules", "joules_per_token",
            "high_gear_pct", "low_gear_pct"
        ])
        if not file_exists:
            writer.writeheader()
        for r in suite_results:
            writer.writerow(r)

    # Save Session JSON
    session_json = os.path.join(output_dir, f"session_{session_timestamp}.json")
    with open(session_json, "w") as f:
        json.dump(suite_results, f, indent=2)

    # Print Final Summary Comparison Table
    print("\n" + "=" * 95)
    print("📊 COGNITIVE SUITE COMPARISON SUMMARY")
    print("=" * 95)
    print(f"{'Category':<22} | {'Tokens':<8} | {'Tok/Sec':<9} | {'Power (W)':<10} | {'Joules/Tok':<12} | {'High Gear %':<12}")
    print("-" * 95)
    for r in suite_results:
        print(f"{r['category']:<22} | {r['tokens']:<8} | {r['throughput_tok_sec']:<9.2f} | {r['avg_power_watts']:<10.1f} | {r['joules_per_token']:<12.4f} | {r['high_gear_pct']:<11.1f}%")
    print("=" * 95)
    print(f"💾 All logs safely preserved in: {output_dir}/")

if __name__ == "__main__":
    run_cognitive_suite()
