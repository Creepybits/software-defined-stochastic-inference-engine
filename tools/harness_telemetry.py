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
        self.current_gear = "HIGH_GEAR (SUB-BYTE)"

    def update(self, entropy_val: float) -> tuple[str, float]:
        self.entropy_history.append(entropy_val)
        if len(self.entropy_history) > self.window_size:
            self.entropy_history.pop(0)

        rolling_h = sum(self.entropy_history) / len(self.entropy_history)

        if self.current_gear == "HIGH_GEAR (SUB-BYTE)":
            if rolling_h > (self.theta_high + self.h):
                self.current_gear = "LOW_GEAR (FP16)"
        else:
            if rolling_h < (self.theta_low - self.h):
                self.current_gear = "HIGH_GEAR (SUB-BYTE)"

        return self.current_gear, rolling_h


def calculate_shannon_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * (log_probs / math.log(2.0)), dim=-1)
    return entropy.item()


def log_telemetry_entry(data: dict):
    telemetry_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry")
    os.makedirs(telemetry_dir, exist_ok=True)

    json_path = os.path.join(telemetry_dir, "telemetry_harness_ledger.json")
    csv_path = os.path.join(telemetry_dir, "telemetry_harness_ledger.csv")

    # 1. Append to JSON array
    ledger = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                ledger = json.load(f)
        except Exception:
            ledger = []
    ledger.append(data)
    with open(json_path, "w") as f:
        json.dump(ledger, f, indent=2)

    # 2. Append to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "model_id", "tokens_generated", "throughput_tok_sec",
            "avg_power_watts", "total_energy_joules", "energy_per_tok_joules",
            "high_gear_pct", "low_gear_pct"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": data["timestamp"],
            "model_id": data["model_id"],
            "tokens_generated": data["metrics"]["tokens_generated"],
            "throughput_tok_sec": data["metrics"]["throughput_tok_per_sec"],
            "avg_power_watts": data["metrics"]["average_power_watts"],
            "total_energy_joules": data["metrics"]["total_energy_joules"],
            "energy_per_tok_joules": data["metrics"]["energy_per_token_joules"],
            "high_gear_pct": data["gear_distribution"]["high_gear_pct"],
            "low_gear_pct": data["gear_distribution"]["low_gear_pct"]
        })
    print(f"📁 Benchmark recorded to {json_path} and {csv_path}")


def run_telemetry_benchmark(model_id="meta-llama/Llama-3.1-8B-Instruct", prompt="Explain the physics behind how a semiconductor transistor functions at the quantum level.", max_new_tokens=60):
    print("=" * 85)
    print("⚡ SDSIE HARDWARE POWER TELEMETRY PROFILER (RTX 5090)")
    print(f"📦 Model: {model_id}")
    print("=" * 85)

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nAllocating model weights on 5090 Blackwell VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    clutch = SchmittTriggerClutch(theta_low=1.0, theta_high=2.5, h=0.25, window_size=3)

    print(f"\nPrompt: \"{prompt}\"")
    print("\n" + "-" * 90)
    print(f"{'Token':<14} | {'H_t (bits)':<10} | {'Gear State':<18} | {'Lat (ms)':<8} | {'Power (W)':<9} | {'Energy (mJ/tok)'}")
    print("-" * 90)

    generated_ids = input_ids
    total_energy_joules = 0.0
    total_latency_sec = 0.0
    tokens_generated = 0
    gear_counts = {"HIGH_GEAR (SUB-BYTE)": 0, "LOW_GEAR (FP16)": 0}

    with torch.inference_mode():
        for step in range(max_new_tokens):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :]

            h_t = calculate_shannon_entropy(next_token_logits)
            gear, smoothed_h = clutch.update(h_t)
            gear_counts[gear] = gear_counts.get(gear, 0) + 1

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token_str = tokenizer.decode(next_token[0], clean_up_tokenization_spaces=False)

            torch.cuda.synchronize()
            t_end = time.perf_counter()

            step_lat_sec = t_end - t_start
            step_lat_ms = step_lat_sec * 1000.0
            step_pwr_w = monitor.get_average_power(t_start, t_end)
            step_energy_mj = step_pwr_w * step_lat_ms

            total_energy_joules += (step_energy_mj / 1000.0)
            total_latency_sec += step_lat_sec
            tokens_generated += 1

            print(f"{repr(token_str):<14} | {h_t:<10.4f} | {gear:<18} | {step_lat_ms:<8.2f} | {step_pwr_w:<9.1f} | {step_energy_mj:<8.2f} mJ")

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    monitor.stop()
    monitor.close()

    avg_tok_per_sec = tokens_generated / total_latency_sec if total_latency_sec > 0 else 0
    avg_mj_per_tok = (total_energy_joules / tokens_generated) * 1000.0 if tokens_generated > 0 else 0
    avg_pwr_w = (total_energy_joules / total_latency_sec) if total_latency_sec > 0 else 0
    high_pct = (gear_counts.get("HIGH_GEAR (SUB-BYTE)", 0) / tokens_generated) * 100.0 if tokens_generated > 0 else 0
    low_pct = (gear_counts.get("LOW_GEAR (FP16)", 0) / tokens_generated) * 100.0 if tokens_generated > 0 else 0

    print("-" * 90)
    print("\n📊 TELEMETRY BENCHMARK SUMMARY:")
    print(f" • Tokens Generated     : {tokens_generated} tokens")
    print(f" • Average Throughput   : {avg_tok_per_sec:.2f} tokens/sec")
    print(f" • Average GPU Power    : {avg_pwr_w:.2f} Watts")
    print(f" • Total Energy Used    : {total_energy_joules:.4f} Joules")
    print(f" • Energy Efficiency    : {avg_mj_per_tok:.2f} mJ / token (Joules/token: {total_energy_joules/tokens_generated:.6f})")
    print(f" • Gear Distribution    : High Gear: {high_pct:.1f}% | Low Gear: {low_pct:.1f}%")
    print("=" * 90)

    # Persist entry to ledger
    log_telemetry_entry({
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5090 32GB",
            "driver": "CUDA 13.3 / Driver 610.88"
        },
        "model_id": model_id,
        "prompt": prompt,
        "gear_distribution": {
            "high_gear_pct": round(high_pct, 2),
            "low_gear_pct": round(low_pct, 2)
        },
        "metrics": {
            "tokens_generated": tokens_generated,
            "throughput_tok_per_sec": round(avg_tok_per_sec, 2),
            "average_power_watts": round(avg_pwr_w, 2),
            "total_energy_joules": round(total_energy_joules, 4),
            "energy_per_token_joules": round(total_energy_joules / tokens_generated, 6),
            "energy_per_token_mj": round(avg_mj_per_tok, 2)
        }
    })

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    run_telemetry_benchmark(model_id=target)
