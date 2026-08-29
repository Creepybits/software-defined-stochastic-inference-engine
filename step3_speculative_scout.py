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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "tools", "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
MASTER_CSV = os.path.join(TELEMETRY_DIR, "telemetry_scout_accept.csv")

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


def run_speculative_scout_benchmark(
    target_model_id="meta-llama/Llama-3.1-8B-Instruct",
    scout_model_id="meta-llama/Llama-3.2-1B-Instruct",
    prompt="Explain how general relativity describes the curvature of spacetime around massive objects.",
    K=5,
    max_target_tokens=80
):
    print("=" * 95)
    print("⚡ SDSIE SPECULATIVE SCOUT ENGINE (STEP 3 BENCHMARK)")
    print(f"🎯 Target Model (Verifier) : {target_model_id}")
    print(f"🚀 Scout Model (Draft)    : {scout_model_id}")
    print(f"🔬 Speculative Lookahead   : K = {K} candidate tokens per cycle")
    print("=" * 95)

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    print("\n[1/3] Loading Shared Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/3] Loading Target Model (8B) into VRAM...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    target_model.eval()

    print("[3/3] Loading Scout Model (1B) into VRAM...")
    scout_model = AutoModelForCausalLM.from_pretrained(
        scout_model_id,
        dtype=torch.bfloat16,
        device_map="cuda"
    )
    scout_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    current_ids = inputs["input_ids"]

    total_drafted = 0
    total_accepted = 0
    cycle_count = 0
    tokens_generated = 0

    torch.cuda.synchronize()
    t_benchmark_start = time.perf_counter()

    with torch.inference_mode():
        while tokens_generated < max_target_tokens:
            cycle_count += 1
            draft_ids = current_ids.clone()
            draft_tokens = []
            
            for _ in range(K):
                scout_out = scout_model(draft_ids)
                next_draft_tok = torch.argmax(scout_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_draft_tok)
                draft_ids = torch.cat([draft_ids, next_draft_tok], dim=-1)

            total_drafted += K
            target_out = target_model(draft_ids)
            target_logits = target_out.logits
            prefix_len = current_ids.shape[1]

            accepted_in_cycle = 0
            new_accepted_ids = []

            for i in range(K):
                expected_token = torch.argmax(target_logits[:, prefix_len - 1 + i, :], dim=-1, keepdim=True)
                drafted_token = draft_tokens[i]

                if expected_token.item() == drafted_token.item():
                    accepted_in_cycle += 1
                    new_accepted_ids.append(drafted_token)
                else:
                    new_accepted_ids.append(expected_token)
                    break
            else:
                bonus_token = torch.argmax(target_logits[:, prefix_len - 1 + K, :], dim=-1, keepdim=True)
                new_accepted_ids.append(bonus_token)

            total_accepted += accepted_in_cycle
            accepted_tensor = torch.cat(new_accepted_ids, dim=-1)
            current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
            tokens_generated += accepted_tensor.shape[1]

            accept_rate = (accepted_in_cycle / K) * 100.0
            print(f"Cycle {cycle_count:<2} | Drafted: {K} | Accepted: {accepted_in_cycle}/{K} ({accept_rate:.1f}%)")

            if tokenizer.eos_token_id in accepted_tensor:
                break

    torch.cuda.synchronize()
    t_benchmark_end = time.perf_counter()

    monitor.stop()
    monitor.close()

    total_latency_sec = t_benchmark_end - t_benchmark_start
    total_pwr_w = monitor.get_average_power(t_benchmark_start, t_benchmark_end)
    total_energy_j = total_pwr_w * total_latency_sec
    tok_per_sec = tokens_generated / total_latency_sec if total_latency_sec > 0 else 0
    joules_per_tok = total_energy_j / tokens_generated if tokens_generated > 0 else 0
    global_accept_rate = (total_accepted / total_drafted) * 100.0 if total_drafted > 0 else 0

    print("=" * 95)
    print("📊 SPECULATIVE SCOUT TELEMETRY SUMMARY:")
    print(f" • Tokens Generated           : {tokens_generated} tokens")
    print(f" • Real Speculative Throughput: {tok_per_sec:.2f} tokens/sec")
    print(f" • Energy Efficiency          : {joules_per_tok:.4f} Joules/token")
    print(f" • Scout Draft Acceptance Rate: {global_accept_rate:.1f}%")
    print("=" * 95)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": "Speculative Scout (1B->8B)",
        "tokens": tokens_generated,
        "throughput_tok_sec": round(tok_per_sec, 2),
        "avg_power_watts": round(total_pwr_w, 2),
        "total_energy_joules": round(total_energy_j, 4),
        "joules_per_token": round(joules_per_tok, 6),
        "high_gear_pct": round(global_accept_rate, 1),
        "low_gear_pct": round(100.0 - global_accept_rate, 1)
    }

    file_exists = os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    spec_json = os.path.join(TELEMETRY_DIR, "speculative_scout_report.json")
    with open(spec_json, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"💾 Benchmark saved to: {MASTER_CSV}")

if __name__ == "__main__":
    run_speculative_scout_benchmark()