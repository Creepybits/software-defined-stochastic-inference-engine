"""
step4_fp16_baseline_matched.py

Purpose-built FP16-only baseline, structurally IDENTICAL to
step4_entropy_gated_scout.py (same NVMLPowerMonitor, same warmup procedure,
same 3 prompts, same N-trial aggregation, same telemetry schema where
applicable) EXCEPT it contains no clutch, no scout model, and no speculative
branching at all -- just a plain target-only forward-pass-per-token loop.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import threading
import pynvml
import os
import json
import csv
import numpy as np
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "tools", "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
MASTER_CSV = os.path.join(TELEMETRY_DIR, "telemetry_fp16_baseline_matched.csv")


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


def run_fp16_baseline(
    target_model_id="meta-llama/Llama-3.1-8B-Instruct",
    prompt="Explain how general relativity describes the curvature of spacetime around massive objects.",
    prompt_label="Custom",
    max_target_tokens=80,
    warmup_steps=5,
    target_model=None,
    tokenizer=None,
    monitor=None,
):
    _owns_monitor = monitor is None
    if monitor is None:
        monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
        monitor.start()

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if target_model is None:
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        target_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    current_ids = inputs["input_ids"]

    # --- Warmup (not timed, not recorded) -- identical procedure to step4 ---
    with torch.inference_mode():
        warm_ids = current_ids.clone()
        for _ in range(warmup_steps):
            t_out = target_model(warm_ids)
            next_tok = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            warm_ids = torch.cat([warm_ids, next_tok], dim=-1)
    torch.cuda.synchronize()

    tokens_generated = 0

    torch.cuda.synchronize()
    t_benchmark_start = time.perf_counter()

    with torch.inference_mode():
        while tokens_generated < max_target_tokens:
            out = target_model(current_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            tokens_generated += 1

            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    t_benchmark_end = time.perf_counter()

    if _owns_monitor:
        monitor.stop()
        monitor.close()

    total_latency_sec = t_benchmark_end - t_benchmark_start
    total_pwr_w = monitor.get_average_power(t_benchmark_start, t_benchmark_end)
    total_energy_j = total_pwr_w * total_latency_sec
    tok_per_sec = tokens_generated / total_latency_sec if total_latency_sec > 0 else 0
    joules_per_tok = total_energy_j / tokens_generated if tokens_generated > 0 else 0

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": "FP16 Baseline (matched harness, no clutch/scout)",
        "prompt_label": prompt_label,
        "tokens": tokens_generated,
        "throughput_tok_sec": round(tok_per_sec, 2),
        "avg_power_watts": round(total_pwr_w, 2),
        "total_energy_joules": round(total_energy_j, 4),
        "joules_per_token": round(joules_per_tok, 6),
    }

    file_exists = os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    return log_entry, target_model, tokenizer, monitor


def run_n_trial_suite(num_trials=5, max_target_tokens=250, warmup_steps=5):
    PROMPTS = [
        ("Poem", "Write an original Chant Royal poem in English celebrating mathematics."),
        ("Physics", "Explain the physics of semiconductor memory bandwidth and the memory wall."),
        ("Code", "Write a Python implementation of a binary search tree with type annotations."),
    ]

    target_model_id = "meta-llama/Llama-3.1-8B-Instruct"

    print("=" * 95)
    print("SDSIE MATCHED FP16 BASELINE (no clutch, no scout -- structural twin of step4)")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading Target Model (8B) into VRAM (shared across all trials/prompts)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    target_model.eval()

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    raw_trials = {label: [] for label, _ in PROMPTS}

    for label, prompt_text in PROMPTS:
        print(f"\n{'#' * 95}\n# PROMPT: {label}  (N={num_trials} trials)\n{'#' * 95}")
        for trial_idx in range(1, num_trials + 1):
            print(f"--- {label} | Trial {trial_idx}/{num_trials} ---")
            entry, target_model, tokenizer, monitor = run_fp16_baseline(
                target_model_id=target_model_id,
                prompt=prompt_text,
                prompt_label=f"{label}_trial{trial_idx}",
                max_target_tokens=max_target_tokens,
                warmup_steps=warmup_steps,
                target_model=target_model,
                tokenizer=tokenizer,
                monitor=monitor,
            )
            print(f"    tok/s={entry['throughput_tok_sec']}  J/tok={entry['joules_per_token']:.3f}")
            raw_trials[label].append(entry)

    monitor.stop()
    monitor.close()

    aggregated = {}
    for label, trials in raw_trials.items():
        tps = [t["throughput_tok_sec"] for t in trials]
        jtok = [t["joules_per_token"] for t in trials]
        pwr = [t["avg_power_watts"] for t in trials]
        aggregated[label] = {
            "n_trials": num_trials,
            "tps_mean": float(np.mean(tps)),
            "tps_std": float(np.std(tps)),
            "j_tok_mean": float(np.mean(jtok)),
            "j_tok_std": float(np.std(jtok)),
            "power_mean": float(np.mean(pwr)),
            "raw_trials": trials,
        }

    summary_json = os.path.join(TELEMETRY_DIR, f"fp16_baseline_matched_n{num_trials}_summary.json")
    with open(summary_json, "w") as f:
        json.dump(aggregated, f, indent=2)

    print("\n" + "=" * 95)
    print(f"N={num_trials} MATCHED FP16 BASELINE COMPLETE -- MEAN +/- STD")
    print(f"{'Prompt':<10} | {'tok/s':<18} | {'J/tok':<18}")
    print("-" * 95)
    for label, a in aggregated.items():
        tps_str = f"{a['tps_mean']:.2f} +/- {a['tps_std']:.2f}"
        j_str = f"{a['j_tok_mean']:.3f} +/- {a['j_tok_std']:.3f}"
        print(f"{label:<10} | {tps_str:<18} | {j_str:<18}")
    print("=" * 95)
    print(f"Summary saved to: {summary_json}")


if __name__ == "__main__":
    run_n_trial_suite(num_trials=5)
