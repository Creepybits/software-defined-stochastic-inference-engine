"""
step4_entropy_gated_scout.py

FIRST REAL INTEGRATION of the validated Schmitt-trigger entropy clutch
(vllm_sdsie/kernels/entropy_clutch.py, via SDSIESpeculativeController) into a
real scout(1B)->target(8B) speculative decoding loop.

This closes the specific gap documented in SDSIE_project_status.md and
sdsie_paper.tex Section 4.5 ("Remaining Integration Gap"): every prior script
either (a) computed a clutch decision and never acted on it (sdsie_server.py,
sweep_real_model.py, harness_telemetry.py, cognitive_benchmark.py), or (b) ran
real speculation with a FIXED draft window (step3_speculative_scout.py, K=5
always). This script does neither -- it asks the clutch for k EVERY cycle and
genuinely branches:
  - k > 0 : real scout-draft + target-verify cycle (same accept/reject logic
            as step3_speculative_scout.py, just with k chosen by the clutch
            instead of hardcoded).
  - k = 0 : single-step fallback -- the scout model is not called at all,
            target does one plain forward pass for one token.

HONEST CAVEAT (read before comparing throughput/energy against step3):
To get a fresh, correct entropy reading before each decision, this script
performs one extra target-model forward pass per cycle (a "resync" pass after
each step) that step3_speculative_scout.py does not need, since step3 reuses
verification logits directly. That means this script pays a small real
compute cost step3 doesn't -- so raw tok/s and J/tok here are NOT directly
apples-to-apples with step3's numbers. This is a real, deliberate simplicity
tradeoff for correctness of the entropy signal, not an oversight. It's a
natural target for later optimization (reusing verification-step logits
instead of a fresh resync pass), consistent with real serving systems.

step3_speculative_scout.py is left completely untouched as the clean,
already-validated reference implementation.
"""

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
import numpy as np
from datetime import datetime

from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "tools", "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)
MASTER_CSV = os.path.join(TELEMETRY_DIR, "telemetry_entropy_gated_scout.csv")


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


def run_entropy_gated_scout_benchmark(
    target_model_id="meta-llama/Llama-3.1-8B-Instruct",
    scout_model_id="meta-llama/Llama-3.2-1B-Instruct",
    prompt="Explain how general relativity describes the curvature of spacetime around massive objects.",
    prompt_label="Custom",
    default_K=5,
    theta_low=0.55,
    theta_high=1.25,
    alpha=0.35,
    max_target_tokens=80,
    warmup_steps=5,
    target_model=None,
    scout_model=None,
    tokenizer=None,
    monitor=None,
    _owns_models=False
):
    print("=" * 95)
    print("SDSIE ENTROPY-GATED SPECULATIVE SCOUT (STEP 4 -- REAL CLUTCH INTEGRATION)")
    print(f"Prompt ({prompt_label}): {prompt[:70]}...")
    print(f"Target Model (Verifier) : {target_model_id}")
    print(f"Scout Model (Draft)     : {scout_model_id}")
    print(f"Clutch thresholds       : theta_low={theta_low}, theta_high={theta_high}, alpha={alpha}")
    print(f"Default draft window K  : {default_K} (used only when clutch is Active)")
    print("=" * 95)

    _owns_monitor = monitor is None
    if monitor is None:
        monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
        monitor.start()

    if tokenizer is None:
        print("\n[1/3] Loading Shared Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    if target_model is None:
        print("[2/3] Loading Target Model (8B) into VRAM...")
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        target_model.eval()

    if scout_model is None:
        print("[3/3] Loading Scout Model (1B) into VRAM...")
        scout_model = AutoModelForCausalLM.from_pretrained(
            scout_model_id, dtype=torch.bfloat16, device_map="cuda"
        )
        scout_model.eval()

    controller = SDSIESpeculativeController(
        default_k=default_K, theta_low=theta_low, theta_high=theta_high, alpha=alpha
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    current_ids = inputs["input_ids"]

    # --- Warmup (not timed, not recorded) ---
    # Avoids the cold-start latency confound documented in sweep_real_model.py.
    print(f"\n[*] Running {warmup_steps} warmup steps (untimed)...")
    with torch.inference_mode():
        warm_ids = current_ids.clone()
        for _ in range(warmup_steps):
            t_out = target_model(warm_ids)
            s_out = scout_model(warm_ids)
            next_tok = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            warm_ids = torch.cat([warm_ids, next_tok], dim=-1)
    torch.cuda.synchronize()
    print("[*] Warmup complete.\n")

    total_drafted = 0
    total_accepted = 0
    speculative_cycles = 0
    fallback_steps = 0
    cycle_count = 0
    tokens_generated = 0
    k_history = []

    torch.cuda.synchronize()
    t_benchmark_start = time.perf_counter()

    with torch.inference_mode():
        # Prime the clutch with real logits from the actual prompt before the loop.
        primer_out = target_model(current_ids)
        current_logits = primer_out.logits[:, -1, :]

        while tokens_generated < max_target_tokens:
            cycle_count += 1
            k = controller.plan_speculation_step(current_logits)
            k_history.append(k)
            entropy_now = controller.clutch.running_entropy

            if k == 0:
                # --- SINGLE-STEP FALLBACK: scout is not called at all ---
                fallback_steps += 1
                next_token = torch.argmax(current_logits, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=-1)
                tokens_generated += 1
                controller.record_verification(num_drafted=0, num_accepted=0)

                print(f"Cycle {cycle_count:<3} | FALLBACK   (k=0) | entropy={entropy_now:.3f} | +1 token")

                if next_token.item() == tokenizer.eos_token_id:
                    break

                resync_out = target_model(current_ids)
                current_logits = resync_out.logits[:, -1, :]
                continue

            # --- SPECULATIVE PATH: real scout-draft + target-verify, k chosen by clutch ---
            speculative_cycles += 1
            draft_ids = current_ids.clone()
            draft_tokens = []
            for _ in range(k):
                scout_out = scout_model(draft_ids)
                next_draft_tok = torch.argmax(scout_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_draft_tok)
                draft_ids = torch.cat([draft_ids, next_draft_tok], dim=-1)

            total_drafted += k
            target_out = target_model(draft_ids)
            target_logits = target_out.logits
            prefix_len = current_ids.shape[1]

            accepted_in_cycle = 0
            new_accepted_ids = []
            for i in range(k):
                expected_token = torch.argmax(target_logits[:, prefix_len - 1 + i, :], dim=-1, keepdim=True)
                drafted_token = draft_tokens[i]
                if expected_token.item() == drafted_token.item():
                    accepted_in_cycle += 1
                    new_accepted_ids.append(drafted_token)
                else:
                    new_accepted_ids.append(expected_token)
                    break
            else:
                bonus_token = torch.argmax(target_logits[:, prefix_len - 1 + k, :], dim=-1, keepdim=True)
                new_accepted_ids.append(bonus_token)

            total_accepted += accepted_in_cycle
            accepted_tensor = torch.cat(new_accepted_ids, dim=-1)
            current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
            tokens_generated += accepted_tensor.shape[1]
            controller.record_verification(num_drafted=k, num_accepted=accepted_in_cycle)

            accept_rate = (accepted_in_cycle / k) * 100.0
            print(f"Cycle {cycle_count:<3} | SPECULATIVE (k={k}) | Accepted: {accepted_in_cycle}/{k} "
                  f"({accept_rate:.1f}%) | entropy={entropy_now:.3f}")

            if tokenizer.eos_token_id in accepted_tensor:
                break

            # Resync pass: fresh logits for the NEXT cycle's clutch decision.
            # (See module docstring: this is the extra pass step3 doesn't pay.)
            resync_out = target_model(current_ids)
            current_logits = resync_out.logits[:, -1, :]

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
    global_accept_rate = (total_accepted / total_drafted) * 100.0 if total_drafted > 0 else 0
    fallback_pct = (fallback_steps / cycle_count) * 100.0 if cycle_count > 0 else 0

    print("=" * 95)
    print("ENTROPY-GATED SPECULATIVE SCOUT -- TELEMETRY SUMMARY:")
    print(f" - Tokens Generated            : {tokens_generated}")
    print(f" - Throughput                  : {tok_per_sec:.2f} tok/s")
    print(f" - Energy Efficiency           : {joules_per_tok:.4f} J/token")
    print(f" - Speculative cycles          : {speculative_cycles}  |  Fallback (k=0) steps: {fallback_steps} "
          f"({fallback_pct:.1f}% of cycles)")
    print(f" - Draft Acceptance Rate       : {global_accept_rate:.1f}% (within speculative cycles only)")
    print(" - NOTE: not directly comparable to step3_speculative_scout.py -- see module docstring "
          "(extra resync forward pass per cycle).")
    print("=" * 95)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "category": "Entropy-Gated Speculative Scout (1B<->8B, real branching)",
        "prompt_label": prompt_label,
        "theta_low": theta_low,
        "theta_high": theta_high,
        "tokens": tokens_generated,
        "throughput_tok_sec": round(tok_per_sec, 2),
        "avg_power_watts": round(total_pwr_w, 2),
        "total_energy_joules": round(total_energy_j, 4),
        "joules_per_token": round(joules_per_tok, 6),
        "speculative_cycles": speculative_cycles,
        "fallback_steps": fallback_steps,
        "fallback_pct": round(fallback_pct, 1),
        "draft_accept_rate_pct": round(global_accept_rate, 1),
        "k_history": k_history
    }

    csv_fields = [k for k in log_entry.keys() if k != "k_history"]
    file_exists = os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: v for k, v in log_entry.items() if k != "k_history"})

    print(f"Benchmark saved to: {MASTER_CSV}")

    return log_entry, target_model, scout_model, tokenizer, monitor


def run_n_trial_suite(
    num_trials=5,
    default_K=5,
    theta_low=0.55,
    theta_high=1.25,
    alpha=0.35,
    max_target_tokens=250,
    warmup_steps=5
):
    """
    Runs the SAME 3 prompts used in benchmark_academic_validation_v4.py,
    N times each, and reports mean +/- std -- matching v4's own methodology
    so results are directly comparable to the existing fixed-K=5 and FP16
    baseline numbers already in the paper (Table 2).

    Note: the clutch's EMA-smoothed entropy state resets fresh for each trial
    (a new SDSIESpeculativeController is created inside
    run_entropy_gated_scout_benchmark on every call), so trials are
    independent repeats of the same deterministic greedy-decoding run --
    any variation across trials reflects timing/power noise, not decoding
    variation, since decoding itself is deterministic given the same prompt.
    """
    PROMPTS = [
        ("Poem", "Write an original Chant Royal poem in English celebrating mathematics."),
        ("Physics", "Explain the physics of semiconductor memory bandwidth and the memory wall."),
        ("Code", "Write a Python implementation of a binary search tree with type annotations."),
    ]

    target_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    scout_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading Target Model (8B) into VRAM (shared across all trials/prompts)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    target_model.eval()

    print("[*] Loading Scout Model (1B) into VRAM (shared across all trials/prompts)...")
    scout_model = AutoModelForCausalLM.from_pretrained(
        scout_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    scout_model.eval()

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    raw_trials = {label: [] for label, _ in PROMPTS}

    for label, prompt_text in PROMPTS:
        print(f"\n{'#' * 95}\n# PROMPT: {label}  (N={num_trials} trials)\n{'#' * 95}")
        for trial_idx in range(1, num_trials + 1):
            print(f"\n--- {label} | Trial {trial_idx}/{num_trials} ---")
            entry, target_model, scout_model, tokenizer, monitor = run_entropy_gated_scout_benchmark(
                target_model_id=target_model_id,
                scout_model_id=scout_model_id,
                prompt=prompt_text,
                prompt_label=f"{label}_trial{trial_idx}",
                default_K=default_K,
                theta_low=theta_low,
                theta_high=theta_high,
                alpha=alpha,
                max_target_tokens=max_target_tokens,
                warmup_steps=warmup_steps,
                target_model=target_model,
                scout_model=scout_model,
                tokenizer=tokenizer,
                monitor=monitor,
            )
            raw_trials[label].append(entry)

    monitor.stop()
    monitor.close()

    # --- Aggregate mean +/- std per prompt, same style as benchmark_academic_validation_v4.py ---
    aggregated = {}
    for label, trials in raw_trials.items():
        tps = [t["throughput_tok_sec"] for t in trials]
        jtok = [t["joules_per_token"] for t in trials]
        pwr = [t["avg_power_watts"] for t in trials]
        fallback_pcts = [t["fallback_pct"] for t in trials]
        accept_rates = [t["draft_accept_rate_pct"] for t in trials]
        spec_cycles = [t["speculative_cycles"] for t in trials]

        aggregated[label] = {
            "n_trials": num_trials,
            "tps_mean": float(np.mean(tps)),
            "tps_std": float(np.std(tps)),
            "j_tok_mean": float(np.mean(jtok)),
            "j_tok_std": float(np.std(jtok)),
            "power_mean": float(np.mean(pwr)),
            "fallback_pct_mean": float(np.mean(fallback_pcts)),
            "fallback_pct_std": float(np.std(fallback_pcts)),
            "accept_rate_pct_mean": float(np.mean(accept_rates)),
            "accept_rate_pct_std": float(np.std(accept_rates)),
            "speculative_cycles_mean": float(np.mean(spec_cycles)),
            "raw_trials": trials,  # full per-trial detail, including each trial's k_history
        }

    summary_json = os.path.join(TELEMETRY_DIR, f"entropy_gated_scout_n{num_trials}_summary.json")
    with open(summary_json, "w") as f:
        json.dump(aggregated, f, indent=2)

    print("\n" + "=" * 95)
    print(f"N={num_trials} TRIAL SUITE COMPLETE -- MEAN +/- STD")
    print(f"{'Prompt':<10} | {'tok/s':<16} | {'J/tok':<16} | {'Fallback %':<16} | {'Accept %':<16}")
    print("-" * 95)
    for label, a in aggregated.items():
        tps_str = f"{a['tps_mean']:.2f} +/- {a['tps_std']:.2f}"
        j_str = f"{a['j_tok_mean']:.3f} +/- {a['j_tok_std']:.3f}"
        fb_str = f"{a['fallback_pct_mean']:.1f} +/- {a['fallback_pct_std']:.1f}"
        acc_str = f"{a['accept_rate_pct_mean']:.1f} +/- {a['accept_rate_pct_std']:.1f}"
        print(f"{label:<10} | {tps_str:<16} | {j_str:<16} | {fb_str:<16} | {acc_str:<16}")
    print("=" * 95)
    print(f"Summary saved to: {summary_json}")
    print("(Each prompt's raw_trials list, including full per-trial k_history, is preserved in the JSON.)")


if __name__ == "__main__":
    run_n_trial_suite(num_trials=5)
