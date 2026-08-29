"""
benchmark_academic_validation_v4.py
Robust Speculative Benchmark using explicit Scout-Target Verification.
Hardware: NVIDIA RTX 5090 Blackwell (100 Hz NVML Polling)
"""

import time
import json
import threading
from pathlib import Path
import numpy as np
import pynvml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SCOUT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda:0"
NUM_TRIALS = 10
MAX_TOKENS = 250
K_DRAFT = 5

PROMPTS = [
    "Write an original Chant Royal poem in English celebrating mathematics.",
    "Explain the physics of semiconductor memory bandwidth and the memory wall.",
    "Write a Python implementation of a binary search tree with type annotations."
]

TELEMETRY_DIR = Path(__file__).resolve().parent / "tools" / "telemetry"
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)


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

    def average_power(self, start_time, end_time):
        with self.lock:
            window = [p for t, p in self.power_samples if start_time <= t <= end_time]
        if not window:
            try:
                return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except Exception:
                return 0.0
        return sum(window) / len(window)

    def close(self):
        self.stop()
        pynvml.nvmlShutdown()


def generate_baseline(target_model, tokenizer, prompt, max_tokens, monitor):
    inputs = tokenizer(
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
        return_tensors="pt"
    ).to(DEVICE)
    
    current_ids = inputs.input_ids
    generated = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for _ in range(max_tokens):
            out = target_model(current_ids)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            current_ids = torch.cat([current_ids, next_token], dim=-1)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_power = monitor.average_power(t0, t1)
    num_tokens = len(generated)
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    j_tok = (avg_power * elapsed) / num_tokens if num_tokens > 0 else 0.0

    return {
        "tokens": generated,
        "elapsed": elapsed,
        "tps": tps,
        "avg_power_w": avg_power,
        "j_tok": j_tok,
        "total_drafted": 0,
        "total_accepted": 0
    }


def generate_speculative_exact(target_model, scout_model, tokenizer, prompt, max_tokens, monitor, K=5):
    inputs = tokenizer(
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True),
        return_tensors="pt"
    ).to(DEVICE)

    current_ids = inputs.input_ids
    generated = []
    total_drafted = 0
    total_accepted = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        while len(generated) < max_tokens:
            draft_ids = current_ids.clone()
            draft_tokens = []

            # 1. Scout proposes K candidate tokens
            for _ in range(K):
                scout_out = scout_model(draft_ids)
                next_draft_tok = torch.argmax(scout_out.logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_draft_tok)
                draft_ids = torch.cat([draft_ids, next_draft_tok], dim=-1)

            total_drafted += K

            # 2. Target verifies all K drafted tokens in one pass
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
            new_ids_list = accepted_tensor.squeeze(0).tolist()
            if isinstance(new_ids_list, int):
                new_ids_list = [new_ids_list]
            generated.extend(new_ids_list)

            if tokenizer.eos_token_id in new_ids_list:
                break

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    generated = generated[:max_tokens]
    elapsed = t1 - t0
    avg_power = monitor.average_power(t0, t1)
    num_tokens = len(generated)
    tps = num_tokens / elapsed if elapsed > 0 else 0.0
    j_tok = (avg_power * elapsed) / num_tokens if num_tokens > 0 else 0.0

    return {
        "tokens": generated,
        "elapsed": elapsed,
        "tps": tps,
        "avg_power_w": avg_power,
        "j_tok": j_tok,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted
    }


def main():
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu)
    pynvml.nvmlShutdown()

    print("=" * 85)
    print(f"[*] SDSIE ACADEMIC BENCHMARK v4 ({gpu_name})")
    print(f"[*] Target: {TARGET_MODEL_ID} | Scout: {SCOUT_MODEL_ID} | Trials: N={NUM_TRIALS}")
    print("=" * 85)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[*] Loading Target Model (8B)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    target_model.eval()

    print("[*] Loading Scout Model (1B)...")
    scout_model = AutoModelForCausalLM.from_pretrained(
        SCOUT_MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    scout_model.eval()

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    # --- TEST 1: Exact Lossless Verification (100% Token Match) ---
    print("\n[1/2] Verifying Exact Lossless Fidelity (Baseline vs. Speculative)...")
    fidelity_by_prompt = {}
    for p_idx, prompt in enumerate(PROMPTS, 1):
        base_res = generate_baseline(target_model, tokenizer, prompt, MAX_TOKENS, monitor)
        spec_res = generate_speculative_exact(target_model, scout_model, tokenizer, prompt, MAX_TOKENS, monitor, K=K_DRAFT)

        match_count = sum(1 for a, b in zip(base_res["tokens"], spec_res["tokens"]) if a == b)
        total_tok = max(len(base_res["tokens"]), len(spec_res["tokens"]))
        match_pct = (match_count / total_tok) * 100.0 if total_tok > 0 else 0.0
        fidelity_by_prompt[f"prompt_{p_idx}"] = match_pct
        print(f"  Prompt {p_idx}: {match_pct:.1f}% exact token match ({len(base_res['tokens'])} tokens)")

    agg_fidelity = float(np.mean(list(fidelity_by_prompt.values())))
    print(f"[*] Aggregate Fidelity Match: {agg_fidelity:.2f}%")

    # --- TEST 2: Multi-Trial Benchmark (N=3 each) ---
    print(f"\n[2/2] Running Multi-Trial Performance Benchmark...")
    ablation = {}

    for p_idx, prompt in enumerate(PROMPTS, 1):
        p_key = f"prompt_{p_idx}"
        ablation[p_key] = {}

        # 1. Baseline Target
        tps_b, j_b, pwr_b = [], [], []
        print(f"  [{p_key}] FP16 Baseline ...", end="", flush=True)
        for _ in range(NUM_TRIALS):
            r = generate_baseline(target_model, tokenizer, prompt, MAX_TOKENS, monitor)
            tps_b.append(r["tps"])
            j_b.append(r["j_tok"])
            pwr_b.append(r["avg_power_w"])
        ablation[p_key]["FP16 Baseline"] = {
            "tps_mean": float(np.mean(tps_b)),
            "tps_std": float(np.std(tps_b)),
            "j_tok_mean": float(np.mean(j_b)),
            "j_tok_std": float(np.std(j_b)),
            "power_mean": float(np.mean(pwr_b))
        }
        print(" done.")

        # 2. Speculative Scout
        tps_s, j_s, pwr_s, acc_s = [], [], [], []
        print(f"  [{p_key}] Speculative (1B->8B) ...", end="", flush=True)
        for _ in range(NUM_TRIALS):
            r = generate_speculative_exact(target_model, scout_model, tokenizer, prompt, MAX_TOKENS, monitor, K=K_DRAFT)
            tps_s.append(r["tps"])
            j_s.append(r["j_tok"])
            pwr_s.append(r["avg_power_w"])
            if r["total_drafted"] > 0:
                acc_s.append((r["total_accepted"] / r["total_drafted"]) * 100.0)
        ablation[p_key]["Speculative (1B->8B)"] = {
            "tps_mean": float(np.mean(tps_s)),
            "tps_std": float(np.std(tps_s)),
            "j_tok_mean": float(np.mean(j_s)),
            "j_tok_std": float(np.std(j_s)),
            "power_mean": float(np.mean(pwr_s)),
            "accept_rate_pct_mean": float(np.mean(acc_s)) if acc_s else None
        }
        print(" done.")

    monitor.close()

    print("\n" + "=" * 85)
    for p_key, m_dict in ablation.items():
        print(f"\n-- {p_key} --")
        print(f"{'Mode':<26} | {'Throughput (tok/s)':<20} | {'Energy (J/token)':<18} | {'Accept %':<10}")
        print("-" * 80)
        for m, st in m_dict.items():
            tps_str = f"{st['tps_mean']:.2f} ± {st['tps_std']:.2f}"
            j_str = f"{st['j_tok_mean']:.3f} ± {st['j_tok_std']:.3f}"
            acc_str = f"{st['accept_rate_pct_mean']:.1f}%" if "accept_rate_pct_mean" in st and st["accept_rate_pct_mean"] is not None else "n/a"
            print(f"{m:<26} | {tps_str:<20} | {j_str:<18} | {acc_str:<10}")
    print("=" * 85)

    out_file = TELEMETRY_DIR / "academic_validation_results_v4.json"
    with open(out_file, "w") as f:
        json.dump({"fidelity_by_prompt": fidelity_by_prompt, "ablation": ablation}, f, indent=2)
    print(f"\n[*] Results saved to: {out_file}")


if __name__ == "__main__":
    main()