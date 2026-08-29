"""
sweep_real_model.py - Empirical Llama-3.1-8B Parameter Sweep
Hardware: NVIDIA GeForce RTX 5090 Blackwell
Task: Chant Royal Poetic Structure (Max 512 tokens)

WARNING (added 2026-08-29, not yet fixed): this script computes k via
SDSIESpeculativeController.plan_speculation_step() and logs it as
'speculation_percentage', but the generation loop below never branches on
k - every step does one plain forward pass + argmax regardless of the
clutch's decision. Numbers here (joules_per_token, throughput_tok_s) reflect
plain FP16 generation with KV-cache, NOT accelerated speculative decoding,
even though the output columns are named as if they were. Same
logged-but-not-acted-on pattern as sdsie_server.py and others - see
SDSIE_project_status.md, 'compute a signal, log it, never act on it'.
Useful for checking clutch/gear behavior (theta_low/theta_high/alpha
sweeps) but NOT as evidence of real speculative decoding performance.
"""

import time
import json
import torch
import pynvml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController

# Initialize NVML
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_name = pynvml.nvmlDeviceGetName(gpu)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda:0"
MAX_TOKENS = 1024
OUTPUT_DIR = Path(__file__).resolve().parent / "tools" / "telemetry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact prompt used in yesterday's baseline
PROMPT = (
    "Write an original Chant Royal poem in English. "
    "Do not include any introductory explanation, definitions, structural outlines, or closing remarks. "
    "Output only the raw poem verses from the very first word to the final line."
)

print("=" * 70)
print(f"[*] SDSIE EMPIRICAL PARAMETER SWEEP ON {gpu_name}")
print(f"[*] Model ID    : {MODEL_ID}")
print(f"[*] Max Tokens  : {MAX_TOKENS}")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto")
model.eval()

# Format chat template
chat_messages = [{"role": "user", "content": PROMPT}]
formatted_prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
prompt_inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

# --- Warmup pass (not timed, not recorded) ---
# First CUDA kernel calls pay a one-time compilation/allocation cost that
# later calls skip. Without this, the first config in the sweep below is
# artificially penalized (confirmed: run 1 showed ~37 tok/s vs ~50 tok/s
# for every other config, in two independent sweeps). Run a few throwaway
# forward passes here so all 9 timed configs start from the same warm state.
print("[*] Running warmup pass (untimed)...")
with torch.no_grad():
    warmup_ids = prompt_inputs.input_ids
    warmup_past = None
    for _ in range(10):
        warmup_out = model(input_ids=warmup_ids, past_key_values=warmup_past, use_cache=True)
        warmup_past = warmup_out.past_key_values
        warmup_ids = torch.argmax(warmup_out.logits[:, -1, :], dim=-1, keepdim=True)
torch.cuda.synchronize()
print("[*] Warmup complete.\n")

# 9 Key Parameter Permutations (theta_low, theta_high, alpha)
configs = [
    (0.35, 1.00, 0.35),
    (0.35, 1.25, 0.35),
    (0.35, 1.50, 0.35),
    (0.55, 1.00, 0.35),
    (0.55, 1.25, 0.35),  # Yesterday's Baseline
    (0.55, 1.50, 0.35),
    (0.75, 1.25, 0.35),
    (0.75, 1.50, 0.35),
    (0.75, 1.75, 0.35),
]

print(f"\n[*] Launching 9-configuration sweep (~3 minutes)...\n")
results = []

for idx, (t_low, t_high, alpha) in enumerate(configs, 1):
    controller = SDSIESpeculativeController(default_k=5, theta_low=t_low, theta_high=t_high)
    controller.clutch.alpha = alpha
    
    current_input_ids = prompt_inputs.input_ids
    past_key_values = None
    power_samples = []
    k_history = []
    generated_tokens = 0
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for step in range(MAX_TOKENS):
            # 100 Hz Instantaneous NVML Power (Watts)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(gpu)
            power_samples.append(power_mw / 1000.0)
            
            outputs = model(
                input_ids=current_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            
            # Stochastic Clutch Decision
            k = controller.plan_speculation_step(next_logits)
            k_history.append(k)
            
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_tokens += 1
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            current_input_ids = next_token

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    avg_power = sum(power_samples) / len(power_samples) if power_samples else 0.0
    joules_per_token = (avg_power * elapsed) / generated_tokens if generated_tokens > 0 else 0.0
    throughput_tps = generated_tokens / elapsed if elapsed > 0 else 0.0
    spec_ratio = (sum(1 for k in k_history if k > 0) / len(k_history)) * 100 if k_history else 0.0
    
    is_baseline = " (BASELINE)" if (t_low == 0.55 and t_high == 1.25) else ""
    print(f"[{idx:02d}/09] θ_low={t_low:.2f} | θ_high={t_high:.2f}{is_baseline:<11} -> {joules_per_token:.3f} J/tok | {throughput_tps:.2f} tok/s | Spec: {spec_ratio:.1f}% ({generated_tokens} tok)")
    
    results.append({
        "theta_low": t_low,
        "theta_high": t_high,
        "alpha": alpha,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "joules_per_token": round(joules_per_token, 3),
        "throughput_tok_s": round(throughput_tps, 2),
        "speculation_percentage": round(spec_ratio, 1),
        "avg_power_watts": round(avg_power, 2)
    })

out_json = OUTPUT_DIR / "real_model_sweep_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print(f"[*] Sweep complete! Full results saved to: {out_json}")
print("=" * 70)