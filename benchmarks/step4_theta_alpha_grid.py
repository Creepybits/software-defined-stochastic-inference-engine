"""
step4_theta_alpha_grid.py

Grid: 4 theta combos x 3 alpha values x 3 prompts x N=3 trials = 108 trials.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import json
import numpy as np
from pathlib import Path

from vllm_sdsie.spec_decode.sdsie_speculator import SDSIESpeculativeController

REPO_ROOT = Path(__file__).resolve().parent.parent
TELEMETRY_DIR = REPO_ROOT / "telemetry"
os.makedirs(TELEMETRY_DIR, exist_ok=True)

# Reuse the existing step4 machinery by importing it directly, so behavior
# (warmup, resync logic, telemetry schema) stays identical to the already-
# validated script -- this file only adds the grid/sweep layer on top.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "step4_entropy_gated_scout", os.path.join(REPO_ROOT / "benchmarks", "step4_entropy_gated_scout.py")
)
step4_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4_mod)

NVMLPowerMonitor = step4_mod.NVMLPowerMonitor
run_entropy_gated_scout_benchmark = step4_mod.run_entropy_gated_scout_benchmark


THETA_COMBOS = [
    (0.35, 1.00),
    (0.55, 1.25),  # current production default
    (0.55, 1.50),
    (0.75, 1.50),
]
ALPHA_VALUES = [0.15, 0.35, 0.65]  # jumpy, current default, smooth

PROMPTS = [
    ("Poem", "Write an original Chant Royal poem in English celebrating mathematics."),
    ("Physics", "Explain the physics of semiconductor memory bandwidth and the memory wall."),
    ("Code", "Write a Python implementation of a binary search tree with type annotations."),
]

NUM_TRIALS = 3
MAX_TOKENS = 250
WARMUP_STEPS = 5


def load_baseline_reference():
    """Load the already-collected matched FP16 baseline (N=5) to compare against."""
    path = os.path.join(TELEMETRY_DIR, "fp16_baseline_matched_n5_summary.json")
    if not os.path.exists(path):
        print(f"[!] WARNING: baseline file not found at {path}. Speedup/energy-delta columns "
              f"will be omitted from the summary; re-run step4_fp16_baseline_matched.py first "
              f"if you want those.")
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    baseline_ref = load_baseline_reference()

    target_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    scout_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    total_combos = len(THETA_COMBOS) * len(ALPHA_VALUES)
    total_trials = total_combos * len(PROMPTS) * NUM_TRIALS
    print("=" * 95)
    print("SDSIE JOINT THETA x ALPHA GRID (real branching, entropy-gated speculative controller)")
    print(f"Theta combos: {THETA_COMBOS}")
    print(f"Alpha values: {ALPHA_VALUES}")
    print(f"Prompts: {[p[0] for p in PROMPTS]}  |  N={NUM_TRIALS} trials/combo/prompt")
    print(f"Total: {total_combos} combos x {len(PROMPTS)} prompts x {NUM_TRIALS} trials = {total_trials} trials")
    print("=" * 95)

    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[*] Loading Target Model (8B) into VRAM (shared across entire grid)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    target_model.eval()

    print("[*] Loading Scout Model (1B) into VRAM (shared across entire grid)...")
    scout_model = AutoModelForCausalLM.from_pretrained(
        scout_model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    scout_model.eval()

    monitor = NVMLPowerMonitor(device_index=0, poll_rate_hz=100)
    monitor.start()

    grid_results = []
    combo_idx = 0

    for theta_low, theta_high in THETA_COMBOS:
        for alpha in ALPHA_VALUES:
            combo_idx += 1
            combo_label = f"theta({theta_low},{theta_high})_alpha{alpha}"
            print(f"\n{'#' * 95}\n# COMBO {combo_idx}/{total_combos}: theta_low={theta_low}, "
                  f"theta_high={theta_high}, alpha={alpha}\n{'#' * 95}")

            for prompt_label, prompt_text in PROMPTS:
                trials = []
                for trial_idx in range(1, NUM_TRIALS + 1):
                    print(f"  [{combo_label} | {prompt_label} | trial {trial_idx}/{NUM_TRIALS}] ...", end=" ", flush=True)
                    entry, target_model, scout_model, tokenizer, monitor = run_entropy_gated_scout_benchmark(
                        target_model_id=target_model_id,
                        scout_model_id=scout_model_id,
                        prompt=prompt_text,
                        prompt_label=f"{combo_label}_{prompt_label}_trial{trial_idx}",
                        default_K=5,
                        theta_low=theta_low,
                        theta_high=theta_high,
                        alpha=alpha,
                        max_target_tokens=MAX_TOKENS,
                        warmup_steps=WARMUP_STEPS,
                        target_model=target_model,
                        scout_model=scout_model,
                        tokenizer=tokenizer,
                        monitor=monitor,
                    )
                    print(f"tok/s={entry['throughput_tok_sec']}  J/tok={entry['joules_per_token']:.3f}  "
                          f"fallback={entry['fallback_pct']}%")
                    trials.append(entry)

                tps = [t["throughput_tok_sec"] for t in trials]
                jtok = [t["joules_per_token"] for t in trials]
                fallback_pcts = [t["fallback_pct"] for t in trials]
                accept_rates = [t["draft_accept_rate_pct"] for t in trials]

                base = baseline_ref.get(prompt_label, {})
                base_tps = base.get("tps_mean")
                base_jtok = base.get("j_tok_mean")

                agg = {
                    "theta_low": theta_low,
                    "theta_high": theta_high,
                    "alpha": alpha,
                    "prompt": prompt_label,
                    "n_trials": NUM_TRIALS,
                    "tps_mean": float(np.mean(tps)),
                    "tps_std": float(np.std(tps)),
                    "j_tok_mean": float(np.mean(jtok)),
                    "j_tok_std": float(np.std(jtok)),
                    "fallback_pct_mean": float(np.mean(fallback_pcts)),
                    "accept_rate_pct_mean": float(np.mean(accept_rates)),
                    "vs_baseline_speedup_pct": round((float(np.mean(tps)) / base_tps - 1) * 100, 1) if base_tps else None,
                    "vs_baseline_energy_delta_pct": round((float(np.mean(jtok)) / base_jtok - 1) * 100, 1) if base_jtok else None,
                }
                grid_results.append(agg)

    monitor.stop()
    monitor.close()

    out_path = os.path.join(TELEMETRY_DIR, "theta_alpha_grid_results.json")
    with open(out_path, "w") as f:
        json.dump(grid_results, f, indent=2)

    print("\n" + "=" * 95)
    print("GRID COMPLETE")
    print(f"{'Theta':<14} | {'Alpha':<6} | {'Prompt':<8} | {'tok/s':<8} | {'J/tok':<8} | "
          f"{'Fallback%':<10} | {'vs Base tok/s':<14} | {'vs Base J/tok':<14}")
    print("-" * 110)
    for r in grid_results:
        theta_str = f"({r['theta_low']},{r['theta_high']})"
        spd = f"{r['vs_baseline_speedup_pct']:+.1f}%" if r['vs_baseline_speedup_pct'] is not None else "n/a"
        ene = f"{r['vs_baseline_energy_delta_pct']:+.1f}%" if r['vs_baseline_energy_delta_pct'] is not None else "n/a"
        print(f"{theta_str:<14} | {r['alpha']:<6} | {r['prompt']:<8} | {r['tps_mean']:<8.2f} | "
              f"{r['j_tok_mean']:<8.3f} | {r['fallback_pct_mean']:<10.1f} | {spd:<14} | {ene:<14}")
    print("=" * 95)
    print(f"Full grid saved to: {out_path}")


if __name__ == "__main__":
    main()