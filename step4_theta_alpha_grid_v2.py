"""
step4_theta_alpha_grid_v2.py

Follow-up to step4_theta_alpha_grid.py: pushes theta MORE conservative than the
best-performing combo found in the first grid (theta_low=0.35, theta_high=1.00),
to see whether the trend toward smaller losses on Poem/Physics continues,
plateaus, or reverses.

Skips alpha=0.15 -- the first grid showed this can cause ZERO speculative
engagement at tight theta (Poem, theta(0.35,1.0), alpha=0.15: 0% accept rate,
100% fallback), which isn't a useful data point for this follow-up.

Grid: 2 theta combos x 2 alpha values x 3 prompts x N=3 trials = 36 trials.
Reuses the same FP16 baseline reference as the first grid.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import numpy as np
import importlib.util

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TELEMETRY_DIR = os.path.join(REPO_ROOT, "tools", "telemetry")
os.makedirs(TELEMETRY_DIR, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "step4_entropy_gated_scout", os.path.join(REPO_ROOT, "step4_entropy_gated_scout.py")
)
step4_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4_mod)

NVMLPowerMonitor = step4_mod.NVMLPowerMonitor
run_entropy_gated_scout_benchmark = step4_mod.run_entropy_gated_scout_benchmark


THETA_COMBOS = [
    (0.25, 0.85),
    (0.15, 0.70),
]
ALPHA_VALUES = [0.35, 0.65]

PROMPTS = [
    ("Poem", "Write an original Chant Royal poem in English celebrating mathematics."),
    ("Physics", "Explain the physics of semiconductor memory bandwidth and the memory wall."),
    ("Code", "Write a Python implementation of a binary search tree with type annotations."),
]

NUM_TRIALS = 3
MAX_TOKENS = 250
WARMUP_STEPS = 5


def load_baseline_reference():
    path = os.path.join(TELEMETRY_DIR, "fp16_baseline_matched_n5_summary.json")
    if not os.path.exists(path):
        print(f"[!] WARNING: baseline file not found at {path}.")
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
    print("SDSIE THETA x ALPHA GRID -- FOLLOW-UP (tighter theta than best result from grid v1)")
    print(f"Theta combos: {THETA_COMBOS}  (v1 best was (0.35, 1.00))")
    print(f"Alpha values: {ALPHA_VALUES}  (0.15 excluded -- caused zero engagement at tight theta in v1)")
    print(f"Total: {total_trials} trials")
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

    out_path = os.path.join(TELEMETRY_DIR, "theta_alpha_grid_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(grid_results, f, indent=2)

    print("\n" + "=" * 95)
    print("FOLLOW-UP GRID COMPLETE")
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
    print("\nCompare against grid v1's best result: theta(0.35,1.00) -- Poem -6.7 to -7.5%, Physics -10.2 to -10.5%")


if __name__ == "__main__":
    main()
